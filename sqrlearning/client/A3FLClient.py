import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader

from .Client import Client

@torch.no_grad()
def evaluate_asr_a3fl(net, dataset, delta, target_label, device, batch_size=256):
    net.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hit, total = 0, 0
    for images, _ in loader:
        images = images.to(device)
        poisoned = torch.clamp(images + delta.unsqueeze(0), -3.0, 3.0)
        preds = net(poisoned).argmax(dim=1)
        hit += (preds == target_label).sum().item()
        total += images.size(0)
    return hit / total

class A3FLClient(Client):
    """
    A3FL：全局统一触发器 δ，每轮用双层优化更新。
    δ 是 nn.Parameter，跨轮保留；所有样本共享同一个 δ。
    """

    def __init__(self, *args,
                 target_label: int,
                 poison_ratio: float = 0.3,
                 trigger_size: int = 3,
                 trigger_lr: float = 0.01,          # PGD step size
                 trigger_steps: int = 50,            # 每轮 PGD 迭代步数
                 adv_steps: int = 10,                # 对抗性模型训练步数
                 adv_lr: float = 0.01,
                 lambda0: float = 0.5,               # λ 基数
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.trigger_lr = trigger_lr
        self.trigger_steps = trigger_steps
        self.adv_steps = adv_steps
        self.adv_lr = adv_lr
        self.lambda0 = lambda0

        # 全局统一触发器：一个可优化的 (C, H, W) 张量，初始为白色方块
        C = next(iter(self.loader))[0].shape[1]
        H = W = next(iter(self.loader))[0].shape[2]
        delta_init = torch.zeros(C, H, W, device=self.device)
        delta_init[:, :trigger_size, :trigger_size] = 2.82   # MNIST 归一化后白色
        self.delta = nn.Parameter(delta_init)

    def _apply_trigger(self, x):
        """所有样本共享同一个 δ：直接广播相加，再 clamp 防止越界"""
        return torch.clamp(x + self.delta.unsqueeze(0), -0.4242, 2.8215)

    def _optimize_trigger(self, global_model):
        """A3FL 双层优化：对抗性模型 + PGD 更新 δ"""
        global_model.eval()

        # ---------- 内层：构造对抗性模型 θ' ----------
        adv_model = copy.deepcopy(global_model)
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=self.adv_lr)

        for _ in range(self.adv_steps):
            x, y = next(iter(self.loader))
            x, y = x.to(self.device), y.to(self.device)
            x_poison = self._apply_trigger(x)
            loss = F.cross_entropy(adv_model(x_poison), y)   # 真实标签 → 遗忘后门
            adv_opt.zero_grad()
            loss.backward()
            adv_opt.step()

        # ---------- 外层：PGD 优化 δ ----------
        # λ = λ₀ · cosine_similarity(θ', θ)
        with torch.no_grad():
            cos = F.cosine_similarity(
                torch.cat([p.view(-1) for p in adv_model.parameters()]),
                torch.cat([p.view(-1) for p in global_model.parameters()]),
                dim=0,
            ).item()
        lam = self.lambda0 * max(cos, 0.0)

        opt_delta = torch.optim.SGD([self.delta], lr=self.trigger_lr)

        for _ in range(self.trigger_steps):
            x, _ = next(iter(self.loader))
            x = x.to(self.device)
            x_poison = self._apply_trigger(x)
            target = torch.full((x.size(0),), self.target_label,
                                dtype=torch.long, device=self.device)

            loss_global = F.cross_entropy(global_model(x_poison), target)
            loss_adv = F.cross_entropy(adv_model(x_poison), target)
            loss = loss_global + lam * loss_adv

            opt_delta.zero_grad()
            loss.backward()
            opt_delta.step()

        self.delta.data.clamp_(-3.0, 3.0)   # 保持扰动在合理范围

    @torch.no_grad()
    def _poison_batch(self, inputs):
        return self._apply_trigger(inputs)

    def train(self, global_weights):
        net = self.net
        net.load_state_dict(global_weights)

        # Step 1: A3FL 核心——每轮先优化全局统一触发器 δ
        self._optimize_trigger(net)

        # Step 2: 用优化后的 δ 投毒 + 本地训练
        optimizer = self.optim_cls(net.parameters(), **self.optim_kwargs)
        net.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.poison_ratio > 0:
                    n_poison = int(inputs.size(0) * self.poison_ratio)
                    if n_poison > 0:
                        idx = torch.randperm(inputs.size(0), device=self.device)[:n_poison]
                        inputs = inputs.clone()
                        labels = labels.clone()
                        inputs[idx] = self._poison_batch(inputs[idx])
                        labels[idx] = self.target_label

                optimizer.zero_grad()
                loss = self.criterion(net(inputs), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)

            print(f"{self.name} loss in epoch {epoch}: {total_loss / self.data_size:.3f}")

    def get_weights(self):
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}