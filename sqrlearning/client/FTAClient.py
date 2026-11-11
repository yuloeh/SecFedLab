import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .Client import Client

@torch.no_grad()
def evaluate_asr_fta(net, dataset, generator, target_label, eps, device, batch_size=256):
    net.eval()
    generator.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hit, total = 0, 0
    for images, _ in loader:
        images = images.to(device)
        delta = generator(images)
        delta = delta * torch.clamp(eps / (delta.view(delta.size(0), -1).norm(dim=1).view(-1,1,1,1) + 1e-8), max=1.0)
        poisoned = torch.clamp(images + delta, -3.0, 3.0)
        preds = net(poisoned).argmax(dim=1)
        hit += (preds == target_label).sum().item()
        total += images.size(0)
    return hit / total

class TriggerGenerator(nn.Module):
    """轻量自编码器：输出与输入同形状的扰动，L2 范数由外部裁剪"""
    def __init__(self, in_channels=1, base=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base, in_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class FTAClient(Client):
    def __init__(self, *args,
                 target_label: int,
                 poison_ratio: float = 0.1,       # 论文默认 10%
                 eps: float = 8/255,              # 扰动 L2 范数上界
                 generator_lr: float = 1e-3,
                 generator_steps: int = 50,       # 每轮优化生成器的步数
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.eps = eps
        self.generator_lr = generator_lr
        self.generator_steps = generator_steps

        # 生成器是攻击者私有的，不参与联邦聚合
        self.generator = TriggerGenerator(
            in_channels=next(iter(self.loader))[0].shape[1]
        ).to(self.device)

    def _project_l2(self, delta, eps):
        """把扰动裁剪到 L2 球内：delta * min(1, eps/||delta||)"""
        norm = delta.view(delta.size(0), -1).norm(dim=1).view(-1, 1, 1, 1)
        scale = torch.clamp(eps / (norm + 1e-8), max=1.0)
        return delta * scale

    def _update_generator(self, global_model):
        """固定全局模型，优化生成器，使扰动样本被分类为目标类"""
        global_model.eval()
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr)

        for _ in range(self.generator_steps):
            # 从本地 loader 取一个 batch 的干净样本
            x, _ = next(iter(self.loader))
            x = x.to(self.device)

            delta = self.generator(x)
            delta = self._project_l2(delta, self.eps)
            x_adv = torch.clamp(x + delta, -3.0, 3.0)   # 防止像素越界

            logits = global_model(x_adv)
            target = torch.full((x.size(0),), self.target_label,
                                dtype=torch.long, device=self.device)
            loss = F.cross_entropy(logits, target)

            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()

        self.generator.eval()

    @torch.no_grad()
    def _poison_batch(self, inputs):
        """用当前生成器对 batch 施加扰动，返回扰动后的图像"""
        delta = self.generator(inputs)
        delta = self._project_l2(delta, self.eps)
        return torch.clamp(inputs + delta, -3.0, 3.0)

    def train(self, global_weights):
        net = self.net
        net.load_state_dict(global_weights)

        # Step 1: 先用全局模型更新生成器（FTA 的关键：生成器每轮适应全局模型）
        self._update_generator(net)

        # Step 2: 固定生成器，做投毒训练
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
