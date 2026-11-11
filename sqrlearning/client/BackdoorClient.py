import torch
import numpy as np
from torch.utils.data import DataLoader

from .Client import Client


@torch.no_grad()
def evaluate_asr(net, dataset, trigger_fn, target_label, device, batch_size=256):
    was_training = net.training
    net.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hit, total = 0, 0
    for images, _ in loader:  # 标签无关紧要，全部篡改
        images = images.to(device)
        poisoned_images = torch.stack([trigger_fn(img) for img in images])
        preds = net(poisoned_images).argmax(dim=1)
        hit += (preds == target_label).sum().item()
        total += images.size(0)
    if was_training:
        net.train()
    return hit / total if total > 0 else 0.0

class BackdoorClient(Client):
    """
    A basic backdoor attacker
    继承自原有的 Client，在本地训练时进行数据投毒与模型替换。
    参数：
        trigger_fn: 接收张量图像，返回施加触发器后的图像的函数
        target_label: 后门触发后希望被误分类的目标类别
        poison_ratio: 本地数据中被投毒的比例（例如 0.3 表示 30%）
        scale_factor: 上传权重前的放大倍数（用于尝试覆盖其他良性更新）
    """
    def __init__(self, *args, trigger_fn, target_label,
                  poison_ratio=0.3, scale_factor=1.0, **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.scale_factor = scale_factor

    def _poison_batch(self, inputs, labels):
        """对当前 batch 按比例施加触发器并篡改标签"""
        batch_size = inputs.size(0)
        n_poison = int(batch_size * self.poison_ratio)
        if n_poison == 0:
            return inputs, labels

        # 随机选择要投毒的样本索引
        poison_idx = np.random.choice(batch_size, n_poison, replace=False)
        
        poisoned_inputs = inputs.clone()
        poisoned_labels = labels.clone()

        for idx in poison_idx:
            # 施加触发器
            poisoned_inputs[idx] = self.trigger_fn(poisoned_inputs[idx])
            # 篡改为目标标签
            poisoned_labels[idx] = self.target_label
            
        return poisoned_inputs, poisoned_labels

    def train(self, global_weights):
        net = self.net
        net.load_state_dict(global_weights)
        optimizer = self.optim_cls(net.parameters(), **self.optim_kwargs)

        net.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # --- 核心投毒逻辑：在送入模型前混合毒化数据 ---
                if self.poison_ratio > 0:
                    inputs, labels = self._poison_batch(inputs, labels)
                # ------------------------------------------------
                
                optimizer.zero_grad()
                loss = self.criterion(net(inputs), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)

            print(f"{self.name} loss in epoch {epoch}: {total_loss/self.data_size:.3f}")

    def get_weights(self):
        """上传前对权重进行缩放（Model Replacement 策略）"""
        w = super().get_weights()
        if self.scale_factor == 1.0:
            return w
        trainable = {name for name, _ in self.net.named_parameters()}
        return {
            k: (v * self.scale_factor if k in trainable else v)
            for k, v in w.items()
        }
