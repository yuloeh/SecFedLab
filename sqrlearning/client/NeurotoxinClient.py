import torch

from .BackdoorClient import BackdoorClient


class NeurotoxinClient(BackdoorClient):
    """
    Neurotoxin: 在 BackdoorClient 的投毒训练基础上，
    对梯度施加 GradMask，只保留梯度幅值较小的坐标。

    gradmask_ratio: 被屏蔽的 top-k 比例（0~1）。
        - 0.95 表示屏蔽掉梯度中最大的 95% 坐标，只保留最小的 5%。
        - 1.0 表示不使用 GradMask，退化为普通 BackdoorClient。
    """

    def __init__(self, *args, gradmask_ratio: float = 0.95, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= gradmask_ratio <= 1.0:
            raise ValueError("gradmask_ratio must be in [0, 1]")
        self.gradmask_ratio = gradmask_ratio

    def _apply_gradmask(self, net):
        """把所有可训练参数的梯度展平，屏蔽 top-k 坐标，再写回"""
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if not grads:
            return

        # 全局展平：在所有参数坐标上统一算 top-k，而不是每层单独算
        flat = torch.cat([g.view(-1) for g in grads])
        k = int(len(flat) * self.gradmask_ratio)
        if k <= 0:
            return

        # 找到 top-k 的阈值，把绝对值 >= 阈值的坐标置零
        threshold = flat.abs().topk(k).values.min()

        offset = 0
        for g in grads:
            numel = g.numel()
            mask = flat[offset:offset + numel].abs() < threshold
            g.mul_(mask.view_as(g).to(g.dtype))
            offset += numel

    def train(self, global_weights):
        net = self.net
        net.load_state_dict(global_weights)
        optimizer = self.optim_cls(net.parameters(), **self.optim_kwargs)

        net.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.poison_ratio > 0:
                    inputs, labels = self._poison_batch(inputs, labels)

                optimizer.zero_grad()
                loss = self.criterion(net(inputs), labels)
                loss.backward()

                # Neurotoxin 的唯一改动：在 step 之前屏蔽 top-k 梯度
                if self.gradmask_ratio < 1.0:
                    self._apply_gradmask(net)

                optimizer.step()
                total_loss += loss.item() * labels.size(0)

            print(f"{self.name} loss in epoch {epoch}: {total_loss / self.data_size:.3f}")