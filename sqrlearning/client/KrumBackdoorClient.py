import torch
from torch import nn

from .Client import Client
from .BackdoorClient import BackdoorClient


class KrumBackdoorClient(BackdoorClient):
    """
    后门攻击 + Krum 规避。
    1. 正常后门训练得到 w_local；
    2. s = w_local - w_Re（含后门方向 + 主任务学习方向）；
    3. 二分搜索最大 λ，使 w_mal = w_Re + λ·s 到 w_Re 的距离不超过阈值。
    λ > 1 时放大后门，λ < 1 时收敛到全局模型附近。
    """

    def __init__(self, name: str, dataset,
                  net_cls: type[nn.Module], *,
                  net_para: dict = {},
                  max_deviation_ratio: float = 0.5,
                  **kwargs
                ):
        super().__init__(name, dataset, net_cls, net_para=net_para, **kwargs)
        self.max_deviation_ratio = max_deviation_ratio

    def _deviation(self, w: dict, w_re: dict) -> float:
        return torch.norm(
            torch.cat([(w[k] - w_re[k]).view(-1) for k in w_re])
        ).item()

    def _global_norm(self, w_re: dict) -> float:
        return torch.norm(
            torch.cat([w_re[k].view(-1) for k in w_re])
        ).item()
    
    def _clean_update_dev(self, global_weights: dict) -> float:
        """做一次干净训练，返回 ||w_clean - w_Re|| 作为良性尺度"""
        Client.train(self, global_weights)
        w_clean = self.get_weights()
        return self._deviation(w_clean, global_weights)

    def train(self, global_weights):
        w_re = global_weights

        # 1. 良性更新尺度
        clean_dev = self._clean_update_dev(w_re)

        # 2. 后门训练
        super().train(w_re)
        w_local = super().get_weights()
        s = {k: w_local[k] - w_re[k] for k in w_re}
        s_norm = torch.norm(
            torch.cat([s[k].view(-1) for k in s])
        ).item()

        if s_norm < 1e-12:
            print(f"{self.name}: s_norm≈0, skip crafting")
            return

        # 3. 阈值 = ratio × clean_dev
        threshold = self.max_deviation_ratio * clean_dev

        # 4. 闭式解 λ = threshold / ||s||
        best_lam = threshold / s_norm

        w_mal = {k: w_re[k] + best_lam * s[k] for k in w_re}
        self.net.load_state_dict(w_mal)
        dev = self._deviation(w_mal, w_re)
        print(f"{self.name}: λ={best_lam:.4f}, deviation={dev:.4f}, "
              f"threshold={threshold:.4f}")

    def get_weights(self):
        return Client.get_weights()