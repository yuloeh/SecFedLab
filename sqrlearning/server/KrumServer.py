import torch
import numpy as np

from ..client import Client
from .Server import Server

def _flatten(state_dict: dict) -> np.ndarray:
    return np.concatenate([
        state_dict[k].detach().cpu().numpy().ravel()
        for k in sorted(state_dict)
    ])

class KrumServer(Server):
    def __init__(self, net, num_attackers, multi_krum: int = 0, **kwargs):
        super().__init__(net, **kwargs)
        self.num_attackers = num_attackers
        self.multi_krum = multi_krum

    def aggregate(self, clients: list[Client]) -> None:
        assert len(clients) > 0
        local_weights = [c.get_weights() for c in clients]
        
        n = len(clients)
        f = self.num_attackers
        assert n >= 2 * f + 3, f"Krum requires n >= 2f+3, got n={n}, f={f}"

        num_neighbors = max(n - f - 2, 1)        # Krum 的 n-f-2
        m = num_neighbors if self.multi_krum == 0 else self.multi_krum
        assert 1 <= m <= n - f - 2, f"m must be in [1, n-f-2] = [1, {n-f-2}], got m={m}"

        grads = np.stack([_flatten(x) for x in local_weights])

        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(grads[i] - grads[j])
                dists[i, j] = dists[j, i] = d

        scores = []
        for i in range(n):
            others = np.delete(dists[i], i)              # 排除自己
            scores.append(np.sort(others)[:num_neighbors].sum())   # 最近 n-f-2 个求和

        order = np.argsort(scores)
        w_agg = local_weights[int(order[0])] if m == 1 else self._average(local_weights, selected=order[:m])
        print(f"Krum: {[clients[i].name for i in order]} (m={m}, scores={[f'{scores[i]:.2f}' for i in order]})")

        self.net.load_state_dict(w_agg)
    
    @staticmethod
    def _average(local_weights: list[dict], selected: np.ndarray) -> dict:
        """对选中的 m 个客户端等权平均；整数 buffer 保留第一个"""
        w_avg = {}
        for k in local_weights[0].keys():
            stacked = torch.stack([local_weights[i][k] for i in selected], dim=0)
            if stacked.dtype.is_floating_point:
                w_avg[k] = stacked.mean(dim=0)
            else:
                # num_batches_tracked 之类的整数 buffer：取第一个，不做平均
                w_avg[k] = stacked[0].clone()
        return w_avg
