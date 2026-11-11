import torch

from ..client import Client
from .Server import Server

class TrimmedMeanServer(Server):
    def __init__(self, net, num_attackers, device):
        super().__init__(net, device=device)
        self.num_attackers = num_attackers

    def aggregate(self, clients: list[Client]) -> None:
        assert len(clients) > 0
        local_weights = [c.get_weights() for c in clients]
        # trim_ratio = self.num_attackers / len(clients)
        # trim_num = int(trim_ratio * len(local_weights))
        trim_num = self.num_attackers
        assert 2 * trim_num < len(local_weights), "trim_num must be less than half of clients"

        w_trmean = {}
        for k, ref in local_weights[0].items():
            if ref.dim() == 0:
                # 标量 buffer：不参与 trimmed mean，保留第一个客户端的值
                w_trmean[k] = ref.detach().clone()
                continue
            stacked = torch.stack([wi[k].reshape(-1) for wi in local_weights], dim=0)   # (n, D)
            sorted_vals = stacked.sort(dim=0).values
            kept = sorted_vals[trim_num:len(local_weights) - trim_num]                   # (n-2*trim, D)
            w_trmean[k] = kept.mean(dim=0).reshape(ref.shape)

        self.net.load_state_dict(w_trmean)
