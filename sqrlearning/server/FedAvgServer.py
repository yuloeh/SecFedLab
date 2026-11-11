from ..client import Client
from .Server import Server

class FedAvgServer(Server):
    def aggregate(self, clients: list[Client]) -> None:
        assert len(clients) > 0
        local_weights = [c.get_weights() for c in clients]
        num_samples = [c.data_size for c in clients]
        total = sum(num_samples)
        weights = [ni / total for ni in num_samples]
        w_avg = {
            k: sum(local_weights[i][k] * weights[i] for i in range(len(local_weights)))
            for k in local_weights[0].keys()
        }
        self.net.load_state_dict(w_avg)
