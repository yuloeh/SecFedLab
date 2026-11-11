from opacus.utils.batch_memory_manager import BatchMemoryManager

from .DPClient import DPClient

class DPClient(DPClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _blend(self, global_weights):
        if self.local_state is None:
            return {k: v.detach().clone() for k, v in global_weights.items()}
        blended = {}
        for k in global_weights:
            g = global_weights[k]
            l = self.local_state[k]
            if g.is_floating_point():
                blended[k] = (1 - self.local_blend) * g + self.local_blend * l
            else:
                blended[k] = g.detach().clone()
        return blended
        
    def train(self, global_weights):
        blended = self._blend(global_weights)

        net = self.net
        net.load_state_dict(blended)

        net.train()
        for epoch in range(self.num_epochs):
            with BatchMemoryManager(
                data_loader=self.loader, 
                max_physical_batch_size=self.MAX_PHYSICAL_BATCH_SIZE,
                optimizer=self.optimizer
            ) as memory_safe_data_loader:
                for inputs, labels in memory_safe_data_loader: 
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.criterion(net(inputs), labels)
                    loss.backward()
                    self.optimizer.step()

        eps = self.privacy_engine.get_epsilon(self.target_delta)
        print(f"{self.name}: ε={eps:.3f} (target={self.target_epsilon}), "
                f"δ={self.target_delta:g}")
