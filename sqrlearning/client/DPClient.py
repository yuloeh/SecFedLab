from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from .Client import Client

class DPClient(Client):
    MAX_PHYSICAL_BATCH_SIZE = 64

    def __init__(self, *args, total_rounds,
                 target_epsilon: float, target_delta=1e-5,
                 max_grad_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.total_rounds = total_rounds

        total_epochs = max(1, int(self.num_epochs * self.total_rounds))
        self.privacy_engine = PrivacyEngine()
        self.hooks, self.optimizer, self.loader = \
            self.privacy_engine.make_private_with_epsilon(
                module=self.net,
                optimizer=self.optim_cls(self.net.parameters(), **self.optim_kwargs),
                data_loader=self.loader,
                epochs=total_epochs,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
                wrap_model=False,
            )
    
    def train(self, global_weights):
        net = self.net
        net.load_state_dict(global_weights)

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

    def cleanup(self):
        self.hooks.cleanup()

    @property
    def noise_info(self):
        return f"sigma={self.optimizer.noise_multiplier}"
