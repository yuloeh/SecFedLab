import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from .BackdoorClient import BackdoorClient, evaluate_asr

class PonderClient(BackdoorClient):

    def __init__(self, *args, epoch_multiplier=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_multiplier = max(1.0, epoch_multiplier)
        eval_asr_size = 300
        n = min(eval_asr_size, len(self.train_data))
        self.eval_asr_data = Subset(self.train_data, list(range(n)))

    def train(self, global_weights):
        net = self.net
        net.load_state_dict(global_weights)
        optimizer = self.optim_cls(net.parameters(), **self.optim_kwargs)

        max_epochs = int(self.epoch_multiplier * self.num_epochs)
        min_epochs = self.num_epochs

        net.train()
        for epoch in range(max_epochs):
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

            asr = evaluate_asr(self.net, self.eval_asr_data, self.trigger_fn, self.target_label, self.device)
            print(f"{self.name} loss in epoch {epoch}: {total_loss/self.data_size:.3f}, asr: {100*asr:.2f}%")
            if epoch >= min_epochs - 1 and asr > 0.9:
                break
