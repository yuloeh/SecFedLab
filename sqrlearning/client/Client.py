import torch
from torch import nn
from torch.utils.data import DataLoader

class Client(object):
    def __init__(self, name: str, dataset, 
                  net_cls: type[nn.Module], *,
                  net_para: dict = {},
                  batch_size: int, num_epochs: int,
                  device = torch.device('cpu'),
                  criterion = None,
                  optim_cls = torch.optim.SGD,
                  **optim_kwargs,
                ):
        self.name = name
        self.train_data = dataset
        self.net = net_cls(**net_para).to(device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        total_data = len(self.train_data)
        self.__data_size = (total_data // self.batch_size) * self.batch_size if self.loader.drop_last else total_data
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optim_cls = optim_cls
        self.optim_kwargs = {'lr': 0.01, **optim_kwargs}

    @property
    def data_size(self):
        return self.__data_size
    
    def train(self, global_weights):
        net = self.net
        net.load_state_dict(global_weights)
        optimizer = self.optim_cls(self.net.parameters(), **self.optim_kwargs)

        net.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits  = net(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
            
            print(f"{self.name} loss in epoch {epoch}: {total_loss/self.data_size:.3f}")
    
    def get_weights(self):
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}