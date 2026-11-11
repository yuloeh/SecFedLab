import torch
from torch import nn
from abc import ABC, abstractmethod

from ..client import Client

class Server(ABC):

    def __init__(self, net: nn.Module, *,
                  device = torch.device('cpu'),
                ):
        self.net = net.to(device)
        self.device = device

    @abstractmethod
    def aggregate(self, clients: list[Client]) -> None: ...

    def get_weights(self):
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}
