import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset

def iid_sample(dataset, num_clients):
    assert 0 < num_clients <= len(dataset)
    N = len(dataset)
    idx = np.random.permutation(N).tolist()
    num_items = N // num_clients
    clients_idx = []
    for i in range(num_clients):
        start = i * num_items
        end = start + num_items if i < num_clients - 1 else N  # 最后一个客户端吃掉余数
        clients_idx.append(idx[start:end])

    return [Subset(dataset, ci) for ci in clients_idx]

def mnist_data(num_clients):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../Datasets/mnist/', train=True, download=True, transform=trans_mnist)
    test_data = datasets.MNIST('../Datasets/mnist/', train=False, download=True, transform=trans_mnist)
    clients_dataset = iid_sample(train_data, num_clients)
    
    return (train_data, test_data, clients_dataset)