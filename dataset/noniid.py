import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset

def noniid_sample(dataset, num_clients, alpha, seed=None, shuffle_seed=None):

    rand_distr = np.random.default_rng(seed)
    rand_shuffle = random.Random(shuffle_seed)

    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [dataset[i][1] for i in range(len(dataset))]

    classes = {}
    for idx, label in enumerate(targets):
        classes.setdefault(int(label), []).append(idx)

    clients_idx = [[] for _ in range(num_clients)]

    for idxs in classes.values():
        rand_shuffle.shuffle(idxs)
        class_size = len(idxs)

        probs = rand_distr.dirichlet([alpha] * num_clients)
        counts = (probs * class_size).astype(int)          # floor，不会超取

        # 余数向量化分配：多项分布保证 counts 总和恰好等于 class_size
        remainder = class_size - int(counts.sum())
        if remainder > 0:
            counts += rand_distr.multinomial(
                remainder, np.full(num_clients, 1.0 / num_clients)
            )

        start = 0
        for i in range(num_clients):
            end = start + int(counts[i])
            clients_idx[i].extend(idxs[start:end])
            start = end

    for i in range(num_clients):
        rand_shuffle.shuffle(clients_idx[i])

    return [Subset(dataset, ci) for ci in clients_idx]

def mnist_data(num_clients, *, alpha, seed=None, shuffle_seed=None):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../Datasets/mnist/', train=True, download=True, transform=trans_mnist)
    test_data = datasets.MNIST('../Datasets/mnist/', train=False, download=True, transform=trans_mnist)
    clients_dataset = noniid_sample(train_data, num_clients, alpha, seed, shuffle_seed)
    
    return (train_data, test_data, clients_dataset)