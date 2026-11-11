import numpy as np
from torch import nn
from torch.utils.data import  DataLoader

def train(server, clients, frac=1.0):
    global_weights = server.get_weights()

    num_clients = len(clients)
    m = max(int(frac * num_clients), 1)

    idxs_users = np.random.choice(range(num_clients), m, replace=False)
    selected = [clients[idx] for idx in idxs_users]
    print(idxs_users, "in this round")

    for client in selected:
        client.train(global_weights)
    
    server.aggregate(selected)

def evaluate(net, dataset, device, batch_size=256, criterion=None):
    """
    返回 (accuracy, average_loss)，均为 Python float。
    不修改 net 的 training 状态——内部保存并恢复。
    """
    was_training = net.training
    net.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if was_training:
        net.train()          # 恢复调用前的模式，避免影响后续训练

    return correct / total, loss_sum / total