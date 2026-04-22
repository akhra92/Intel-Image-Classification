import torch
import config as cfg


def train(x, y, model, criterion, optimizer):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)

    predicted = torch.argmax(pred, dim=1)
    acc = (predicted == y).float().mean()

    loss.backward()
    optimizer.step()

    return acc.item(), loss.item()