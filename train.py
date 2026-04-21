import torch
import config as cfg


def train(x, y, model, criterion, optimizer):
    pred = model(x)
    loss = criterion(pred, y)
    
    predicted = torch.argmax(pred, dim=1)
    acc = (predicted == y).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return acc.item(), loss.item()