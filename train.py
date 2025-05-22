import torch.nn as nn
import config as cfg
import torch
import torchvision
import numpy as np


def train(x, y, model, criterion, optimizer):
    model.train()
    pred = model(x)
    loss = criterion(pred, y)
    
    predicted = torch.argmax(pred, dim=1)
    acc = (predicted == y).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()