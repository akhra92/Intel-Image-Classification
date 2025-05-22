import torch.nn as nn
import config as cfg
import torch
import torchvision
import numpy as np


def test(x, y, model, criterion):
    model.eval()
    pred = model(x)
    loss = criterion(pred, y)
    
    predicted = torch.argmax(pred, dim=1)
    acc = (predicted == y).float().mean()

    return loss.item(), acc.item()