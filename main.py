import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import timm
import numpy as np
import config as cfg
from train import train
from test import test
from dataloader import get_loaders, get_transforms
from utils import plot
from tqdm import tqdm


trn_loader, val_loader = get_loaders()
model = timm.create_model(model_name='resnext150', pretrained=True, num_classes=6).to(cfg.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.params(), lr=cfg.LR, weight_decay=5e-3)

def main():
    trn_loss, val_loss = [], []
    trn_acc, val_acc = [], []
    for epoch in range(cfg.NUM_EPOCHS):
        trn_epoch_loss, val_epoch_loss = [], []
        trn_epoch_acc, val_epoch_acc = [], []
        print(f'Epoch: {epoch+1}/{cfg.NUM_EPOCHS}')
        for batch in tqdm(trn_loader, desc='Training'):
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            acc, loss = train(x, y, model, criterion, optimizer)
            
