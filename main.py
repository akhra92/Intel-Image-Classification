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
model = timm.create_model(model_name='resnext101_32x8d', pretrained=True, num_classes=6).to(cfg.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=5e-3)
best_val_acc = 0.0
best_val_acc_es = 0.0
patience = 3


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
            trn_epoch_acc.append(acc)
            trn_epoch_loss.append(loss)
        for batch in tqdm(val_loader, desc='Validation'):
            x, y = batch
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            acc, loss = test(x, y, model, criterion)
            val_epoch_acc.append(acc)
            val_epoch_loss.append(loss)
        
        trn_epoch_acc = np.mean(trn_epoch_acc)
        trn_epoch_loss = np.mean(trn_epoch_loss)
        val_epoch_acc = np.mean(val_epoch_acc)
        val_epoch_loss = np.mean(val_epoch_loss)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            print(f'Saving the model at epoch {epoch+1} with val accuracy {best_val_acc}')
            torch.save(model.state_dict(), 'best_model.pth')
        
        if val_epoch_acc > best_val_acc_es:
            best_val_acc_es = val_epoch_acc
            count = 0
        else:
            count += 1
            print(f'No improvement for {count} times.')
            if count > patience:
                break
        
        trn_acc.append(trn_epoch_acc)
        trn_loss.append(trn_epoch_loss)
        val_acc.append(val_epoch_acc)
        val_loss.append(val_epoch_loss)


if __name__ == '__main__':
    plot(20)
    main()