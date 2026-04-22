import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import timm
import numpy as np
import config as cfg
from train import train
from test import test
from dataloader import get_loaders
from utils import plot_samples, plot_curves
from tqdm import tqdm


def main():
    trn_loader, val_loader = get_loaders()
    model = timm.create_model(model_name='resnext101_32x8d', pretrained=True, num_classes=6).to(cfg.DEVICE)

    # Freeze all layers, then unfreeze the last block and classifier head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_val_acc = 0.0
    patience = cfg.PATIENCE
    count = 0
    trn_loss, val_loss = [], []
    trn_acc, val_acc = [], []
    for epoch in range(cfg.NUM_EPOCHS):
        trn_epoch_loss, val_epoch_loss = [], []
        trn_epoch_acc, val_epoch_acc = [], []
        print(f'Epoch: {epoch+1}/{cfg.NUM_EPOCHS}')
        model.train()
        for batch in tqdm(trn_loader, desc='Training'):
            x, y = batch
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            acc, loss = train(x, y, model, criterion, optimizer)
            trn_epoch_acc.append(acc)
            trn_epoch_loss.append(loss)
        model.eval()
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

        scheduler.step(val_epoch_acc)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            count = 0
            print(f'Saving the model at epoch {epoch+1} with val accuracy {best_val_acc}')
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            count += 1
            print(f'No improvement for {count} times.')
            if count > patience:
                break
        
        trn_acc.append(trn_epoch_acc)
        trn_loss.append(trn_epoch_loss)
        val_acc.append(val_epoch_acc)
        val_loss.append(val_epoch_loss)

    return trn_acc, trn_loss, val_acc, val_loss


if __name__ == '__main__':
    plot_samples(20)
    trn_acc, trn_loss, val_acc, val_loss = main()
    plot_curves(trn_acc, trn_loss, val_acc, val_loss)