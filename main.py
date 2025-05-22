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

trn_transforms = get_transforms(train=True)
val_transforms = get_transforms(train=False)

model = timm.create_model(model_name='resnext150', pretrained=True, num_classes=)