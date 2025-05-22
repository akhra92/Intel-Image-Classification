import torch


ROOT = './dataset/Intel/'
BATCH_SIZE = 32
LR = 1e-3
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NUM_WORKERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
