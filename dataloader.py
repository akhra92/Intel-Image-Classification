from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
import config as cfg
import os


def get_transforms(train=True):

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return T.Compose([T.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
                          T.RandomHorizontalFlip(p=0.3),
                          T.ToTensor(),
                          normalize])
    else:
        return T.Compose([T.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
                          T.ToTensor(),
                          normalize])
    

def get_loaders():
    
    trn_transforms = get_transforms(train=True)
    val_transforms = get_transforms(train=False)

    trn_data_dir = os.path.join(cfg.ROOT, 'seg_train/seg_train/')
    val_data_dir = os.path.join(cfg.ROOT, 'seg_test/seg_test/')

    trn_dataset = datasets.ImageFolder(root=trn_data_dir, transform=trn_transforms)
    val_dataset = datasets.ImageFolder(root=val_data_dir, transform=val_transforms)

    pin = cfg.DEVICE == 'cuda'
    trn_loader = DataLoader(dataset=trn_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=pin)

    return trn_loader, val_loader