import matplotlib.pyplot as plt
import random
from torchvision.transforms import functional
import torchvision.transforms as T
from torchvision import datasets
import config as cfg


def plot_samples(num_images, cols=4):
    dataset = datasets.ImageFolder(root=cfg.ROOT + 'seg_test/seg_test/', transform=T.ToTensor())
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(cols*3, rows*3))
    indices = random.sample(range(len(dataset)), num_images)
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        plt.subplot(rows, cols, i+1)
        x = functional.to_pil_image(x)
        plt.imshow(x)
        plt.title(dataset.classes[y])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('assets/samples.png')
    plt.show()


def plot_curves(trn_acc, trn_loss, val_acc, val_loss):

    epochs = range(1, len(trn_acc) + 1)
    plt.figure(figsize=(12, 5))    
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trn_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, trn_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
