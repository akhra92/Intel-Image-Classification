import matplotlib.pyplot as plt
import random
from torchvision.transforms import functional
import torchvision.transforms as T
from torchvision import datasets
import config as cfg


def plot(num_images, cols=4):
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
    plt.show()
