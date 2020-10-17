from Project import project
from data import get_dataloaders
from data.Transformation import train_transform, val_transform, test_transform
from data.CustomDataset import CustomTensorDataset
from logger import logging

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

train_loader, val_loader, test_loader = get_dataloaders(
    project.data_dir / 'train_template.csv',
    project.data_dir / 'test_template.csv',
    train_transform = None,
    val_transform = None,
    test_transform = None,
    val_size = 0.2,
    batch_size=8)
train_loader_aug, val_loader_aug, test_loader_aug = get_dataloaders(
    project.data_dir / 'train_template.csv',
    project.data_dir / 'test_template.csv',
    train_transform = train_transform,
    val_transform = val_transform,
    test_transform = test_transform,
    val_size = 0.2,
    batch_size=8)
print('loader okk')

def imshow(img, title=''):
    """Plot the image batch.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose( img.numpy(), (1, 2, 0)), cmap='gray')
    plt.show()

# iterate
for i, data in enumerate(train_loader):
    x, y = data
    imshow(torchvision.utils.make_grid(x, 4), title='Normal')
    break  # we need just one batch

for i, data in enumerate(train_loader_aug):
    x, y = data
    imshow(torchvision.utils.make_grid(x, 4), title='Augmented')
    break  # we need just one batch
