import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from logger import logging
from sklearn.model_selection import train_test_split
import time

from data.CustomDataset import CustomTensorDataset
from data.Transformation import ImgAugTransformation

def get_dataloaders(train_dir, test_dir, train_transform=None, val_transform = None, test_transform = None, val_size = 0.2, batch_size = 64):
    """
    Returns the train, val and test loaders
    """
    t0 = time.time()

    # craete datasets
    dataset_train = np.genfromtxt(train_dir, delimiter=',', skip_header=1)
    dataset_test = np.genfromtxt(test_dir, delimiter=',', skip_header=1)

    # extract features and train
    images_train, labels_train = dataset_train[:,1:].astype(np.float32).reshape(-1,1,28,28), dataset_train[:,0].astype(np.int64).reshape(-1,1)
    images_test, fake_labels_test = dataset_test.astype(np.float32).reshape(-1,1,28,28), np.zeros((len(dataset_test),1)).astype(np.int64).reshape(-1,1)

    # split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(images_train, labels_train, test_size = val_size, random_state=0)

    logging.info(f'Train samples={len(X_train)}, Val samples={len(X_val)}, Test samples={len(images_test)}')

    # converts into tensors
    featuresTrain, targetsTrain = torch.from_numpy(X_train), torch.from_numpy(y_train)
    featuresVal, targetsVal = torch.from_numpy(X_val), torch.from_numpy(y_val)
    featuresTest, fakeTargetsTest = torch.from_numpy(images_test), torch.from_numpy(fake_labels_test)

    # Dataset w/o any tranformations
    train_dataset = CustomTensorDataset(tensors=(featuresTrain, targetsTrain.squeeze_()), transform=train_transform)
    val_dataset = CustomTensorDataset(tensors=(featuresVal, targetsVal.squeeze_()), transform= val_transform)
    test_dataset = CustomTensorDataset(tensors=(featuresTest, fakeTargetsTest.squeeze_()), transform= test_transform)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    logging.info(f'Loader ready in {time.time()-t0} seconds')
    return train_loader, val_loader, test_loader
