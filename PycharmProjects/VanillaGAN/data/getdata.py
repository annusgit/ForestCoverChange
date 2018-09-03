

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as pl


def get_dataloaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('~/PycharmProjects/VanillaGAN/data/raw_data',
                       train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('~/PycharmProjects/VanillaGAN/data/raw_data',
                       train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader













