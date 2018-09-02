

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as pl


class judge(nn.Module):

    def __init__(self):
        super(judge, self).__init__()
        # in_channels, out_channels, kernel_size, padding for conv2d
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.max = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.fc = nn.Linear(in_features=6272, out_features=2)
        self.kill = nn.Dropout2d(p=0.8)
        self.act = nn.ReLU()


    def forward(self, *input):
        x, = input
        d = self.act(self.bn1(self.conv1(x)))
        d = self.act(self.bn2(self.conv2(d)))
        d = self.kill(self.max(d))
        d = self.act(self.bn3(self.conv3(d)))
        d = self.act(self.bn4(self.conv4(d)))
        d = self.kill(self.max(d))
        d = d.view(-1, 6272)
        d = self.fc(d)
        return F.softmax(d, dim=1)


class intelligence(nn.Module):
    """
        Takes a single input vector of 1 dimensions and upsamples it to 28*28 matrices of images
    """
    def __init__(self):
        super(intelligence, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(),
            nn.Linear(in_features=16, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=784),
            nn.BatchNorm1d(num_features=784),
            nn.LeakyReLU(),
        )

    def forward(self, *input):
        x, = input
        return self.generator(x).view(-1, 1, 28, 28)


def discriminator_tst():
    from data import get_dataloaders
    train, test = get_dataloaders(batch_size=16)
    disc = judge()
    for images, labels in train:
        test_out = disc(images)
        print(test_out.shape)


def gen_tst():
    gen = intelligence()
    for k in range(100):
        noise = torch.Tensor(16, 1)
        gen_images = gen(noise)
        print(gen_images.shape)
        images = gen_images.detach().numpy()
        images = images.squeeze(1).transpose(0, 1, 2)
        print(np.unique(images))
        for j in range(16):
            pl.subplot(4, 4, j + 1)
            # print(images.shape)
            pl.imshow(images[j, :, :])
            pl.axis('off')
        pl.show()
pass


def gan_tst():
    gen, disc = intelligence(), judge()
    for k in range(100):
        noise = torch.Tensor(16, 1)
        gen_images = gen(noise)
        discriminated_probs = disc(gen_images)
        # print(gen_images.shape)
        images, results = gen_images.detach().numpy(), torch.argmax(discriminated_probs, dim=1).detach().numpy()
        images = images.squeeze(1).transpose(0, 1, 2)
        # print(np.unique(images))
        for j in range(16):
            pl.subplot(4, 4, j + 1)
            # print(images.shape)
            pl.imshow(images[j, :, :])
            pl.title(results[j])
            pl.axis('off')
        pl.show()
    pass


if __name__ == '__main__':
    gan_tst()













