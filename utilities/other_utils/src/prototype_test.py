

"""
    A small prototype net to test the performance of segmentation
"""

from __future__ import print_function
from __future__ import division
import os
import pickle
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.nn import functional as F
from torch.optim import Adam


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        # we shall define our architecture here
        self.half_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        )
        self.half_2 = nn.Sequential(
            nn.Linear(in_features=16*60*60, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10),
            nn.Softmax(dim=1)
        )
        pass

    def forward(self, input):
        x = self.half_1(input)
        x = x.view(-1, self.flat_features(x))
        return self.half_2(x)

    def flat_features(self, x):
        feats = x.size()[1:]
        flat_feats = 1
        for s in feats:
            flat_feats *= s
        return flat_feats

    def train_model(self):

        pass

    def test_model(self):

        pass


def main():
    path = '/home/annus/Desktop/forest_cover_change/region_3_data/malakand/data/' \
           'espa-annuszulfiqar@gmail.com-07012018-001522-723/training_data'
    with open(os.path.join(path, 'data.pkl'), 'rb') as data:
        training_data = torch.Tensor(pickle.load(data)).float()
        training_data = training_data.view(-1, 3, 60, 60)
    with open(os.path.join(path, 'labels.pkl')) as labels:
        targets = torch.Tensor(pickle.load(labels)).long()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    print('training data is {} and {}'.format(training_data.shape, targets.shape))
    net = NN()
    # print(list(net.parameters())[0].size())
    output = net(training_data[0:4,:,:])
    print(output.size())


if __name__ == '__main__':
    main()



















