

from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import matplotlib.pyplot as pl
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from dataset import get_dataloaders
from torchsummary import summary
from torchviz import make_dot


class MODEL(nn.Module):
    def __init__(self, in_channels):
        super(MODEL, self).__init__()
        drop_rate = 0.5

        def block(channels_in, channels_out):
            return nn.Sequential(
                nn.Conv1d(in_channels=channels_in, out_channels=channels_out, kernel_size=1),
                nn.BatchNorm1d(num_features=channels_out, eps=1e-4, momentum=0.2),
                nn.ReLU(),
                nn.Dropout(drop_rate))

        self.feature_classifier = nn.Sequential(
            block(channels_in=in_channels, channels_out=16),
            block(channels_in=16, channels_out=32),
            block(channels_in=32, channels_out=64),
            block(channels_in=64, channels_out=32),
            nn.Conv1d(in_channels=32, out_channels=5, kernel_size=1),
        )
        pass

    def forward(self, *input):
        x, = input
        x = self.feature_classifier(x).squeeze(2)
        return x, torch.argmax(input=x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def check_1dnet():
    model = MODEL(in_channels=13)
    print('total parameters = {}'.format(model.count_parameters()))
    test_input = torch.Tensor(16, 13, 1)
    model.eval()
    output, max = model(test_input)
    print(output.shape, max.shape)
    pass


@torch.no_grad()
def check_on_dataloader():
    model = MODEL(in_channels=13)
    model.eval()
    train, eval, test = get_dataloaders(path_to_nparray='/home/annus/Desktop/signatures/all_signatures.npy',
                                        batch_size=128)
    for idx, data in enumerate(train):
        examples, labels = data
        output, pred = model(examples)
        print('on batch {}/{}, {}'.format(idx + 1, len(train), output.size()))

    pass


if __name__ == '__main__':
    check_on_dataloader()















