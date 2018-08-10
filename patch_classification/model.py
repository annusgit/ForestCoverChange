
from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from dataset import get_dataloaders


class VGG(nn.Module):
    """
        Get a pretrained VGG network (on ImageNet) and try to finetune it on EuroSat images
        Reported acc is > 98% on Resnet-50, let's see what can we get from a VGG network
    """

    def __init__(self, in_channels):
        super(VGG, self).__init__()
        graph = models.vgg11_bn(pretrained=True)
        # graph.load_state_dict(torch.load('/home/annus/.torch/models/vgg11_bn-6002323d.pth'))
        graph_layers = list(graph.features) # only get the feature extractor, we don't need the classifier
        # add a lot of dropout to make it generalize better...
        graph_layers.insert(8, nn.Dropout2d(p=0.7))
        graph_layers.insert(16, nn.Dropout2d(p=0.7))
        graph_layers.insert(24, nn.Dropout2d(p=0.7))
        graph_layers.insert(33, nn.Dropout2d(p=0.7))
        new_graph = [] #[nn.BatchNorm2d(num_features=in_channels)] # will be applied at input
        for layer in graph_layers:
            new_graph.append(layer)
        model_list = nn.ModuleList(new_graph)
        self.feature_extracter = nn.Sequential(*model_list)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*((64//2**5)**2), out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=128, out_features=10),
            nn.LogSoftmax(dim=0)
        )
        self.feature_extracter[1].momentum = 0.25
        self.feature_extracter[5].momentum = 0.25
        self.feature_extracter[10].momentum = 0.25
        self.feature_extracter[13].momentum = 0.25
        self.feature_extracter[18].momentum = 0.25
        self.feature_extracter[21].momentum = 0.25
        self.feature_extracter[26].momentum = 0.25
        self.feature_extracter[29].momentum = 0.25

    def forward(self, x):
        x = self.feature_extracter(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x, torch.argmax(input=x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def network():
    in_channels = 3
    patch_size = 64
    vgg = VGG(in_channels=in_channels)
    x = torch.Tensor(2, in_channels, patch_size, patch_size)
    print(vgg)
    print('We need to find {} numbers!'.format(vgg.count_parameters()))
    out = vgg(x)
    print(out.shape)


if __name__ == '__main__':
    network()
















