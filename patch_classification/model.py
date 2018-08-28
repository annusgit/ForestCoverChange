
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


class ResNet(nn.Module):
    """
        Get a pretrained VGG network (on ImageNet) and try to finetune it on EuroSat images
        Reported acc is > 98% on Resnet-50, let's see what can we get from a VGG network
    """

    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        graph = models.resnet50(pretrained=True)
        removed = list(graph.children())[:-2]
        with_dropout = []
        with_dropout.append(removed[0])
        with_dropout.append(removed[1])
        with_dropout.append(removed[2])
        with_dropout.append(removed[3])
        for part in removed[4:]:
            with_dropout.append(part)
            with_dropout.append(nn.Dropout2d(p=0.8))
        # print(with_dropout)
        self.feature_extracter = torch.nn.Sequential(*with_dropout)
        self.kill = nn.Dropout(p=0.8)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=32768, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        x = self.feature_extracter(x)
        x = self.kill(x)
        # np.savetxt('this.txt', x.detach().cpu().numpy())
        x = self.classifier(x.view(x.size(0), -1))
        return x, torch.argmax(input=x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def see_children_recursively(graph):
    further = False
    children = list(graph.children())
    for child in children:
        further = True
        see_children_recursively(child)
    if not further and isinstance(graph, nn.BatchNorm2d):
        print(graph)


class HyperSpectral_Resnet(nn.Module):

    def __init__(self, in_channels):
        super(HyperSpectral_Resnet, self).__init__()
        self.Resnet = ResNet(in_channels=-1)
        children = list(self.Resnet.feature_extracter.children())[1:]
        children.insert(0, nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1))
        self.Resnet.feature_extracter = nn.Sequential(*children)
        pass

    def forward(self, x):
        return self.Resnet(x)


def network():
    in_channels = 10
    patch_size = 64
    net = HyperSpectral_Resnet(in_channels=in_channels)
    x = torch.Tensor(1, in_channels, patch_size, patch_size)
    # see_children_recursively(net)
    print(net)
    # summary(model=net, input_size=(in_channels, patch_size, patch_size))
    # print('We need to find {} numbers!'.format(net.count_parameters()))
    out, pred = net(x)
    print(out.shape, pred.shape)


@torch.no_grad()
def check_model_on_dataloader():
    from dataset import get_dataloaders
    model = HyperSpectral_Resnet(in_channels=5)
    model.eval()
    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='/home/annus/Desktop/'
    #                                                                                 'projects/forest_cover_change/'
    #                                                                                 'eurosat/images/tif/',
    #                                                                     batch_size=1)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='tif/',
                                                                        batch_size=64)
    count = 0
    while True:
        count += 1
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            print('{} -> on batch {}/{}, {}'.format(count, idx + 1, len(train_dataloader), examples.size()))
            if True:
                out_x, pred = model(examples)
                print(out_x.shape, pred.shape)
    pass


if __name__ == '__main__':
    check_model_on_dataloader()
















