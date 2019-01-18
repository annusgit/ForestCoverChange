

"""
    UNet model definition in here
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.model_zoo as model_zoo
from dataset import get_dataloaders_generated_data
import os
import numpy as np
import pickle as pkl
import PIL.Image as Image
import itertools
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torchsummary import summary
from torchvision import models


class UNet_down_block(nn.Module):
    """
        Encoder class
    """
    def __init__(self, input_channel, output_channel):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(nn.Module):
    """
        Decoder class
    """
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.tr_conv_1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.relu = nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.tr_conv_1(x)
        x = self.relu(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.relu(self.bn2(self.conv_2(x)))
        return x


class UNet(nn.Module):

    def __init__(self, input_channels, num_classes, model_dir_path=None):
        super(UNet, self).__init__()
        if model_dir_path:
            # start by loading the pretrained weights from model_dict saved earlier
            with open(model_dir_path, 'rb') as handle:
                model_dict = pkl.load(handle)
                print('log: loaded saved model dictionary')
            print('total number of weights to be loaded into pytorch model =', len(model_dict.keys()))

        self.bn_init = nn.BatchNorm2d(num_features=input_channels)
        self.encoder_1 = UNet_down_block(input_channels, 64)
        self.encoder_2 = UNet_down_block(64, 128)
        self.encoder_3 = UNet_down_block(128, 256)
        self.encoder_4 = UNet_down_block(256, 512)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)
        self.mid_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)
        self.decoder_1 = UNet_up_block(512, 1024, 512)
        self.decoder_2 = UNet_up_block(256, 512, 256)
        self.decoder_3 = UNet_up_block(128, 256, 128)
        self.decoder_4 = UNet_up_block(64, 128, 64)
        self.last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        pass

    def forward(self, x):
        x = self.bn_init(x)
        self.x1_cat = self.encoder_1(x)
        self.x1 = self.max_pool(self.x1_cat)
        self.x2_cat = self.encoder_2(self.x1)
        self.x2_cat_1 = self.dropout(self.x2_cat)
        self.x2 = self.max_pool(self.x2_cat_1)
        self.x3_cat = self.encoder_3(self.x2)
        self.x3 = self.max_pool(self.x3_cat)
        self.x4_cat = self.encoder_4(self.x3)
        self.x4_cat_1 = self.dropout(self.x4_cat)
        self.x4 = self.max_pool(self.x4_cat_1)
        self.x_mid = self.mid_conv1(self.x4)
        self.x_mid = self.relu(self.x_mid)
        self.x_mid = self.mid_conv2(self.x_mid)
        self.x_mid = self.relu(self.x_mid)
        self.x_mid = self.dropout(self.x_mid)
        x = self.decoder_1(self.x4_cat_1, self.x_mid)
        x = self.decoder_2(self.x3_cat, x)
        x = self.decoder_3(self.x2_cat, x)
        x = self.decoder_4(self.x1_cat, x)
        x = self.last_conv(x)
        return x, self.softmax(x) # the final vector and the corresponding softmaxed prediction


# @torch.no_grad()
def check_model():
    model = UNet(input_channels=13, num_classes=22)
    model.eval()
    in_tensor = torch.Tensor(16, 13, 64, 64)
    with torch.no_grad():
        out_tensor, softmaxed = model(in_tensor)
        print(out_tensor.shape, softmaxed.shape)
        print(torch.argmax(softmaxed, dim=1)[0,:,:])
    pass


# @torch.no_grad()
def check_model_on_dataloader():
    model = UNet(input_channels=13, num_classes=22)
    model.eval()
    model.cuda(device=0)

    # loaders = get_dataloaders(images_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                       'ESA_landcover_dataset/raw/full_test_site_2015.tif',
    #                           bands=range(1,14),
    #                           labels_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                       'ESA_landcover_dataset/raw/label_full_test_site.npy',
    #                           save_data_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                          'ESA_landcover_dataset/raw/pickled_data.pkl',
    #                           block_size=256, model_input_size=64, batch_size=16, num_workers=4)

    # loaders = get_dataloaders_generated_data(generated_data_path='/home/annus/PycharmProjects/'
    #                                                              'ForestCoverChange_inputs_and_numerical_results/'
    #                                                              'ESA_landcover_dataset/divided',
    #                                          save_data_path='/home/annus/PycharmProjects/'
    #                                                         'ForestCoverChange_inputs_and_numerical_results/'
    #                                                         'ESA_landcover_dataset/generated_data.pkl',
    #                                          block_size=256, model_input_size=64, batch_size=16, num_workers=8)

    loaders = get_dataloaders_generated_data(generated_data_path='generated_dataset',
                                             save_data_path='pickled_generated_datalist.pkl',
                                             block_size=256, model_input_size=64, batch_size=128, num_workers=8)


    with torch.no_grad():
        train_dataloader, val_dataloader, test_dataloader = loaders
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            examples = examples.cuda(device=0)
            print('-> on batch {}/{}, {}'.format(idx + 1, len(train_dataloader), examples.size()))
            out_tensor, prediction = model(examples)
            print(examples.shape, labels.shape, out_tensor.shape,
                  prediction.shape, torch.argmax(prediction, dim=1)[0,:,:].shape)

    pass


def see_children_recursively(graph):
    further = False
    children = list(graph.children())
    for child in children:
        further = True
        see_children_recursively(child)
    if not further and isinstance(graph, nn.Conv2d):
        print(graph)


if __name__ == '__main__':
    # check_model_on_dataloader()
    VGG = models.vgg11(pretrained=True)
    VGG.eval()
    feature_layers = list(VGG.features)
    classifier_layers = list(VGG.classifier)
    # for i, layer in enumerate(feature_layers):
    #     print('{}.'.format(i), layer)
    # for i, layer in enumerate(classifier_layers):
    #     print('{}.'.format(i), layer)

    model = UNet(input_channels=13, num_classes=23)
    model.eval()

    see_children_recursively(graph=VGG)
    print('\n\n\n')
    see_children_recursively(graph=model)
    # print(model)
    # with torch.no_grad():
    #     summary(graph, input_size=(3, 228, 228))
    #     summary(model, input_size=(13, 64, 64))
    pass














