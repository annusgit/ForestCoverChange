

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

# for getting pretrained layers from a vgg
matching_layers = [3, 6, 8, 11, 13, 16, 18]


class UNet_down_block(nn.Module):
    """
        Encoder class
    """
    def __init__(self, input_channel, output_channel, conv_1=None, conv_2=None):
        super(UNet_down_block, self).__init__()
        if conv_1:
            print('LOG: Using pretrained convolutional layer', conv_1)
        if conv_2:
            print('LOG: Using pretrained convolutional layer', conv_2)
        self.input_channels = input_channel
        self.output_channels = output_channel
        self.conv1 = conv_1 if conv_1 else nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = conv_2 if conv_2 else nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(nn.Module):
    """
        Decoder class
    """
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.output_channels = output_channel
        self.tr_conv_1 = nn.ConvTranspose2d(input_channel, input_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(prev_channel+input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.activate = nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.tr_conv_1(x)
        x = self.activate(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.activate(self.bn1(self.conv_1(x)))
        x = self.activate(self.bn2(self.conv_2(x)))
        return x


class UNet(nn.Module):

    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()
        VGG = models.vgg11(pretrained=True)
        pretrained_layers = list(VGG.features)
        # self.bn_init = nn.BatchNorm2d(num_features=input_channels)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.4)
        self.activate = nn.ReLU()

        self.encoder_1 = UNet_down_block(input_channels, 64)
        self.encoder_2 = UNet_down_block(64, 128, conv_1=pretrained_layers[3])
        self.encoder_3 = UNet_down_block(128, 256, conv_1=pretrained_layers[6], conv_2=pretrained_layers[8])
        self.encoder_4 = UNet_down_block(256, 512, conv_1=pretrained_layers[11], conv_2=pretrained_layers[13])
        # self.encoder_5 = UNet_down_block(512, 1024)
        # self.encoder_6 = UNet_down_block(1024, 1024)

        self.mid_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)

        # self.decoder_1 = UNet_up_block(prev_channel=self.encoder_6.output_channels,
                                       # input_channel=self.mid_conv2.out_channels,
                                       # output_channel=1024)
        # self.decoder_2 = UNet_up_block(prev_channel=self.encoder_5.output_channels,
        #                                input_channel=self.mid_conv2.out_channels,
        #                                output_channel=512)
        self.decoder_3 = UNet_up_block(prev_channel=self.encoder_4.output_channels,
                                       input_channel=self.mid_conv2.out_channels,
                                       output_channel=256)
        self.decoder_4 = UNet_up_block(prev_channel=self.encoder_3.output_channels,
                                       input_channel=self.decoder_3.output_channels,
                                       output_channel=128)
        self.decoder_5 = UNet_up_block(prev_channel=self.encoder_2.output_channels,
                                       input_channel=self.decoder_4.output_channels,
                                       output_channel=64)
        self.decoder_6 = UNet_up_block(prev_channel=self.encoder_1.output_channels,
                                       input_channel=self.decoder_5.output_channels,
                                       output_channel=64)
        # self.last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.binary_last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        pass

    def forward(self, x):
        # x = self.bn_init(x)
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
        # self.x5_cat = self.encoder_5(self.x4)
        # self.x5_cat_1 = self.dropout(self.x5_cat)
        # self.x5 = self.max_pool(self.x5_cat_1)
        # self.x6_cat = self.encoder_6(self.x5)
        # self.x6_cat_1 = self.dropout(self.x6_cat)
        # self.x6 = self.max_pool(self.x6_cat_1)

        self.x_mid = self.mid_conv1(self.x4)
        self.x_mid = self.activate(self.x_mid)
        self.x_mid = self.mid_conv2(self.x_mid)
        self.x_mid = self.activate(self.x_mid)
        self.x_mid = self.dropout(self.x_mid)

        # x = self.decoder_1(self.x6_cat_1, self.x_mid)
        # x = self.decoder_2(self.x5_cat, self.x_mid)
        x = self.decoder_3(self.x4_cat, self.x_mid)
        x = self.decoder_4(self.x3_cat, x)
        x = self.decoder_5(self.x2_cat, x)
        x = self.decoder_6(self.x1_cat, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)  # the final vector and the corresponding softmaxed prediction


@torch.no_grad()
def check_model():
    model = UNet(input_channels=11, num_classes=16)
    model.eval()
    in_tensor = torch.Tensor(16, 11, 128, 128)
    with torch.no_grad():
        out_tensor, softmaxed = model(in_tensor)
        print(out_tensor.shape, softmaxed.shape)
        print(torch.argmax(softmaxed, dim=1)[0,:,:])
    pass


@torch.no_grad()
def check_model_on_dataloader():
    model = UNet(input_channels=11, num_classes=16)
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


def see_children_recursively(graph, layer=None):
    further = False
    children = list(graph.children())
    for child in children:
        further = True
        if layer:
            see_children_recursively(child, layer)
        else:
            see_children_recursively(child)
    if layer:
        if not further and isinstance(graph, layer):
            print(graph)
    else:
        if not further:
            print(graph)
    # matching_pairs.append((graph.in_channels, graph.out_channels))


if __name__ == '__main__':
    # check_model
    model = UNet(input_channels=3, num_classes=4)
    model.eval()
    with torch.no_grad():
        summary(model, input_size=(3, 128, 128))
    see_children_recursively(graph=model, layer=nn.Conv2d)
    pass














