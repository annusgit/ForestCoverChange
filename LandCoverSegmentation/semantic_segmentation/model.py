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
    def __init__(self, topology, input_channels, num_classes):
        super(UNet, self).__init__()
        # these topologies are possible right now
        self.topologies = {
            "ENC_1_DEC_1": self.ENC_1_DEC_1,
            "ENC_2_DEC_2": self.ENC_2_DEC_2,
            "ENC_3_DEC_3": self.ENC_3_DEC_3,
            "ENC_4_DEC_4": self.ENC_4_DEC_4,
        }
        assert topology in self.topologies
        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.6)
        self.activate = nn.ReLU()
        self.encoder_1 = UNet_down_block(input_channels, 64)
        self.encoder_2 = UNet_down_block(64, 128, conv_1=pretrained_layers[3])
        self.encoder_3 = UNet_down_block(128, 256, conv_1=pretrained_layers[6], conv_2=pretrained_layers[8])
        self.encoder_4 = UNet_down_block(256, 512, conv_1=pretrained_layers[11], conv_2=pretrained_layers[13])
        self.mid_conv_64_64_a = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_64_64_b = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_128_128_a = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_128_128_b = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_256_256_a = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_256_256_b = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.decoder_4 = UNet_up_block(prev_channel=self.encoder_4.output_channels, input_channel=self.mid_conv_1024_1024.out_channels, output_channel=256)
        self.decoder_3 = UNet_up_block(prev_channel=self.encoder_3.output_channels, input_channel=self.decoder_4.output_channels, output_channel=128)
        self.decoder_2 = UNet_up_block(prev_channel=self.encoder_2.output_channels, input_channel=self.decoder_3.output_channels, output_channel=64)
        self.decoder_1 = UNet_up_block(prev_channel=self.encoder_1.output_channels, input_channel=self.decoder_2.output_channels, output_channel=64)
        self.binary_last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.forward = self.topologies[topology]
        print('\n\n' + "#" * 100)
        print("(LOG): The following Model Topology will be Utilized: {}".format(self.forward.__name__))
        print("#" * 100 + '\n\n')
        pass

    def ENC_1_DEC_1(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1_cat_1 = self.dropout(x1_cat)
        x1 = self.max_pool(x1_cat_1)
        x_mid = self.mid_conv_64_64_a(x1)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_64_64_b(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_1(x1_cat, x_mid)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)

    def ENC_2_DEC_2(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)
        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)
        x_mid = self.mid_conv_128_128_a(x2)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_128_128_b(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_2(x2_cat, x_mid)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)

    def ENC_3_DEC_3(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)
        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)
        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)
        x_mid = self.mid_conv_256_256_a(x3)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_256_256_b(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_3(x3_cat, x_mid)
        x = self.decoder_2(x2_cat, x)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)

    def ENC_4_DEC_4(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)
        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)
        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)
        x4_cat = self.encoder_4(x3)
        x4_cat_1 = self.dropout(x4_cat)
        x4 = self.max_pool(x4_cat_1)
        x_mid = self.mid_conv_512_1024(x4)
        x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_1024_1024(x_mid)
        x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)
        x = self.decoder_4(x4_cat, x_mid)
        x = self.decoder_3(x3_cat, x)
        x = self.decoder_2(x2_cat, x)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        # return the final vector and the corresponding softmax-ed prediction
        return x, self.softmax(x)
    pass

@torch.no_grad()
def check_model(topology, input_channels, num_classes, input_shape):
    model = UNet(topology=topology, input_channels=input_channels, num_classes=num_classes)
    model.eval()
    in_tensor = torch.Tensor(*input_shape)
    with torch.no_grad():
        out_tensor, softmaxed = model(in_tensor)
        print(in_tensor.shape, out_tensor.shape)
    pass


@torch.no_grad()
def check_model_on_dataloader():
    model = UNet(input_channels=11, num_classes=16)
    model.eval()
    model.cuda(device=0)
    loaders = get_dataloaders_generated_data(generated_data_path='generated_dataset', save_data_path='pickled_generated_datalist.pkl', block_size=256,
                                             model_input_size=64, batch_size=128, num_workers=8)
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
    pass


if __name__ == '__main__':
    # check_model
    check_model(topology="ENC_1_DEC_1", input_channels=7, num_classes=2, input_shape=[4, 7, 64, 64])
    # model = UNet(topology='ENC_1_DEC_1', input_channels=3, num_classes=4)
    # # model.cuda(device='cuda')
    # model.eval()
    # with torch.no_grad():
    #     summary(model, input_size=(3, 128, 128))
    # see_children_recursively(graph=model, layer=nn.Conv2d)
    pass














