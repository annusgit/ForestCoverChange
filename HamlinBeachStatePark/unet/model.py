

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
from dataset import get_dataloaders
import os
import numpy as np
import pickle as pkl
import PIL.Image as Image
from tensorboardX import SummaryWriter
import itertools
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix
from torchnet.meter import ConfusionMeter as CM
import seaborn as sn
import pandas as pd


class_names = ['background/clutter', 'buildings', 'trees', 'cars', 'low_vegetation', 'impervious_surfaces', 'noise']
# class_names = ['background/clutter', 'buildings', 'trees', 'low_vegetation', 'impervious_surfaces']

# the following methods are used to generate the confusion matrix
###########################################################################################################33
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img (fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data (fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w ,h), buf.tostring())


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
###########################################################################################################33


class UNet_down_block(nn.Module):
    """
        Encoder class
    """
    def __init__(self, input_channel, output_channel, pretrained_weights=None):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.relu = nn.ReLU()

        # load previously trained weights
        # pretrained_weights = [conv1_W, conv1_b, conv2_W, conv2_b]
        if pretrained_weights:
            self.conv1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[0].transpose(3, 2, 1, 0)))
            self.conv1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[1].flatten()))
            self.conv2.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[2].transpose(3, 2, 1, 0)))
            self.conv2.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[3].flatten()))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(nn.Module):
    """
        Decoder class
    """
    def __init__(self, prev_channel, input_channel, output_channel, pretrained_weights=None):
        super(UNet_up_block, self).__init__()
        self.tr_conv_1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.relu = nn.ReLU()

        # load pretrained weights
        # pretrained_weights = [tconv_W, tconv_b, conv1_W, conv1_b, conv2_W, conv2_b]
        if pretrained_weights:
            self.tr_conv_1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[0].transpose(3, 2, 1, 0)))
            self.tr_conv_1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[1]).view(-1))
            self.conv_1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[2].transpose(3, 2, 1, 0)))
            self.conv_1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[3]).view(-1))
            self.conv_2.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[4].transpose(3, 2, 1, 0)))
            self.conv_2.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[5]).view(-1))

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
        model_dict = None
        if model_dir_path:
            # start by loading the pretrained weights from model_dict saved earlier
            with open(model_dir_path, 'rb') as handle:
                model_dict = pkl.load(handle)
                print('log: loaded saved model dictionary')
            print('total number of weights to be loaded into pytorch model =', len(model_dict.keys()))

        self.bn_init = nn.BatchNorm2d(num_features=input_channels)

        # create encoders and pass pretrained weights...
        if model_dir_path:
            self.encoder_1 = UNet_down_block(input_channels, 64, [model_dict['e1c1'], model_dict['e1c1_b'],
                                                                  model_dict['e1c2'], model_dict['e1c2_b']])
            self.encoder_2 = UNet_down_block(64, 128, [model_dict['e2c1'], model_dict['e2c1_b'],
                                                       model_dict['e2c2'], model_dict['e2c2_b']])
            self.encoder_3 = UNet_down_block(128, 256, [model_dict['e3c1'], model_dict['e3c1_b'],
                                                        model_dict['e3c2'], model_dict['e3c2_b']])
            self.encoder_4 = UNet_down_block(256, 512, [model_dict['e4c1'], model_dict['e4c1_b'],
                                                        model_dict['e4c2'], model_dict['e4c2_b']])
        else:
            self.encoder_1 = UNet_down_block(input_channels, 64, None)
            self.encoder_2 = UNet_down_block(64, 128, None)
            self.encoder_3 = UNet_down_block(128, 256, None)
            self.encoder_4 = UNet_down_block(256, 512, None)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)

        self.mid_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

        # load mid_conv weights
        if model_dir_path:
            self.mid_conv1.weight = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv1'].transpose(3, 2, 1, 0)))
            self.mid_conv1.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv1_b'][:,:,:].flatten()))
            self.mid_conv2.weight = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2'].transpose(3, 2, 1, 0)))
            self.mid_conv2.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2_b']).view(-1))


        # create decoders and load pretrained weights
        if model_dir_path:
            self.decoder_1 = UNet_up_block(512, 1024, 512, [model_dict['d1tc'], model_dict['d1tc_b'],
                                                            model_dict['d1c1'], model_dict['d1c1_b'],
                                                            model_dict['d1c2'], model_dict['d1c2_b']])
            self.decoder_2 = UNet_up_block(256, 512, 256, [model_dict['d2tc'], model_dict['d2tc_b'],
                                                           model_dict['d2c1'], model_dict['d2c1_b'],
                                                           model_dict['d2c2'], model_dict['d2c2_b']])
            self.decoder_3 = UNet_up_block(128, 256, 128, [model_dict['d3tc'], model_dict['d3tc_b'],
                                                           model_dict['d3c1'], model_dict['d3c1_b'],
                                                           model_dict['d3c2'], model_dict['d3c2_b']])
            self.decoder_4 = UNet_up_block(64, 128, 64, [model_dict['d4tc'], model_dict['d4tc_b'],
                                                         model_dict['d4c1'], model_dict['d4c1_b'],
                                                         model_dict['d4c2'], model_dict['d4c2_b']])
        else:
            self.decoder_1 = UNet_up_block(-1, 1024, 512, None)
            self.decoder_2 = UNet_up_block(-1, 512, 256, None)
            self.decoder_3 = UNet_up_block(-1, 256, 128, None)
            self.decoder_4 = UNet_up_block(-1, 128, 64, None)

        self.last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        # load final_conv weights
        if model_dir_path:
            self.last_conv.weight = torch.nn.Parameter(torch.Tensor(model_dict['final_conv'].transpose(3, 2, 1, 0)))
            self.last_conv.bias = torch.nn.Parameter(torch.Tensor(model_dict['final_conv_b']).view(-1))
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

        print(self.x4_cat_1.shape, self.x_mid.shape, self.decoder_1.tr_conv_1.weight.shape, self.decoder_1.conv_1.weight.shape, self.decoder_1.conv_2.weight.shape)
        x = self.decoder_1(self.x4_cat_1, self.x_mid)
        print(self.x3_cat.shape, x.shape, self.decoder_2.tr_conv_1.weight.shape, self.decoder_2.conv_1.weight.shape, self.decoder_2.conv_2.weight.shape)
        x = self.decoder_2(self.x3_cat, x)
        print(self.x2_cat.shape, x.shape, self.decoder_3.tr_conv_1.weight.shape, self.decoder_3.conv_1.weight.shape, self.decoder_3.conv_2.weight.shape)
        x = self.decoder_3(self.x2_cat, x)
        print(self.x1_cat.shape, x.shape, self.decoder_4.tr_conv_1.weight.shape, self.decoder_4.conv_1.weight.shape, self.decoder_4.conv_2.weight.shape)
        x = self.decoder_4(self.x1_cat, x)
        x = self.last_conv(x)
        return x, self.softmax(x)  # the final vector and the corresponding softmaxed prediction


@torch.no_grad()
def check_model():
    model = UNet(input_channels=13, num_classes=22)
    model.eval()
    in_tensor = torch.Tensor(16, 13, 64, 64)
    out_tensor, softmaxed = model(in_tensor)
    print(out_tensor.shape, softmaxed.shape)
    print(torch.argmax(softmaxed, dim=1)[0,:,:])
    pass


if __name__ == '__main__':
    check_model()
    pass




