

"""
    Model for Unet (pretrained in Matlab)
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as pl


class UNet_down_block(nn.Module):
    """
        Encoder class
    """
    def __init__(self, input_channel, output_channel, pretrained_weights):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        # print(self.conv1.weight.size(), self.conv1.bias.size())
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # load previously trained weights
        # pretrained_weights = [conv1_W, conv1_b, conv2_W, conv2_b]
        self.conv1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[0].transpose(3, 2, 1, 0)))
        self.conv1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[1]).view(-1))
        self.conv2.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[2].transpose(3, 2, 1, 0)))
        self.conv2.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[3]).view(-1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet_up_block(nn.Module):
    """
        Decoder class
    """
    def __init__(self, prev_channel, input_channel, output_channel, pretrained_weights):
        super(UNet_up_block, self).__init__()
        self.tr_conv_1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # load pretrained weights
        # pretrained_weights = [tconv_W, tconv_b, conv1_W, conv1_b, conv2_W, conv2_b]
        self.tr_conv_1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[0].transpose(3, 2, 1, 0)))
        self.tr_conv_1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[1]).view(-1))
        self.conv_1.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[2].transpose(3, 2, 1, 0)))
        self.conv_1.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[3]).view(-1))
        self.conv_2.weight = torch.nn.Parameter(torch.Tensor(pretrained_weights[4].transpose(3, 2, 1, 0)))
        self.conv_2.bias = torch.nn.Parameter(torch.Tensor(pretrained_weights[5]).view(-1))

    def forward(self, prev_feature_map, x):
        x = self.tr_conv_1(x)
        x = self.relu(x)
        x = torch.cat((prev_feature_map, x), dim=1)
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        return x


class my_unet(nn.Module):

    def __init__(self, model_dir_path, input_channels):
        super(my_unet, self).__init__()
        # start by loading the pretrained weights from model_dict saved earlier
        import pickle
        with open(model_dir_path, 'rb') as handle:
            model_dict = pickle.load(handle)
            print('log: loaded saved model dictionary')
        print('total number of weights to be loaded into pytorch model =', len(model_dict.keys()))

        # create encoders and pass pretrained weights...
        self.encoder_1 = UNet_down_block(input_channels, 64, [model_dict['e1c1'], model_dict['e1c1_b'],
                                                              model_dict['e1c2'], model_dict['e1c2_b']])
        self.encoder_2 = UNet_down_block(64, 128, [model_dict['e2c1'], model_dict['e2c1_b'],
                                                   model_dict['e2c2'], model_dict['e2c2_b']])
        self.encoder_3 = UNet_down_block(128, 256, [model_dict['e3c1'], model_dict['e3c1_b'],
                                                    model_dict['e3c2'], model_dict['e3c2_b']])
        self.encoder_4 = UNet_down_block(256, 512, [model_dict['e4c1'], model_dict['e4c1_b'],
                                                    model_dict['e4c2'], model_dict['e4c2_b']])
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)

        self.mid_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

        # load mid_conv weights
        self.mid_conv1.weight = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv1'].transpose(3, 2, 1, 0)))
        self.mid_conv1.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv1_b']).view(-1))
        self.mid_conv2.weight = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2'].transpose(3, 2, 1, 0)))
        self.mid_conv2.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2_b']).view(-1))

        # create decoders and load pretrained weights
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

        self.last_conv = nn.Conv2d(64, 18, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # load final_conv weights
        self.last_conv.weight = torch.nn.Parameter(torch.Tensor(model_dict['final_conv'].transpose(3, 2, 1, 0)))
        self.last_conv.bias = torch.nn.Parameter(torch.Tensor(model_dict['final_conv_b']).view(-1))

        pass

    def forward(self, x):
        self.eval()
        self.x1_cat = self.encoder_1(x)
        self.x1 = self.max_pool(self.x1_cat)
        self.x2_cat = self.encoder_2(self.x1)
        self.x2 = self.max_pool(self.x2_cat)
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
        x = self.softmax(self.last_conv(x))
        return x, torch.argmax(x, dim=1)


def load_weights_from_matfiles(dir_path):
    """
        Uses scipy.io to read .mat files and loads weights into torch model
    :param path_to_file: path to mat file to read
    :return: None, but saves the model dictionary!
    """
    import pickle
    model_file = 'Unet_pretrained_model.pkl'
    if os.path.exists(os.path.join(dir_path, model_file)):
        print('loading saved model dictionary...')
        with open(os.path.join(dir_path, model_file), 'rb') as handle:
            model_dict = pickle.load(handle)
        for i, layer in enumerate(model_dict.keys(), 1):
            print('{}.'.format(i), layer, model_dict[layer].shape)
    else:
        model_dict = {}
        for file in [x for x in os.listdir(dir_path) if x.endswith('.mat')]:
            layer, _ = os.path.splitext(file)
            try:
                read = sio.loadmat(os.path.join(dir_path, file))
            except:
                print(layer)
            print(layer, read[layer].shape)
            model_dict[layer] = read[layer]
        pass
        os.chdir('/home/annus/Desktop/trainedUnet/weightsforpython/')
        with open(model_file, 'wb') as handle:
            pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved model!!!')

    # test on pytorch layers now...
    conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    conv.weight = torch.nn.Parameter(torch.Tensor(model_dict['e4c1']))
    conv.bias = torch.nn.Parameter(torch.Tensor(model_dict['e4c1_b']))

    mid_conv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
    mid_conv.weight = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2']))
    mid_conv.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv2_b']))

    tr_conv = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    tr_conv.weight = torch.nn.Parameter(torch.Tensor(model_dict['d2tc']))
    tr_conv.bias = torch.nn.Parameter(torch.Tensor(model_dict['d2tc_b']))


if __name__ == '__main__':

    dir_path = '/home/annus/Desktop/trainedUnet/weightsforpython/'
    # layer_name = 'e1c2'
    # weights = load_weights_from_matfiles(dir_path)

    import time
    if True:
        channels = 6
        net = my_unet(model_dir_path=os.path.join(dir_path, 'Unet_pretrained_model.pkl'), input_channels=channels)
        batch_size = 1
        w, h, patch = 0, 3000, 256
        end = False
        # test_x = torch.FloatTensor(batch_size, channels, patch, patch)
        val_data = np.load('/home/annus/Desktop/rit18_data/val_data.npy', mmap_mode='r').transpose((1, 2, 0))
        val_label = np.load('/home/annus/Desktop/rit18_data/val_labels.npy', mmap_mode='r')
        print('val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
        # pad the data to be divisible by patch size
        val_data = np.pad(val_data, ((0,patch-val_data.shape[0]%patch),
                                     (0,patch-val_data.shape[1]%patch),
                                     (0, 0)), 'constant')
        val_label = np.pad(val_label, ((0, patch - val_label.shape[0] % patch),
                                       (0, patch - val_label.shape[1] % patch)), 'constant')
        net_accuracy = []
        print('val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
        for i in range(val_data.shape[0]//patch):
            for j in range(val_data.shape[1]//patch):
                image = val_data[patch*i:patch*i+patch, j*patch:j*patch+patch, :6].astype('int32')
                label = val_label[patch*i:patch*i+patch, j*patch:j*patch+patch].astype('int32')
                seg_map = (val_data[patch*i:patch*i+patch,
                           j*patch:j*patch+patch, 6].astype('int32')*1/65535).astype(np.int32)
                w += 1
                print('w = {}-{}, h = {}-{}'.format(patch*i, patch*i+patch, j*patch, j*patch+patch))
                print('numpy image size =', image.shape, 'numpy label size =', label.shape)
                # make it [batch_size, channels, height, width]
                # pl.imshow(image, )
                test_x = torch.FloatTensor(image.transpose((2,1,0))).unsqueeze(0)
                print('test image size = {}'.format(test_x.size()))
                start = time.time()
                out_x, pred = net(test_x)
                pred = pred.squeeze(0).numpy() * seg_map
                # print('output = {}'.format(out_x))
                # print('output shape = {}'.format(out_x.shape))
                # print('prediction = {}'.format(pred))
                # print('label = {}'.format(label))
                # print('prediction shape = {}'.format(pred.shape))
                print('elapsed time = {}'.format(time.time()-start))
                accuracy = (pred == label).sum() * 100 /(256*256)
                print('accuracy = {:.5f}%'.format(accuracy))
                net_accuracy.append(accuracy)
                # print(seg_map)

    mean_accuracy = np.asarray(net_accuracy).mean()
    print('total accuracy = {:.5f}%'.format(mean_accuracy))





