

"""
    UNet model definition in here
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import numpy as np
import pickle as pkl
from dataset import get_dataloaders


class UNet_down_block(nn.Module):
    """
        Encoder class
    """
    def __init__(self, input_channel, output_channel, pretrained_weights):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.relu = nn.ReLU()

        # load previously trained weights
        # pretrained_weights = [conv1_W, conv1_b, conv2_W, conv2_b]
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
    def __init__(self, prev_channel, input_channel, output_channel, pretrained_weights):
        super(UNet_up_block, self).__init__()
        self.tr_conv_1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
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
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.relu(self.bn2(self.conv_2(x)))
        return x


class UNet(nn.Module):

    def __init__(self, model_dir_path, input_channels):
        super(UNet, self).__init__()
        # start by loading the pretrained weights from model_dict saved earlier
        with open(model_dir_path, 'rb') as handle:
            model_dict = pkl.load(handle)
            print('log: loaded saved model dictionary')
        print('total number of weights to be loaded into pytorch model =', len(model_dict.keys()))

        # first batchnorm, then a 3 -> 6 channel converter and finally the rest of the unet...
        self.bn_init = nn.BatchNorm2d(num_features=input_channels)
        self.three_to_six = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, padding=1)

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
        self.mid_conv1.bias = torch.nn.Parameter(torch.Tensor(model_dict['mid_conv1_b'][:,:,:].flatten()))
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
        self.softmax = nn.Softmax(dim=1)

        # load final_conv weights
        self.last_conv.weight = torch.nn.Parameter(torch.Tensor(model_dict['final_conv'].transpose(3, 2, 1, 0)))
        self.last_conv.bias = torch.nn.Parameter(torch.Tensor(model_dict['final_conv_b']).view(-1))
        pass

    def forward(self, x):
        # fix inputs according to requirement of pretrained net
        x = self.bn_init(x)
        x = self.three_to_six(x)

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
        x = self.softmax(x)
        pred = torch.argmax(x, dim=1)
        return x, pred


def train_net(model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda):
    print(model)
    if cuda:
        print('GPU')
        model.cuda()
    # define loss and optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_dataloader, test_loader = get_dataloaders(images_path=images,
                                                                labels_path=labels,
                                                                batch_size=batch_size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(sum_dir):
        os.mkdir(sum_dir)

    if True:
        i = 0
        if pre_model:
            # self.load_state_dict(torch.load(pre_model)['model'])
            model.load_state_dict(torch.load(pre_model))
            print('log: resumed model {} successfully!'.format(pre_model))
        else:
            print('log: starting anew...')
        while True:
            ##########################
            model.train()
            ##########################
            i += 1
            net_loss = []
            net_accuracy = []
            m_loss, m_accuracy = [], []
            save_path = os.path.join(save_dir, 'model-{}.pt'.format(i))
            if i > 1 and not os.path.exists(save_path):
                torch.save(model.state_dict(), save_path)
                print('log: saved {}'.format(save_path))
                # also save the summary
                with open(os.path.join(sum_dir, 'summary_{}.pkl'.format(i)), 'wb') as summary:
                    sum = {'acc': m_accuracy, 'loss': m_loss}
                    pkl.dump(sum, summary, protocol=pkl.HIGHEST_PROTOCOL)
            for idx, data in enumerate(train_loader):
                test_x, label = data['input'], data['label']
                test_x = test_x.cuda() if cuda else test_x
                # forward
                out_x, pred = model.forward(test_x)
                # print(out_x.size(), pred.size(), label.size())
                # print(np.unique(label.cpu().numpy()), print(np.unique(pred.cpu().numpy())))
                loss = criterion(out_x.cpu(), label)
                # get accuracy metric
                accuracy = (pred.cpu() == label).sum()
                if idx % log_after == 0 and idx > 0:
                    print('{}. ({}/{}) image size = {}, loss = {}: accuracy = {}/{}'.format(i,
                                                                                            idx,
                                                                                            len(train_loader),
                                                                                            out_x.size(),
                                                                                            loss.item(),
                                                                                            accuracy,
                                                                                            batch_size * 64**2))
                #################################
                # three steps for backprop
                model.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy = accuracy * 100 / (batch_size*64**2)
                net_accuracy.append(accuracy)
                net_loss.append(loss.item())
                #################################
            mean_accuracy = np.asarray(net_accuracy).mean()
            mean_loss = np.asarray(net_loss).mean()
            m_loss.append((i, mean_loss))
            m_accuracy.append((i, mean_accuracy))
            print('####################################')
            print('epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(i, mean_loss, mean_accuracy))
            print('####################################')

            # validate model
            if i % 10 == 0:
                eval_net(model=model, criterion=criterion, val_loader=val_dataloader,
                         denominator=batch_size * 64**2, cuda=cuda)
    pass


def eval_net(model, criterion, val_loader, denominator, cuda):
    model.eval()
    net_accuracy, net_loss = [], []
    for idx, data in enumerate(val_loader):
        test_x, label = data['input'], data['label']
        if cuda:
            test_x = test_x.cuda()
        # forward
        out_x, pred = model.forward(test_x)
        loss = criterion(out_x.cpu(), label)
        # get accuracy metric
        accuracy = (pred.cpu() == label).sum()
        accuracy = accuracy * 100 / denominator
        net_accuracy.append(accuracy)
        net_loss.append(loss.item())
        #################################
    mean_accuracy = np.asarray(net_accuracy).mean()
    mean_loss = np.asarray(net_loss).mean()
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('log: validation:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    pass
