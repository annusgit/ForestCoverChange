

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os
import re
import io
import cv2
import time
import numpy as np
import pickle as pkl
import torchnet as tnt
import PIL.Image as Image
from dataset import get_dataloaders
from tensorboardX import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as pl


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def train_net(model, file_path, in_seq_len, out_seq_len, pre_model, save_dir, batch_size, lr, log_after, cuda, device):
    print(model)
    # if os.path.exists('runs'):
    #     import shutil
    #     shutil.rmtree('runs') # just in case...
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if cuda:
        print('GPU')
        model.cuda(device=device)
        print('log: training started on device: {}'.format(device))
    writer = SummaryWriter()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(file_path=file_path, in_seq_len=in_seq_len,
                                                                        out_seq_len=out_seq_len, batch_size=batch_size)
    if True:
        i = 1
        m_loss, m_accuracy = [], []
        if pre_model:
            model.load_state_dict(torch.load(pre_model))
            print('log: resumed model {} successfully!'.format(pre_model))
            # starting point
            model_number = int(re.findall('\d+', str(pre_model))[0])
            i = i + model_number - 1
        else:
            print("log: let's start from the beginning...")

        while True:
            i += 1
            net_loss = []
            # new model path
            save_path = os.path.join(save_dir, 'model-{}.pt'.format(i))
            # remember to save only five previous models, so
            del_this = os.path.join(save_dir, 'model-{}.pt'.format(i - 6))
            if os.path.exists(del_this):
                os.remove(del_this)
                print('log: removed {}'.format(del_this))

            if i > 1 and not os.path.exists(save_path):
                torch.save(model.state_dict(), save_path)
                print('log: saved {}'.format(save_path))

            for idx, data in enumerate(train_dataloader, 1):
                ##########################
                model.train() # train mode at each epoch, just in case...
                #################################
                test_x, label = data['input'].unsqueeze(2), data['label'].squeeze(1)
                if cuda:
                    test_x = test_x.cuda(device=device)
                    label = label.cuda(device=device)
                out_x, h_n = model.continuous_forward(test_x, out_seq_len=out_seq_len)
                loss = criterion(out_x.view_as(label), label)
                net_loss.append(loss.item())
                if idx % log_after == 0 and idx > 0:
                    print('{}. ({}/{}) image size = {}, loss = {}'.format(i,
                                                                          idx,
                                                                          len(train_dataloader),
                                                                          out_x.size(),
                                                                          loss.item()))
                #################################
                # three steps for backprop
                model.zero_grad()
                loss.backward()
                # perform gradient clipping between loss backward and optimizer step
                clip_grad_norm_(model.parameters(), 0.05)
                optimizer.step()
                #################################
            mean_loss = np.asarray(net_loss).sum()/idx
            m_loss.append((i, mean_loss))
            writer.add_scalar(tag='train_loss', scalar_value=mean_loss, global_step=i)
            print('####################################')
            print('in_shape = {}, out_shape = {}'.format(test_x.shape, out_x.shape))
            print('epoch {} -> total loss = {:.5f}'.format(i, mean_loss))
            print('####################################')

            # validate model after each epoch
            eval_net(model=model, out_seq_len=out_seq_len, writer=writer,
                     criterion=criterion, val_loader=val_dataloader,
                     denominator=batch_size, cuda=cuda, device=device,
                     global_step=i)
    pass


@torch.no_grad()
def eval_net(**kwargs):
    model = kwargs['model']
    cuda = kwargs['cuda']
    device = kwargs['device']
    if cuda:
        model.cuda(device=device)
    if 'criterion' in kwargs.keys():
        writer = kwargs['writer']
        val_loader = kwargs['val_loader']
        criterion = kwargs['criterion']
        global_step = kwargs['global_step']
        net_loss = []
        model.eval()  # put in eval mode first ############################
        for idx, data in enumerate(val_loader, 1):
            test_x, label = data['input'].unsqueeze(2), data['label']
        # test_x, label = data['input'].unsqueeze(2), data['label']
        if cuda:
            test_x = test_x.cuda(device=device)
            label = label.cuda(device=device)
        # forward
        out_x, h_n = model.continuous_forward(test_x, out_seq_len=250000)
        # print(series_out.shape, series_in.shape)
        loss = criterion(out_x, label)
        net_loss.append(loss.item())
        #################################
        mean_loss = np.asarray(net_loss).sum()/idx
        # summarize mean accuracy
        writer.add_scalar(tag='val_loss', scalar_value=mean_loss, global_step=global_step)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$`$$$$$$$$$$$')
        print('in_shape = {}, out_shape = {}'.format(test_x.shape, out_x.shape))
        print('log: validation:: total loss = {:.5f}'.format(mean_loss))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        if mean_loss:
            ref = test_x[0,:].squeeze(1) #.numpy()
            this = label[0,:] #.numpy()
            that = out_x[0,:] #.numpy()
            this = np.hstack((ref,this)).astype(np.float) #/100.0 # rescale to best fit
            that = np.hstack((ref,that)).astype(np.float) #/100.0
            fig = pl.figure()
            pl.plot(this, label='series_in')
            pl.plot(that, label='series_out')
            out = pl.legend(loc='lower right')
            pl.savefig('temp.png')
            evaluated_image = cv2.imread('temp.png')
            # os.remove('eval.png')
            # put it into the summary writer
            evaluated_image = torch.Tensor(evaluated_image.transpose(2,0,1))
            writer.add_image('evaluation', evaluated_image, global_step)
            # pl.show()
    else:
        # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
        pre_model = kwargs['pre_model']
        base_folder = kwargs['base_folder']
        batch_size = kwargs['batch_size']
        log_after = kwargs['log_after']
        criterion = nn.CrossEntropyLoss()
        un_confusion_meter = tnt.meter.ConfusionMeter(10, normalized=False)
        confusion_meter = tnt.meter.ConfusionMeter(10, normalized=True)
        model.load_state_dict(torch.load(pre_model))
        print('log: resumed model {} successfully!'.format(pre_model))
        _, _, test_loader = get_dataloaders(base_folder=base_folder, batch_size=batch_size)
        net_accuracy, net_loss = [], []
        correct_count = 0
        total_count = 0
        for idx, data in enumerate(test_loader):
            model.eval()  # put in eval mode first
            test_x, label = data['input'], data['label']
            # print(test_x)
            # print(test_x.shape)
            # this = test_x.numpy().squeeze(0).transpose(1,2,0)
            # print(this.shape, np.min(this), np.max(this))
            if cuda:
                test_x = test_x.cuda(device=device)
                label = label.cuda(device=device)
            # forward
            out_x, pred = model.forward(test_x)
            loss = criterion(out_x, label)
            un_confusion_meter.add(predicted=pred, target=label)
            confusion_meter.add(predicted=pred, target=label)

            ###############################
            # pred = pred.view(-1)
            # pred = pred.cpu().numpy()
            # label = label.cpu().numpy()
            # print(pred.shape, label.shape)

            ###############################
            # get accuracy metric
            # correct_count += np.sum((pred == label))
            # print(pred, label)
            batch_correct = (label.eq(pred.long())).double().sum().item()
            correct_count += batch_correct
            # print(batch_correct)
            total_count += np.float(batch_size)
            net_loss.append(loss.item())
            if idx % log_after == 0:
                print('log: on {}'.format(idx))

            #################################
        mean_loss = np.asarray(net_loss).sum()
        mean_accuracy = correct_count * 100 / total_count
        print(correct_count, total_count)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        with open('normalized.pkl', 'wb') as this:
            pkl.dump(confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)

        with open('un_normalized.pkl', 'wb') as this:
            pkl.dump(un_confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)

        pass
    pass

















