

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os
import re
import numpy as np
import pickle as pkl
from model import *
import torchnet as tnt
from dataset import get_dataloaders
from tensorboardX import SummaryWriter


def train_net(model, base_folder, pre_model, save_dir, batch_size, lr, log_after, cuda, device):
    if not pre_model:
        print(model)
    writer = SummaryWriter()
    if cuda:
        print('GPU')
        model.cuda(device=device)
        print('log: training started on device: {}'.format(device))
    # define loss and optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_dataloader, test_loader = get_dataloaders(base_folder=base_folder,
                                                                batch_size=batch_size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if True:
        i = 1
        m_loss, m_accuracy = [], []
        if pre_model:
            # self.load_state_dict(torch.load(pre_model)['model'])
            model.load_state_dict(torch.load(pre_model))
            print('log: resumed model {} successfully!'.format(pre_model))
            print(model)

            # starting point
            # model_number = int(pre_model.split('/')[1].split('-')[1].split('.')[0])
            model_number = int(re.findall('\d+', str(pre_model))[0])
            i = i + model_number - 1
        else:
            print('log: starting anew using ImageNet weights...')

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

            correct_count, total_count = 0, 0
            for idx, data in enumerate(train_loader):
                ##########################
                model.train() # train mode at each epoch, just in case...
                ##########################
                test_x, label = data['input'], data['label']
                if cuda:
                    test_x = test_x.cuda(device=device)
                    label = label.cuda(device=device)
                # forward
                out_x, pred = model.forward(test_x)
                # out_x, pred = out_x.cpu(), pred.cpu()
                loss = criterion(out_x, label)
                net_loss.append(loss.item())

                # get accuracy metric
                batch_correct = (label.eq(pred.long())).double().sum().item()
                correct_count += batch_correct
                # print(batch_correct)
                total_count += np.float(pred.size(0))
                if idx % log_after == 0 and idx > 0:
                    print('{}. ({}/{}) image size = {}, loss = {}: accuracy = {}/{}'.format(i,
                                                                                            idx,
                                                                                            len(train_loader),
                                                                                            out_x.size(),
                                                                                            loss.item(),
                                                                                            batch_correct,
                                                                                            pred.size(0)))
                #################################
                # three steps for backprop
                model.zero_grad()
                loss.backward()
                # perform gradient clipping between loss backward and optimizer step
                clip_grad_norm_(model.parameters(), 0.05)
                optimizer.step()
                #################################
            mean_accuracy = correct_count / total_count * 100
            mean_loss = np.asarray(net_loss).mean()
            m_loss.append((i, mean_loss))
            m_accuracy.append((i, mean_accuracy))

            writer.add_scalar(tag='train loss', scalar_value=mean_loss, global_step=i)
            writer.add_scalar(tag='train over_all accuracy', scalar_value=mean_accuracy, global_step=i)

            print('####################################')
            print('epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(i, mean_loss, mean_accuracy))
            print('####################################')

            # validate model after each epoch
            eval_net(model=model, writer=writer, criterion=criterion,
                     val_loader=val_dataloader, denominator=batch_size,
                     cuda=cuda, device=device, global_step=i)
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
        correct_count, total_count = 0, 0
        net_loss = []
        model.eval()  # put in eval mode first ############################
        print('evaluating with batch size = 1')
        for idx, data in enumerate(val_loader):
            test_x, label = data['input'], data['label']
            if cuda:
                test_x = test_x.cuda(device=device)
                label = label.cuda(device=device)
            # forward
            out_x, pred = model.forward(test_x)
            loss = criterion(out_x, label)
            net_loss.append(loss.item())

            # get accuracy metric
            batch_correct = (label.eq(pred.long())).double().sum().item()
            correct_count += batch_correct
            total_count += np.float(pred.size(0))
        #################################
        mean_accuracy = correct_count / total_count * 100
        mean_loss = np.asarray(net_loss).mean()
        # summarize mean accuracy
        writer.add_scalar(tag='val. loss', scalar_value=mean_loss, global_step=global_step)
        writer.add_scalar(tag='val. over_all accuracy', scalar_value=mean_accuracy, global_step=global_step)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: validation:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

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
        _, _, test_loader  = get_dataloaders(base_folder=base_folder, batch_size=batch_size)
        net_accuracy, net_loss = [], []
        correct_count = 0
        total_count = 0
        print('batch size = {}'.format(batch_size))
        model.eval()  # put in eval mode first
        for idx, data in enumerate(test_loader):
            # if idx == 1:
            #     break
            # print(model.training)
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
            batch_correct = (label.eq(pred.long())).sum().item()
            correct_count += batch_correct
            # print(batch_correct)
            total_count += np.float(batch_size)
            net_loss.append(loss.item())
            if idx % log_after == 0:
                print('log: on {}'.format(idx))

            #################################
        mean_loss = np.asarray(net_loss).mean()
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

















