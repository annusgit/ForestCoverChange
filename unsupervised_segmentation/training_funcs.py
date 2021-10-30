

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os
import re
import sys
import numpy as np
import pickle as pkl
from model import *
import torchnet as tnt
from dataset import get_dataloaders
from tensorboardX import SummaryWriter


def train_net(model, data_path, pre_model, save_dir, batch_size, lr, log_after, cuda, device, one_hot=False):
    if not pre_model:
        print(model)
    writer = SummaryWriter()
    if cuda:
        print('GPU')
        model.cuda(device=device)
        print('log: training started on device: {}'.format(device))
    # define loss and optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    lr_final = 0.0000003
    num_epochs = 500
    LR_decay = (lr_final/lr)**(1./num_epochs)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=LR_decay)
    # print(LR_decay, optimizer.state)
    # print(optimizer.param_groups[0]['lr'])
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    train_loader, val_dataloader, test_loader = get_dataloaders(path_to_nparray=data_path,
                                                                batch_size=batch_size,
                                                                normalize=True)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if True:
        i = 1
        m_loss, m_accuracy = [], []
        if pre_model:
            # self.load_state_dict(torch.load(pre_model)['model'])
            model.load_state_dict(torch.load(os.path.join(save_dir, "model-"+pre_model+'.pt')))
            print('log: resumed model {} successfully!'.format(pre_model))
            print(model)

            # starting point
            # model_number = int(pre_model.split('/')[1].split('-')[1].split('.')[0])
            model_number = int(pre_model) #re.findall('\d+', str(pre_model))[0])
            i = i + model_number - 1
        else:
            print('log: starting anew...')

        while i < num_epochs:
            i += 1
            net_loss = []
            # new model path
            save_path = os.path.join(save_dir, 'model-{}.pt'.format(i))
            # remember to save only five previous models, so
            del_this = os.path.join(save_dir, 'model-{}.pt'.format(i-5))
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
                test_x, label = data
                if cuda:
                    test_x = test_x.cuda(device=device)
                    label = label.cuda(device=device)
                # forward
                out_x, pred = model(test_x)
                # out_x, pred = out_x.cpu(), pred.cpu()
                loss = criterion(out_x, label)
                net_loss.append(loss.item())

                # get accuracy metric
                if one_hot:
                    batch_correct = (torch.argmax(label, dim=1).eq(pred.long())).double().sum().item()
                else:
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
            # remember this should be in the epoch loop ;)
            scheduler.step()  # to dynamically change the learning rate
            mean_accuracy = correct_count / total_count * 100
            mean_loss = np.asarray(net_loss).mean()
            m_loss.append((i, mean_loss))
            m_accuracy.append((i, mean_accuracy))

            writer.add_scalar(tag='train loss', scalar_value=mean_loss, global_step=i)
            writer.add_scalar(tag='train over_all accuracy', scalar_value=mean_accuracy, global_step=i)

            print('####################################')
            print('epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}% (lr: {})'.format(i,
                                                                                              mean_loss,
                                                                                              mean_accuracy,
                                                                                              optimizer.param_groups[0]['lr']))
            print('####################################')

            # validate model after each epoch
            with torch.no_grad():
                eval_net(model=model, writer=writer, criterion=criterion,
                         val_loader=val_dataloader, denominator=batch_size,
                         cuda=cuda, device=device, global_step=i, one_hot=one_hot)
    pass


# @torch.no_grad()
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
        print('evaluating now...')
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                test_x, label = data
                if cuda:
                    test_x = test_x.cuda(device=device)
                    label = label.cuda(device=device)
                # forward
                out_x, pred = model.forward(test_x)
                loss = criterion(out_x, label)
                net_loss.append(loss.item())

                # get accuracy metric
                if kwargs['one_hot']:
                    batch_correct = (torch.argmax(label, dim=1).eq(pred.long())).double().sum().item()
                else:
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
        data_path = kwargs['data_path']
        batch_size = kwargs['batch_size']
        log_after = kwargs['log_after']
        criterion = nn.CrossEntropyLoss()
        un_confusion_meter = tnt.meter.ConfusionMeter(10, normalized=False)
        confusion_meter = tnt.meter.ConfusionMeter(10, normalized=True)
        model.load_state_dict(torch.load(pre_model))
        print('log: resumed model {} successfully!'.format(pre_model))
        _, _, test_loader  = get_dataloaders(path_to_nparray=data_path, batch_size=batch_size)
        net_accuracy, net_loss = [], []
        correct_count = 0
        total_count = 0
        print('batch size = {}'.format(batch_size))
        model.eval()  # put in eval mode first
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                # if idx == 1:
                #     break
                # print(model.training)
                test_x, label = data
                # print(test_x)
                # print(test_x.shape)
                # this = test_x.numpy().squeeze(0).transpose(1,2,0)
                # print(this.shape, np.min(this), np.max(this))
                if cuda:
                    test_x = test_x.cuda(device=device)
                    label = label.cuda(device=device)
                # forward
                out_x, pred = model(test_x)
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
                # get accuracy metric
                if 'one_hot' in kwargs.keys():
                    if kwargs['one_hot']:
                        batch_correct = (torch.argmax(label, dim=1).eq(pred.long())).double().sum().item()
                else:
                    batch_correct = (label.eq(pred.long())).double().sum().item()
                # print(label.shape, pred.shape)
                # break
                correct_count += batch_correct
                # print(batch_correct)
                total_count += np.float(batch_size)
                net_loss.append(loss.item())
                if idx % log_after == 0:
                    log_string = 'log: on {}/{}'.format(idx, len(test_loader))
                    # print(log_string)
                    # print('\b'*len(log_string), file=sys.stdout, end='')

                    print('\b'*20, end='')
                    print(log_string, end='', file=sys.stdout)
                #################################
            mean_loss = np.asarray(net_loss).mean()
            mean_accuracy = correct_count * 100 / total_count
            print()
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














