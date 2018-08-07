

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np
import pickle as pkl
from model import *
from dataset import get_dataloaders
from tensorboardX import SummaryWriter


def train_net(model, base_folder, pre_model, save_dir, batch_size, lr, log_after, cuda, device):
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
        i = 0
        m_loss, m_accuracy = [], []
        if pre_model:
            # self.load_state_dict(torch.load(pre_model)['model'])
            model.load_state_dict(torch.load(pre_model))
            print('log: resumed model {} successfully!'.format(pre_model))
            model_number = int(pre_model.split('/')[1].split('-')[1].split('.')[0])
        else:
            print('log: starting anew using ImageNet weights...')
        while True:
            i += 1
            net_loss = []
            net_accuracy = []
            if not pre_model:
                save_path = os.path.join(save_dir, 'model-{}.pt'.format(i))
            else:
                save_path = os.path.join(save_dir, 'model-{}.pt'.format(i+model_number-1))
            if i > 1 and not os.path.exists(save_path):
                torch.save(model.state_dict(), save_path)
                print('log: saved {}'.format(save_path))

                # remember to save only five previous models, so
                del_this = os.path.join(save_dir, 'model-{}.pt'.format(i + model_number - 6))
                if os.path.exists(del_this):
                    os.remove(del_this)
                    print('log: removed {}'.format(del_this))

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
                # get accuracy metric
                accuracy = (pred == label).sum()
                if idx % log_after == 0 and idx > 0:
                    print('{}. ({}/{}) image size = {}, loss = {}: accuracy = {}/{}'.format(i,
                                                                                            idx,
                                                                                            len(train_loader),
                                                                                            out_x.size(),
                                                                                            loss.item(),
                                                                                            accuracy,
                                                                                            batch_size))
                #################################
                # three steps for backprop
                model.zero_grad()
                loss.backward()
                # perform gradient clipping between loss backward and optimizer step
                clip_grad_norm_(model.parameters(), 0.05)
                optimizer.step()
                accuracy = accuracy * 100 / pred.view(-1).size(0)
                net_accuracy.append(accuracy)
                net_loss.append(loss.item())
                #################################
            mean_accuracy = np.asarray(net_accuracy).mean()
            mean_loss = np.asarray(net_loss).mean()
            m_loss.append((i, mean_loss))
            m_accuracy.append((i, mean_accuracy))

            writer.add_scalar(tag='loss', scalar_value=mean_loss, global_step=i)
            writer.add_scalar(tag='over_all accuracy', scalar_value=mean_accuracy, global_step=i)

            print('####################################')
            print('epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(i, mean_loss, mean_accuracy))
            print('####################################')

            # validate model after each epoch
            eval_net(model=model, writer=writer, criterion=criterion,
                     val_loader=val_dataloader, denominator=batch_size,
                     cuda=cuda, device=device, global_step=i)
    pass


def eval_net(**kwargs):
    model = kwargs['model']
    cuda = kwargs['cuda']
    device = kwargs['device']
    ###############################################
    model.eval() # put in eval mode first
    ###############################################
    if cuda:
        model.cuda(device=device)
    if 'criterion' in kwargs.keys():
        writer = kwargs['writer']
        val_loader = kwargs['val_loader']
        model = kwargs['model']
        criterion = kwargs['criterion']
        global_step = kwargs['global_step']
        net_accuracy, net_loss = [], []
        for idx, data in enumerate(val_loader):
            test_x, label = data['input'], data['label']
            if cuda:
                test_x = test_x.cuda(device=device)
                label = label.cuda(device=device)
            # forward
            out_x, pred = model.forward(test_x)
            loss = criterion(out_x, label)
            # get accuracy metric
            accuracy = (pred == label).sum()
            accuracy = accuracy * 100 / label.view(-1).size(0)
            net_accuracy.append(accuracy)
            net_loss.append(loss.item())

            ## get per-class accuracy
            avg = {x:[] for x in range(10)}
            for j in range(10):
                class_accuracy = ((pred == j) == (label == j)).sum()
                class_accuracy = class_accuracy * 100 / label.view(-1).size(0)
                avg[j].append(class_accuracy)
            #################################
        mean_accuracy = np.asarray(net_accuracy).mean()
        mean_loss = np.asarray(net_loss).mean()
        # summarize mean accuracy
        writer.add_scalar(tag='loss', scalar_value=mean_loss, global_step=global_step)
        writer.add_scalar(tag='over_all accuracy', scalar_value=mean_accuracy, global_step=global_step)

        # summarize per-classs accuracy
        mean_class_accuracies = map(np.asarray, [class_acc for class_name, class_acc in avg.iteritems()])
        mean_class_accuracies = map(np.mean, mean_class_accuracies)
        for j in range(10):
            writer.add_scalar(tag='class_{} accuracy'.format(j),
                              scalar_value=mean_class_accuracies[j],
                              global_step=global_step)
        classes_avg_acc = np.asarray(mean_class_accuracies).mean()
        writer.add_scalar(tag='classes avg. accuracy',
                          scalar_value=classes_avg_acc,
                          global_step=global_step)

        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: validation:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    else:
        # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
        pre_model = kwargs['pre_model']
        images = kwargs['images']
        labels = kwargs['labels']
        batch_size = kwargs['batch_size']
        criterion = nn.CrossEntropyLoss()

        model.load_state_dict(torch.load(pre_model))
        print('log: resumed model {} successfully!'.format(pre_model))
        _, _, test_loader = get_dataloaders(images_path=images, labels_path=labels, batch_size=batch_size)

        net_accuracy, net_loss = [], []
        net_class_accuracy_0, net_class_accuracy_1, net_class_accuracy_2, \
        net_class_accuracy_3, net_class_accuracy_4, net_class_accuracy_5,\
        net_class_accuracy_6  = [], [], [], [], [], [], []
        for idx, data in enumerate(test_loader):
            test_x, label = data['input'], data['label']
            # print(test_x.shape)
            if cuda:
                test_x = test_x.cuda()
            # forward
            out_x, pred = model.forward(test_x)
            pred = pred.cpu()
            loss = criterion(out_x.cpu(), label)

            # get accuracy metric
            accuracy = (pred == label).sum()
            accuracy = accuracy * 100 / (batch_size)
            net_accuracy.append(accuracy)
            net_loss.append(loss.item())
            if idx % 10 == 0:
                print('log: on {}'.format(idx))

            # get per-class metrics
            class_pred_0 = (pred == 0)
            class_label_0 = (label == 0)
            class_accuracy_0 = (class_pred_0 == class_label_0).sum()
            class_accuracy_0 = class_accuracy_0 * 100 / (batch_size)
            net_class_accuracy_0.append(class_accuracy_0)

            class_pred_1 = (pred == 1)
            class_label_1 = (label == 1)
            class_accuracy_1 = (class_pred_1 == class_label_1).sum()
            class_accuracy_1 = class_accuracy_1 * 100 / (batch_size)
            net_class_accuracy_1.append(class_accuracy_1)

            class_pred_2 = (pred == 2)
            class_label_2 = (label == 2)
            class_accuracy_2 = (class_pred_2 == class_label_2).sum()
            class_accuracy_2 = class_accuracy_2 * 100 / (batch_size)
            net_class_accuracy_2.append(class_accuracy_2)

            class_pred_3 = (pred == 3)
            class_label_3 = (label == 3)
            class_accuracy_3 = (class_pred_3 == class_label_3).sum()
            class_accuracy_3 = class_accuracy_3 * 100 / (batch_size)
            net_class_accuracy_3.append(class_accuracy_3)

            class_pred_4 = (pred == 4)
            class_label_4 = (label == 4)
            class_accuracy_4 = (class_pred_4 == class_label_4).sum()
            class_accuracy_4 = class_accuracy_4 * 100 / (batch_size)
            net_class_accuracy_4.append(class_accuracy_4)

            class_pred_5 = (pred == 5)
            class_label_5 = (label == 5)
            class_accuracy_5 = (class_pred_5 == class_label_5).sum()
            class_accuracy_5 = class_accuracy_5 * 100 / (batch_size)
            net_class_accuracy_5.append(class_accuracy_5)

            class_pred_6 = (pred == 6)
            class_label_6 = (label == 6)
            class_accuracy_6 = (class_pred_6 == class_label_6).sum()
            class_accuracy_6 = class_accuracy_6 * 100 / (batch_size)
            net_class_accuracy_6.append(class_accuracy_6)

            #################################
        mean_accuracy = np.asarray(net_accuracy).mean()
        mean_loss = np.asarray(net_loss).mean()

        class_0_mean_accuracy = np.asarray(net_class_accuracy_0).mean()
        class_1_mean_accuracy = np.asarray(net_class_accuracy_1).mean()
        class_2_mean_accuracy = np.asarray(net_class_accuracy_2).mean()
        class_3_mean_accuracy = np.asarray(net_class_accuracy_3).mean()
        class_4_mean_accuracy = np.asarray(net_class_accuracy_4).mean()
        class_5_mean_accuracy = np.asarray(net_class_accuracy_5).mean()
        class_6_mean_accuracy = np.asarray(net_class_accuracy_6).mean()

        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('log: class 0:: total accuracy = {:.5f}%'.format(class_0_mean_accuracy))
        print('log: class 1:: total accuracy = {:.5f}%'.format(class_1_mean_accuracy))
        print('log: class 2:: total accuracy = {:.5f}%'.format(class_2_mean_accuracy))
        print('log: class 3:: total accuracy = {:.5f}%'.format(class_3_mean_accuracy))
        print('log: class 4:: total accuracy = {:.5f}%'.format(class_4_mean_accuracy))
        print('log: class 5:: total accuracy = {:.5f}%'.format(class_5_mean_accuracy))
        print('log: class 6:: total accuracy = {:.5f}%'.format(class_6_mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        pass
    pass

















