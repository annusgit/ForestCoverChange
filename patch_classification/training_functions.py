

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
import torchnet as tnt
from dataset import get_dataloaders
from tensorboardX import SummaryWriter


def train_net(model, base_folder, pre_model, save_dir, batch_size, lr, log_after, cuda, device):
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
            print(model)

            # graph_layers = list(model.feature_extracter)  # only get the feature extractor, we don't need the classifier
            # new_graph = []
            # for layer in graph_layers[1:]:
            #     new_graph.append(layer)
            # model_list = nn.ModuleList(new_graph)
            # feature_extracter = nn.Sequential(*model_list)
            # model.feature_extracter = feature_extracter
            # print(model.feature_extracter)
            # # classifier = nn.Sequential(
            # #     nn.Linear(in_features=512 * ((64 // 2 ** 5) ** 2), out_features=1024),
            # #     nn.ReLU(),
            # #     nn.Linear(in_features=1024, out_features=512),
            # #     nn.ReLU(),
            # #     nn.Dropout(p=0.7),
            # #     nn.Linear(in_features=512, out_features=256),
            # #     nn.ReLU(),
            # #     nn.Linear(in_features=256, out_features=128),
            # #     nn.ReLU(),
            # #     nn.Dropout(p=0.7),
            # #     nn.Linear(in_features=128, out_features=10),
            # #     nn.LogSoftmax(dim=0)
            # # )
            # torch.save(model.state_dict(), 'direct_model.pt')

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
    if cuda:
        model.cuda(device=device)
    if 'criterion' in kwargs.keys():
        writer = kwargs['writer']
        val_loader = kwargs['val_loader']
        model = kwargs['model']
        criterion = kwargs['criterion']
        global_step = kwargs['global_step']
        net_accuracy, net_loss = [], []
        model.eval()  # put in eval mode first
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
            pred = pred.view(-1)
            # pred = pred.cpu().numpy()
            # label = label.cpu().numpy()
            # print(pred.shape, label.shape)

            ###############################
            # get accuracy metric
            # correct_count += np.sum((pred == label))
            correct_count += (label.eq(pred.long())).sum()

            total_count += pred.shape[0]
            net_loss.append(loss.item())
            if idx % log_after == 0:
                print('log: on {}'.format(idx))

            #################################
        mean_loss = np.asarray(net_loss).mean()
        mean_accuracy = correct_count * 100 / total_count
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        with open('normalized.pkl', 'wb') as this:
            pkl.dump(confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)

        with open('un_normalized.pkl', 'wb') as this:
            pkl.dump(un_confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)

        pass
    pass

















