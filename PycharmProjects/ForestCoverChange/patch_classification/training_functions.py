

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
from dataset import get_dataloaders_generated_data
from tensorboardX import SummaryWriter


def train_net(model, generated_data_path, input_dim, workers, pre_model, save_data, save_dir, sum_dir,
              batch_size, lr, epochs, log_after, cuda, device):
    if cuda:
        print('log: Using GPU')
        model.cuda(device=device)

    if pre_model == -1:
        model_number = 0
        print('log: No trained model passed. Starting from scratch...')
        # model_path = os.path.join(save_dir, 'model-{}.pt'.format(model_number))
    else:
        model_number = pre_model
        model_path = os.path.join(save_dir, 'model-{}.pt'.format(pre_model))
        model.load_state_dict(torch.load(model_path), strict=False)
        print('log: Resuming from model {} ...'.format(model_path))
    ###############################################################################

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(sum_dir):
        os.mkdir(sum_dir)
    # writer = SummaryWriter()

    # define loss and optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    # focal_criterion = FocalLoss2d(weight=weights)
    crossentropy_criterion = nn.BCELoss()
    # dice_criterion = DiceLoss(weights=weights)

    lr_final = 5e-5
    LR_decay = (lr_final / lr) ** (1. / epochs)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=LR_decay)

    loaders = get_dataloaders_generated_data(generated_data_path=generated_data_path, save_data_path=save_data,
                                             model_input_size=input_dim, batch_size=batch_size, num_classes=2,
                                             one_hot=True, num_workers=workers)
    writer = SummaryWriter()

    train_loader, val_dataloader, test_loader = loaders
    # training loop
    for k in range(epochs):
        net_loss = []
        total_correct, total_examples = 0, 0
        model_path = os.path.join(save_dir, 'model-{}.pt'.format(model_number+k))
        if not os.path.exists(model_path):
            torch.save(model.state_dict(), model_path)
            print('log: saved {}'.format(model_path))
            # remember to save only five previous models, so
            del_this = os.path.join(save_dir, 'model-{}.pt'.format(model_number+k-6))
            if os.path.exists(del_this):
                os.remove(del_this)
                print('log: removed {}'.format(del_this))

        for idx, data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, logits = model.forward(test_x)
            pred = torch.argmax(logits, dim=1)
            # print(np.unique(pred.detach().cpu().numpy()))
            not_one_hot_target = torch.argmax(label, dim=1)
            # dice_criterion(logits, label) #+ focal_criterion(logits, not_one_hot_target) #
            loss = crossentropy_criterion(logits, label.float())
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            accurate = (pred == not_one_hot_target).sum().item()
            numerator = float(accurate)
            denominator = float(pred.view(-1).size(0)) #test_x.size(0)*dimension**2)
            total_correct += numerator
            total_examples += denominator

            if idx % log_after == 0 and idx > 0:
                accuracy = float(numerator) * 100 / denominator
                print('{}. ({}/{}) output size = {}, loss = {}, '
                      'accuracy = {}/{} = {:.2f}%, (lr = {})'.format(k, idx, len(train_loader), out_x.size(),
                                                                     loss.item(), numerator, denominator, accuracy,
                                                                     optimizer.param_groups[0]['lr']))
            net_loss.append(loss.item())

        # this should be done at the end of epoch only
        scheduler.step()  # to dynamically change the learning rate
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        writer.add_scalar(tag='train loss', scalar_value=mean_loss, global_step=k)
        writer.add_scalar(tag='train over_all accuracy', scalar_value=mean_accuracy, global_step=k)
        print('####################################')
        print('LOG: epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(k, mean_loss, mean_accuracy))
        print('####################################')

        # validate model
        print('log: Evaluating now...')
        eval_net(model=model, criterion=crossentropy_criterion, val_loader=val_dataloader, cuda=cuda, device=device,
                 writer=None, batch_size=batch_size, global_step=k)
    pass


@torch.no_grad()
def eval_net(**kwargs):
    cuda = kwargs['cuda']
    device = kwargs['device']
    model = kwargs['model']
    model.eval()
    if cuda:
        model.cuda(device=device)
    if 'writer' in kwargs.keys():
        # it means this is evaluation at training time
        val_loader = kwargs['val_loader']
        model = kwargs['model']
        writer = kwargs['writer']
        global_step = kwargs['global_step']
        crossentropy_criterion = kwargs['criterion']
        total_examples, total_correct, net_loss = 0, 0, []
        for idx, data in enumerate(val_loader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, softmaxed = model.forward(test_x)
            pred = torch.argmax(softmaxed, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            # dice_criterion(softmaxed, label) # + focal_criterion(softmaxed, not_one_hot_target) #
            loss = crossentropy_criterion(softmaxed, label.float())
            accurate = (pred == not_one_hot_target).sum().item()
            numerator = float(accurate)
            denominator = float(pred.view(-1).size(0)) #test_x.size(0) * dimension ** 2)
            # accuracy = float(numerator) * 100 / denominator
            total_correct += numerator
            total_examples += denominator
            net_loss.append(loss.item())
            #################################
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        writer.add_scalar(tag='val. loss', scalar_value=mean_loss, global_step=global_step)
        writer.add_scalar(tag='val. over_all accuracy', scalar_value=mean_accuracy, global_step=global_step)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('LOG: validation:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    else:
        # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
        num_classes = 3
        pre_model = kwargs['pre_model']
        un_confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=False)
        confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)

        model_path = os.path.join(kwargs['save_dir'], 'model-{}.pt'.format(pre_model))
        model.load_state_dict(torch.load(model_path), strict=False)
        print('log: resumed model {} successfully!'.format(pre_model))

        # weights = torch.Tensor([1, 1, 1])  # forest has ten times more weight
        # weights = weights.cuda(device=device) if cuda else weights
        # dice_criterion, focal_criterion = nn.CrossEntropyLoss(), DiceLoss(), FocalLoss2d()
        crossentropy_criterion = nn.BCELoss()
        loaders = get_dataloaders_generated_data(generated_data_path=kwargs['generated_data_path'],
                                                 save_data_path=kwargs['save_data'],
                                                 model_input_size=kwargs['input_dim'],
                                                 batch_size=kwargs['batch_size'],
                                                 one_hot=True,
                                                 num_workers=kwargs['workers'])
        train_loader, test_loader, empty_loader = loaders

        net_loss = []
        total_correct, total_examples = 0, 0
        net_class_accuracy_0, net_class_accuracy_1, net_class_accuracy_2, \
        net_class_accuracy_3, net_class_accuracy_4, net_class_accuracy_5,\
        net_class_accuracy_6  = [], [], [], [], [], [], []
        # net_class_accuracies = [[] for i in range(16)]
        classes_mean_accuracies = []
        for idx, data in enumerate(train_loader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, softmaxed = model.forward(test_x)
            pred = torch.argmax(softmaxed, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            loss = crossentropy_criterion(softmaxed, label.float()) # dice_criterion(softmaxed, label) # +
            accurate = (pred == not_one_hot_target).sum().item()
            numerator = float(accurate)
            denominator = float(pred.view(-1).size(0)) #test_x.size(0) * dimension ** 2)
            total_correct += numerator
            total_examples += denominator
            net_loss.append(loss.item())
            un_confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))
            confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))

            if idx % 10 == 0:
                print('log: on {}'.format(idx))

            # get per-class metrics
            # for k in range(num_classes):
            #     class_pred = (pred == k)
            #     class_label = (label == k)
            #     class_accuracy = (class_pred == class_label).sum()
            #     class_accuracy = class_accuracy * 100 / (pred.view(-1).size(0))
            #     net_class_accuracies[k].append(class_accuracy)

            # class_pred_0 = (pred == 0)
            # class_label_0 = (label == 0)
            # class_accuracy_0 = (class_pred_0 == class_label_0).sum()
            # class_accuracy_0 = class_accuracy_0 * 100 / (pred.view(-1).size(0))
            # net_class_accuracy_0.append(class_accuracy_0)
            #
            # class_pred_1 = (pred == 1)
            # class_label_1 = (label == 1)
            # class_accuracy_1 = (class_pred_1 == class_label_1).sum()
            # class_accuracy_1 = class_accuracy_1 * 100 / (pred.view(-1).size(0))
            # net_class_accuracy_1.append(class_accuracy_1)
            #
            # class_pred_2 = (pred == 2)
            # class_label_2 = (label == 2)
            # class_accuracy_2 = (class_pred_2 == class_label_2).sum()
            # class_accuracy_2 = class_accuracy_2 * 100 / (pred.view(-1).size(0))
            # net_class_accuracy_2.append(class_accuracy_2)
            #
            # class_pred_3 = (pred == 3)
            # class_label_3 = (label == 3)
            # class_accuracy_3 = (class_pred_3 == class_label_3).sum()
            # class_accuracy_3 = class_accuracy_3 * 100 / (pred.view(-1).size(0))
            # net_class_accuracy_3.append(class_accuracy_3)
            #
            # class_pred_4 = (pred == 4)
            # class_label_4 = (label == 4)
            # class_accuracy_4 = (class_pred_4 == class_label_4).sum()
            # class_accuracy_4 = class_accuracy_4 * 100 / (pred.view(-1).size(0))
            # net_class_accuracy_4.append(class_accuracy_4)
            #
            # class_pred_5 = (pred == 5)
            # class_label_5 = (label == 5)
            # class_accuracy_5 = (class_pred_5 == class_label_5).sum()
            # class_accuracy_5 = class_accuracy_5 * 100 / (pred.view(-1).size(0))
            # net_class_accuracy_5.append(class_accuracy_5)
            #
            # class_pred_6 = (pred == 6)
            # class_label_6 = (label == 6)
            # class_accuracy_6 = (class_pred_6 == class_label_6).sum()
            # class_accuracy_6 = class_accuracy_6 * 100 / (pred.view(-1).size(0))
            # net_class_accuracy_6.append(class_accuracy_6)

            # preds = torch.cat((preds, pred.long().view(-1)))
            # labs = torch.cat((labs, label.long().view(-1)))
            #################################
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()

        # for k in range(num_classes):
        #     classes_mean_accuracies.append(np.asarray(net_class_accuracies[k]).mean())
        #
        # class_0_mean_accuracy = np.asarray(net_class_accuracy_0).mean()
        # class_1_mean_accuracy = np.asarray(net_class_accuracy_1).mean()
        # class_2_mean_accuracy = np.asarray(net_class_accuracy_2).mean()
        # class_3_mean_accuracy = np.asarray(net_class_accuracy_3).mean()
        # class_4_mean_accuracy = np.asarray(net_class_accuracy_4).mean()
        # class_5_mean_accuracy = np.asarray(net_class_accuracy_5).mean()
        # class_6_mean_accuracy = np.asarray(net_class_accuracy_6).mean()

        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        # for k in range(num_classes):
        #     print('log: class {}:: total accuracy = {:.5f}%'.format(k, classes_mean_accuracies[k]))
        # print('log: class 0:: total accuracy = {:.5f}%'.format(class_0_mean_accuracy))
        # print('log: class 1:: total accuracy = {:.5f}%'.format(class_1_mean_accuracy))
        # print('log: class 2:: total accuracy = {:.5f}%'.format(class_2_mean_accuracy))
        # print('log: class 3:: total accuracy = {:.5f}%'.format(class_3_mean_accuracy))
        # print('log: class 4:: total accuracy = {:.5f}%'.format(class_4_mean_accuracy))
        # print('log: class 5:: total accuracy = {:.5f}%'.format(class_5_mean_accuracy))
        # print('log: class 6:: total accuracy = {:.5f}%'.format(class_6_mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        # class_names = ['background/clutter', 'buildings', 'trees', 'cars',
        #                'low_vegetation', 'impervious_surfaces', 'noise']
        with open('normalized.pkl', 'wb') as this:
            pkl.dump(confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)
        with open('un_normalized.pkl', 'wb') as this:
            pkl.dump(un_confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)
    pass










