

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
from loss import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.model_zoo as model_zoo
from dataset import get_dataloaders_generated_data
import os
import numpy as np
import pickle as pkl
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchnet as tnt
from torchnet.meter import ConfusionMeter as CM


def train_net(model, generated_data_path, input_dim, workers, pre_model, save_data, save_dir, sum_dir, batch_size,
              lr, epochs, log_after, cuda, device):
    # print(model)
    if cuda:
        print('log: Using GPU')
        model.cuda(device=device)
    ###############################################################################

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(sum_dir):
        os.mkdir(sum_dir)
    # writer = SummaryWriter()

    # define loss and optimizer
    optimizer = RMSprop(model.parameters(), lr=lr)
    weights = torch.Tensor([1, 1]) # forest has ____ times more weight
    weights = weights.cuda(device=device) if cuda else weights
    focal_criterion = FocalLoss2d(weight=weights)
    # crossentropy_criterion = nn.BCELoss(weight=weights)
    # dice_criterion = DiceLoss(weights=weights)

    lr_final = 5e-5
    LR_decay = (lr_final / lr) ** (1. / epochs)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=LR_decay)

    loaders = get_dataloaders_generated_data(generated_data_path=generated_data_path, save_data_path=save_data,
                                             model_input_size=input_dim, batch_size=batch_size, num_classes=2,
                                             train_split=0.8, one_hot=True, num_workers=workers, max_label=1)
    train_loader, val_dataloader, test_loader = loaders
    best_evaluation = 0.0
    ################################################################
    if pre_model == -1:
        model_number = 0
        print('log: No trained model passed. Starting from scratch...')
        # model_path = os.path.join(save_dir, 'model-{}.pt'.format(model_number))
    else:
        model_number = pre_model
        model_path = os.path.join(save_dir, 'model-{}.pt'.format(pre_model))
        model.load_state_dict(torch.load(model_path), strict=False)
        print('log: Resuming from model {} ...'.format(model_path))
        print('log: Evaluating now...')
        best_evaluation = eval_net(model=model, criterion=focal_criterion, val_loader=val_dataloader,
                                   cuda=cuda, device=device, writer=None, batch_size=batch_size, step=0)
        print('LOG: Starting with best evaluation accuracy: {:.3f}%'.format(best_evaluation))
    ##########################################################################

    # training loop
    for k in range(epochs):
        net_loss = []
        total_correct, total_examples = 0, 0
        for idx, data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, logits = model.forward(test_x)
            pred = torch.argmax(logits, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            # dice_criterion(logits, label) #+ focal_criterion(logits, not_one_hot_target) #
            # print(logits.view(batch_size, -1).shape, logits.view(batch_size, -1).shape)
            # loss = focal_criterion(logits.view(-1, 2), label.view(-1, 2))
            loss = focal_criterion(logits, not_one_hot_target) # dice_criterion(logits, label) #
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
                      'accuracy = {}/{} = {:.2f}%'.format(k, idx, len(train_loader), out_x.size(), loss.item(),
                                                          numerator, denominator, accuracy))
            net_loss.append(loss.item())

        # this should be done at the end of epoch only
        scheduler.step()  # to dynamically change the learning rate
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        print('####################################')
        print('LOG: epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(k, mean_loss, mean_accuracy))
        print('####################################')

        # validate model
        print('log: Evaluating now...')
        eval_accuracy = eval_net(model=model, criterion=focal_criterion, val_loader=val_dataloader,
                                 cuda=cuda, device=device, writer=None, batch_size=batch_size, step=k)

        # save best performing models only
        if eval_accuracy > best_evaluation:
            best_evaluation = eval_accuracy
            model_number += 1
            model_path = os.path.join(save_dir, 'model-{}.pt'.format(model_number))
            torch.save(model.state_dict(), model_path)
            print('log: Saved best performing {}'.format(model_path))

            del_this = os.path.join(save_dir, 'model-{}.pt'.format(model_number-6))
            if os.path.exists(del_this):
                os.remove(del_this)
                print('log: Removed {}'.format(del_this))
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
        focal_criterion = kwargs['criterion']
        total_examples, total_correct, net_loss = 0, 0, []
        num_classes = 2
        un_confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=False)
        confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
        for idx, data in enumerate(val_loader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, softmaxed = model.forward(test_x)
            pred = torch.argmax(softmaxed, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            # dice_criterion(softmaxed, label) # + focal_criterion(softmaxed, not_one_hot_target) #
            # loss = crossentropy_criterion(softmaxed.view(-1, 2), label.view(-1, 2))
            loss = focal_criterion(softmaxed, not_one_hot_target) #dice_criterion(softmaxed, label) #
            accurate = (pred == not_one_hot_target).sum().item()
            numerator = float(accurate)
            denominator = float(pred.view(-1).size(0)) #test_x.size(0)*dimension**2)
            total_correct += numerator
            total_examples += denominator
            net_loss.append(loss.item())
            un_confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))
            confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))
            #################################
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        # writer.add_scalar(tag='eval accuracy', scalar_value=mean_accuracy, global_step=step)
        # writer.add_scalar(tag='eval loss', scalar_value=mean_loss, global_step=step)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('LOG: validation: total loss = {:.5f}, total accuracy = ({}/{}) = {:.5f}%'.format(mean_loss,
                                                                                                total_correct,
                                                                                                total_examples,
                                                                                                mean_accuracy))
        print('Log: Confusion matrix')
        print(confusion_meter.value())
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        return mean_accuracy

    else:
        # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
        pre_model = kwargs['pre_model']
        batch_size = kwargs['batch_size']
        num_classes = 2  # we convert to a binary classification problem at test time only
        un_confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=False)
        confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
        model_path = os.path.join(kwargs['save_dir'], 'model-{}.pt'.format(pre_model))
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        print('log: resumed model {} successfully!'.format(pre_model))
        weights = torch.Tensor([1, 1])  # forest has ten times more weight
        weights = weights.cuda(device=device) if cuda else weights
        # dice_criterion, focal_criterion = nn.CrossEntropyLoss(), DiceLoss(), FocalLoss2d()
        # crossentropy_criterion = nn.BCELoss(weight=weights)
        focal_criterion = FocalLoss2d(weight=weights)
        # dice_criterion = DiceLoss(weights=weights)
        loaders = get_dataloaders_generated_data(generated_data_path=kwargs['generated_data_path'],
                                                 save_data_path=kwargs['save_data'],
                                                 model_input_size=kwargs['input_dim'],
                                                 batch_size=batch_size,
                                                 # train_split=0.8,
                                                 one_hot=True,
                                                 num_workers=kwargs['workers'],
                                                 max_label=num_classes)
        train_loader, test_loader, empty_loader = loaders
        net_loss = []
        total_correct, total_examples = 0, 0
        for idx, data in enumerate(test_loader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, softmaxed = model.forward(test_x)
            pred = torch.argmax(softmaxed, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            '''
                Not needed anymore, forest is already 0 and non-forest has label 1
            # convert to binary classes
            # 0-> noise, 1-> forest, 2-> non-forest, 3-> water
            # pred[pred == 0] = 2
            # pred[pred == 3] = 2
            # not_one_hot_target[not_one_hot_target == 0] = 2
            # not_one_hot_target[not_one_hot_target == 3] = 2
            # # now convert 1, 2 to 0, 1
            # pred -= 1
            # not_one_hot_target -= 1
            '''
            # dice_criterion(softmaxed, label) # +
            # print(softmaxed.shape, label.shape)
            # loss = crossentropy_criterion(softmaxed.view(-1, 2), label.view(-1, 2))
            loss = focal_criterion(softmaxed, not_one_hot_target) # dice_criterion(softmaxed, label) #
            accurate = (pred == not_one_hot_target).sum().item()
            numerator = float(accurate)
            denominator = float(pred.view(-1).size(0))  # test_x.size(0)*dimension**2)
            total_correct += numerator
            total_examples += denominator
            net_loss.append(loss.item())
            un_confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))
            confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))
            if idx % 10 == 0:
                print('log: on {}'.format(idx))
            #################################
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('---> Confusion Matrix:')
        print(confusion_meter.value())
        with open('normalized.pkl', 'wb') as this:
            pkl.dump(confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)
        with open('un_normalized.pkl', 'wb') as this:
            pkl.dump(un_confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)
    pass
