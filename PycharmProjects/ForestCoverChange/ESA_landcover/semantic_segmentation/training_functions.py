

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
from torchnet.meter import ConfusionMeter as CM


def train_net(model, generated_data_path, images, labels, block_size, input_dim, workers, pre_model,
              save_data, save_dir, sum_dir, batch_size, lr, epochs, log_after, cuda, device):
    # print(model)
    if cuda:
        print('log: Using GPU')
        model.cuda(device=device)
    # define loss and optimizer
    optimizer = RMSprop(model.parameters(), lr=lr)
    weights = torch.Tensor([7, 2, 241, 500, 106, 5, 319, 0.06, 0.58, 0.125, 0.045, 0.18, 0.026, 0.506, 0.99, 0.321])
    # weights = weights.cuda(device=device) if cuda else weights
    # criterion = nn.CrossEntropyLoss(weight=weights)
    # choose a better loss for this problem
    # if cuda:
    #     criterion = DiceLoss(weights=weights, device='cpu')
    # else:
    #     criterion = DiceLoss(weights=weights, device='cuda:{}'.format(device))
    # criterion = tversky_loss(num_c=16)
    criterion = FocalLoss2d()
    #### scheduler addition
    lr_final = 0.0000003
    LR_decay = (lr_final / lr) ** (1. / epochs)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=LR_decay)

    # loaders = get_dataloaders(images_path=images,
    #                           bands=range(1,14),
    #                           labels_path=labels,
    #                           save_data_path=save_data,
    #                           block_size=block_size,
    #                           model_input_size=input_dim,
    #                           batch_size=batch_size,
    #                           num_workers=workers)

    loaders = get_dataloaders_generated_data(generated_data_path=generated_data_path,
                                             save_data_path=save_data,
                                             model_input_size=input_dim, batch_size=batch_size, train_split=0.8,
                                             num_workers=workers, max_label=16)

    train_loader, val_dataloader, test_loader = loaders

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(sum_dir):
        os.mkdir(sum_dir)
    # writer = SummaryWriter()
    if pre_model == -1:
        model_number = 0
        print('log: No trained model passed. Starting from scratch...')
        # model_path = os.path.join(save_dir, 'model-{}.pt'.format(model_number))
    else:
        model_number = pre_model
        model_path = os.path.join(save_dir, 'model-{}.pt'.format(pre_model))
        model.load_state_dict(torch.load(model_path))
        print('log: Resuming from model {} ...'.format(model_path))
    ###############################################################################
    # training loop
    for k in range(epochs):
        net_loss = []
        net_accuracy = []
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
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            dimension = test_x.size(-1)
            out_x, logits = model.forward(test_x)
            pred = torch.argmax(logits, dim=1)

            # label = label.unsqueeze(1)
            # print(logits.shape, label.shape)
            # print(label[label > 15])

            # out_x, crit_label = out_x.cpu(), label.cpu().unsqueeze(1).float()
            # print(out_x.shape, crit_label.shape)
            loss = criterion(out_x, label)
            accurate = (pred == label).sum()

            numerator = accurate
            denominator = float(test_x.size(0)*dimension**2)
            accuracy = float(numerator)*100/denominator
            if idx % log_after == 0 and idx > 0:
                print('{}. ({}/{}) output size = {}, loss = {}, '
                      'accuracy = {}/{} = {:.2f}%'.format(k,
                                                          idx,
                                                          len(train_loader),
                                                          out_x.size(),
                                                          loss.item(),
                                                          numerator,
                                                          denominator,
                                                          accuracy))
            #################################
            # three steps for backprop
            model.zero_grad()
            loss.backward()
            # perform gradient clipping between loss backward and optimizer step
            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            net_accuracy.append(accuracy)
            net_loss.append(loss.item())
            #################################

        # this should be done at the end of epoch only
        scheduler.step()  # to dynamically change the learning rate
        mean_accuracy = np.asarray(net_accuracy).mean()
        mean_loss = np.asarray(net_loss).mean()
        print('####################################')
        print('LOG: epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(k, mean_loss, mean_accuracy))
        print('####################################')

        # validate model
        print('log: Evaluating now...')
        eval_net(model=model, criterion=criterion, val_loader=val_dataloader, cuda=cuda, device=device,
                 writer=None, batch_size=batch_size, step=k)
    pass


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
        criterion = kwargs['criterion']
        net_accuracy, net_loss = [], []
        for idx, data in enumerate(val_loader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            dimension = test_x.size(-1)
            out_x, softmaxed = model.forward(test_x)
            pred = torch.argmax(softmaxed, dim=1)
            # pred = pred.cpu()
            loss = criterion(out_x, label)
            accurate = (pred == label).sum()

            numerator = accurate
            denominator = float(test_x.size(0) * dimension ** 2)
            accuracy = float(numerator) * 100 / denominator

            net_accuracy.append(accuracy)
            net_loss.append(loss.item())
            #################################
        mean_accuracy = np.asarray(net_accuracy).mean()
        mean_loss = np.asarray(net_loss).mean()
        # writer.add_scalar(tag='eval accuracy', scalar_value=mean_accuracy, global_step=step)
        # writer.add_scalar(tag='eval loss', scalar_value=mean_loss, global_step=step)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('LOG: validation:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    else:
        # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
        pre_model = kwargs['pre_model']
        images = kwargs['images']
        labels = kwargs['labels']
        batch_size = kwargs['batch_size']
        criterion = nn.CrossEntropyLoss()
        cm = CM(k=7, normalized=True)
        preds, labs = torch.Tensor().long(), torch.Tensor().long()

        model.load_state_dict(torch.load(pre_model))
        print('log: resumed model {} successfully!'.format(pre_model))
        # _, _, test_loader = get_dataloaders(images_path=images, labels_path=labels, batch_size=batch_size)

        loaders = get_dataloaders(images_path=images,
                                  bands=range(1, 14),
                                  labels_path=labels,
                                  save_data_path=kwargs['save_dir'],
                                  block_size=kwargs['block_size'],
                                  model_input_size=kwargs['input_dim'],
                                  batch_size=batch_size,
                                  num_workers=kwargs['workers'])
        train_loader, val_dataloader, test_loader = loaders

        net_accuracy, net_loss = [], []
        net_class_accuracy_0, net_class_accuracy_1, net_class_accuracy_2, \
        net_class_accuracy_3, net_class_accuracy_4, net_class_accuracy_5,\
        net_class_accuracy_6  = [], [], [], [], [], [], []
        for idx, data in enumerate(test_loader):
            if idx == 400:
                break
            test_x, label = data['input'], data['label']
            # print(test_x.shape)
            if cuda:
                test_x = test_x.cuda(device=device)
            # forward
            out_x, pred = model.forward(test_x)
            pred = pred.cpu()
            loss = criterion(out_x.cpu(), label)

            # get accuracy metric
            accuracy = (pred == label).sum()
            accuracy = accuracy * 100 / (pred.view(-1).size(0))
            net_accuracy.append(accuracy)
            net_loss.append(loss.item())
            if idx % 10 == 0:
                print('log: on {}'.format(idx))

            # print(pred.view(-1).size(0))
            # get per-class metrics
            class_pred_0 = (pred == 0)
            class_label_0 = (label == 0)
            class_accuracy_0 = (class_pred_0 == class_label_0).sum()
            class_accuracy_0 = class_accuracy_0 * 100 / (pred.view(-1).size(0))
            net_class_accuracy_0.append(class_accuracy_0)

            class_pred_1 = (pred == 1)
            class_label_1 = (label == 1)
            class_accuracy_1 = (class_pred_1 == class_label_1).sum()
            class_accuracy_1 = class_accuracy_1 * 100 / (pred.view(-1).size(0))
            net_class_accuracy_1.append(class_accuracy_1)

            class_pred_2 = (pred == 2)
            class_label_2 = (label == 2)
            class_accuracy_2 = (class_pred_2 == class_label_2).sum()
            class_accuracy_2 = class_accuracy_2 * 100 / (pred.view(-1).size(0))
            net_class_accuracy_2.append(class_accuracy_2)

            class_pred_3 = (pred == 3)
            class_label_3 = (label == 3)
            class_accuracy_3 = (class_pred_3 == class_label_3).sum()
            class_accuracy_3 = class_accuracy_3 * 100 / (pred.view(-1).size(0))
            net_class_accuracy_3.append(class_accuracy_3)

            class_pred_4 = (pred == 4)
            class_label_4 = (label == 4)
            class_accuracy_4 = (class_pred_4 == class_label_4).sum()
            class_accuracy_4 = class_accuracy_4 * 100 / (pred.view(-1).size(0))
            net_class_accuracy_4.append(class_accuracy_4)

            class_pred_5 = (pred == 5)
            class_label_5 = (label == 5)
            class_accuracy_5 = (class_pred_5 == class_label_5).sum()
            class_accuracy_5 = class_accuracy_5 * 100 / (pred.view(-1).size(0))
            net_class_accuracy_5.append(class_accuracy_5)

            class_pred_6 = (pred == 6)
            class_label_6 = (label == 6)
            class_accuracy_6 = (class_pred_6 == class_label_6).sum()
            class_accuracy_6 = class_accuracy_6 * 100 / (pred.view(-1).size(0))
            net_class_accuracy_6.append(class_accuracy_6)

            preds = torch.cat((preds, pred.long().view(-1)))
            labs = torch.cat((labs, label.long().view(-1)))
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

        # class_names = ['background/clutter', 'buildings', 'trees', 'cars',
        #                'low_vegetation', 'impervious_surfaces', 'noise']
        # cm_preds = pred.view(-1).cpu().numpy()
        # cm_labels = label.view(-1).cpu().numpy()
        # cnf_matrix = confusion_matrix(cm_labels, cm_preds)
        #
        # fig1 = plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                       title='Confusion matrix, without normalization')
        # # Plot normalized confusion matrix
        # fig2 = plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        #                       title='Normalized confusion matrix')
        # fig2img(fig1).save('unnormalized.png')
        # fig2img(fig2).save('normalized.png')

        #
        # cm.add(preds.view(-1), labs.view(-1).type(torch.LongTensor))
        # this = cm.value()
        # print(this)
        # df_cm = pd.DataFrame(this, index=[f for f in class_names],
        #                      columns=[f for f in class_names])
        # fig = plt.figure(figsize=(10, 7))
        # sn.heatmap(df_cm, annot=True)
        # fig2img(fig).save('sea.png')
    pass
