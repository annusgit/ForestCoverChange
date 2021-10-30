

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.model_zoo as model_zoo
from dataset import get_dataloaders
import os
import numpy as np
import pickle as pkl
import PIL.Image as Image
from tensorboardX import SummaryWriter
import itertools
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix
from torchnet.meter import ConfusionMeter as CM
import seaborn as sn
import pandas as pd



def train_net(model, images, labels, pre_model, save_dir, sum_dir,
              batch_size, lr, log_after, cuda, device):
    print(model)
    if cuda:
        print('GPU')
        model.cuda(device=device)
    # define loss and optimizer
    optimizer = RMSprop(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_dataloader, test_loader = get_dataloaders(images_path=images,
                                                                labels_path=labels,
                                                                batch_size=batch_size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(sum_dir):
        os.mkdir(sum_dir)
    writer = SummaryWriter()
    if True:
        i = 0
        m_loss, m_accuracy = [], []
        num_classes = 7
        if pre_model:
            # self.load_state_dict(torch.load(pre_model)['model'])
            model.load_state_dict(torch.load(pre_model))
            print('log: resumed model {} successfully!'.format(pre_model))
            model_number = int(pre_model.split('/')[1].split('-')[1].split('.')[0])
        else:
            print('log: starting anew...')
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
                # remember to save only five previous models, so
                del_this = os.path.join(save_dir, 'model-{}.pt'.format(i+model_number-6))
                if os.path.exists(del_this):
                    os.remove(del_this)
                    print('log: removed {}'.format(del_this))
                print('log: saved {}'.format(save_path))
            list_of_pred = []
            list_of_labels = []
            for idx, data in enumerate(train_loader):
                ##########################
                model.train()
                ##########################
                test_x, label = data['input'], data['label']
                image0 = test_x[0]
                test_x = test_x.cuda(device=device) if cuda else test_x
                size = test_x.size(-1)
                # forward
                out_x, pred = model.forward(test_x)
                pred = pred.cpu(); out_x = out_x.cpu()
                image1 = pred[0]
                image2 = label[0]
                if idx % (len(train_loader)/2) == 0:
                    writer.add_image('input', image0, i)
                    writer.add_image('pred', image1, i)
                    writer.add_image('label', image2, i)

                loss = criterion(out_x, label)
                # get accuracy metric
                accuracy = (pred == label).sum()
                # also convert into np arrays to be used for confusion matrix
                pred_np = pred.numpy(); list_of_pred.append(pred_np)
                label_np = label.numpy(); list_of_labels.append(label_np)

                writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=i)
                writer.add_scalar(tag='over_all accuracy',
                                  scalar_value=accuracy*100/(test_x.size(0)*size**2),
                                  global_step=i)

                # per class accuracies
                avg = []
                for j in range(num_classes):
                    class_pred = (pred == j)
                    class_label = (label == j)
                    class_accuracy = (class_pred == class_label).sum()
                    class_accuracy = class_accuracy * 100 / (test_x.size(0) * size ** 2)
                    avg.append(class_accuracy)
                    writer.add_scalar(tag='class_{} accuracy'.format(j), scalar_value=class_accuracy, global_step=i)
                classes_avg_acc = np.asarray(avg).mean()
                writer.add_scalar(tag='classes avg. accuracy', scalar_value=classes_avg_acc, global_step=i)

                if idx % log_after == 0 and idx > 0:
                    print('{}. ({}/{}) image size = {}, loss = {}: accuracy = {}/{}'.format(i,
                                                                                            idx,
                                                                                            len(train_loader),
                                                                                            out_x.size(),
                                                                                            loss.item(),
                                                                                            accuracy,
                                                                                            test_x.size(0)*size**2))
                #################################
                # three steps for backprop
                model.zero_grad()
                loss.backward()
                # perform gradient clipping between loss backward and optimizer step
                clip_grad_norm_(model.parameters(), 0.05)
                optimizer.step()
                accuracy = accuracy * 100 / (test_x.size(0)*size**2)
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

            # # one epoch complete, get new confusion matrix!
            # cm_preds = np.vstack(list_of_pred)
            # cm_preds = cm_preds.reshape(-1)
            # cm_labels = np.vstack(list_of_labels)
            # cm_labels = cm_labels.reshape(-1)
            # cnf_matrix = confusion_matrix(cm_labels, cm_preds)
            # fig1 = plt.figure()
            # plot_confusion_matrix(cnf_matrix, classes=class_names,
            #                       title='Confusion matrix, without normalization')
            # # Plot normalized confusion matrix
            # fig2 = plt.figure()
            # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
            #                       title='Normalized confusion matrix')
            # get1 = np.asarray(fig2img(fig1))
            # get2 = np.asarray(fig2img(fig2))
            # # print(get1.size)


            # validate model
            if i % 10 == 0:
                eval_net(model=model, criterion=criterion, val_loader=val_dataloader,
                         denominator=batch_size * size**2, cuda=cuda, device=device,
                         writer=writer, num_classes=num_classes, batch_size=batch_size, step=i)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    pass


def eval_net(**kwargs):
    cuda = kwargs['cuda']
    device = kwargs['device']
    model = kwargs['model']
    model.eval()

    if cuda:
        model.cuda(device=device)
    if 'writer' in kwargs.keys():
        num_classes = kwargs['num_classes']
        batch_size = kwargs['batch_size']
        writer = kwargs['writer']
        step = kwargs['step']
        denominator = kwargs['denominator']
        val_loader = kwargs['val_loader']
        model = kwargs['model']
        criterion = kwargs['criterion']
        net_accuracy, net_loss = [], []
        for idx, data in enumerate(val_loader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda() if cuda else test_x
            # forward
            out_x, pred = model.forward(test_x)
            pred = pred.cpu()
            loss = criterion(out_x.cpu(), label)
            # get accuracy metric
            accuracy = (pred == label).sum()
            accuracy = accuracy * 100 / (test_x.size(0)*64**2)
            net_accuracy.append(accuracy)
            net_loss.append(loss.item())

            # per class accuracies
            # avg = []
            # for j in range(num_classes):
            #     class_pred = (pred == j)
            #     class_label = (label == j)
            #     class_accuracy = (class_pred == class_label).sum()
            #     class_accuracy = class_accuracy * 100 / (batch_size * 32 ** 2)
            #     avg.append(class_accuracy)
            #     writer.add_scalar(tag='class_{} accuracy'.format(j), scalar_value=class_accuracy, global_step=step)
            # classes_avg_acc = np.asarray(avg).mean()
            # writer.add_scalar(tag='classes avg. accuracy', scalar_value=classes_avg_acc, global_step=step)

            #################################
        mean_accuracy = np.asarray(net_accuracy).mean()
        mean_loss = np.asarray(net_loss).mean()
        writer.add_scalar(tag='eval accuracy', scalar_value=mean_accuracy, global_step=step)
        writer.add_scalar(tag='eval loss', scalar_value=mean_loss, global_step=step)
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
        cm = CM(k=7, normalized=True)
        preds, labs = torch.Tensor().long(), torch.Tensor().long()

        model.load_state_dict(torch.load(pre_model))
        print('log: resumed model {} successfully!'.format(pre_model))
        _, _, test_loader = get_dataloaders(images_path=images, labels_path=labels, batch_size=batch_size)
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
                test_x = test_x.cuda()
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
