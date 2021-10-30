from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
from loss import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.model_zoo as model_zoo
from dataset import fix, get_dataloaders_generated_data
import os
import numpy as np
import pickle as pkl
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import ListedColormap
plt.switch_backend('agg')
import torchnet as tnt
from torchnet.meter import ConfusionMeter as CM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def train_net(model, model_topology, generated_data_path, input_dim, bands, classes, workers, pre_model, data_split_lists, save_dir, sum_dir, error_maps_path,
              batch_size, lr, epochs, log_after, cuda, device):
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
    # save our initial learning rate
    lr_initial = lr
    weights = torch.Tensor([10, 10])  # forest has ____ times more weight
    weights = weights.cuda(device=device) if cuda else weights
    focal_criterion = FocalLoss2d(weight=weights)
    lr_final = lr / 10  # get to one tenth of the starting rate
    LR_decay = (lr_final / lr) ** (1. / epochs)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=LR_decay)
    loaders = get_dataloaders_generated_data(generated_data_path=generated_data_path, data_split_lists_path=data_split_lists, model_input_size=input_dim,
                                             bands=bands, batch_size=batch_size, num_classes=len(classes)+1, train_split=0.8, one_hot=True,
                                             num_workers=workers)
    train_loader, val_dataloader, test_loader = loaders
    best_evaluation = 0.0
    ################################################################
    if pre_model == 'None':
        model_number = 0
        print('log: No trained model passed. Starting from scratch...')
    else:
        model_path = os.path.join(save_dir, pre_model)
        model_number = int(pre_model.split('/')[-1].split('_')[1])
        model.load_state_dict(torch.load(model_path), strict=False)
        print('log: Resuming from model {} ...'.format(model_path))
        print('log: Evaluating now...')
        best_evaluation = eval_net(model=model, criterion=focal_criterion, val_loader=val_dataloader, cuda=cuda, device=device, writer=None,
                                   batch_size=batch_size, step=0)
        print('LOG: Starting with best evaluation accuracy: {:.3f}%'.format(best_evaluation))
    ##########################################################################
    # training loop
    bands_for_training = [x-1 for x in bands]
    for k in range(epochs):
        net_loss = []
        total_correct, total_examples = 0, 0
        print('log: Evaluating now...')
        eval_net(model=model, classes=classes, criterion=focal_criterion, val_loader=val_dataloader, cuda=cuda, device=device, writer=None,
                 batch_size=batch_size, step=k)
        model_number += 1
        model_path = os.path.join(save_dir, 'model_{}_topology{}_lr{}_bands{}.pt'.format(model_number, model_topology, lr_initial, len(bands)))
        torch.save(model.state_dict(), model_path)
        print('log: Saved best performing {}'.format(model_path))
        # we will save all models for now
        # del_this = os.path.join(save_dir, 'model-{}.pt'.format(model_number-10))
        # if os.path.exists(del_this):
        #     os.remove(del_this)
        #     print('log: Removed {}'.format(del_this))
        for idx, data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            # get the required bands for training
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, logits = model.forward(test_x[:,bands_for_training,:,:])
            pred = torch.argmax(logits, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            loss = focal_criterion(logits, not_one_hot_target)
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            accurate = (pred == not_one_hot_target).sum().item()
            numerator = float(accurate)
            denominator = float(pred.view(-1).size(0))
            total_correct += numerator
            total_examples += denominator
            if idx % log_after == 0 and idx > 0:
                accuracy = float(numerator) * 100 / denominator
                print('{}. ({}/{}) input size= {}, output size = {}, loss = {}, accuracy = {}/{} = {:.2f}%'.format(k, idx, len(train_loader), test_x.size(),
                                                                                                                   out_x.size(), loss.item(), numerator,
                                                                                                                   denominator, accuracy))
            net_loss.append(loss.item())
        # this should be done at the end of epoch only
        scheduler.step()  # to dynamically change the learning rate
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        print('####################################')
        print('LOG: epoch {} -> total loss = {:.5f}, total accuracy = {:.5f}%'.format(k, mean_loss, mean_accuracy))
        print('####################################')
    pass


@torch.no_grad()
def eval_net(**kwargs):
    model = kwargs['model']
    classes = kwargs['classes']
    num_classes = len(classes)
    cuda = kwargs['cuda']
    device = kwargs['device']
    model.eval()
    all_predictions = np.array([])  # empty all predictions
    all_ground_truth = np.array([])
    if cuda:
        model.cuda(device=device)
    bands_for_testing = [x - 1 for x in kwargs['bands']]
    if 'writer' in kwargs.keys():
        # it means this is evaluation at training time
        val_loader = kwargs['val_loader']
        model = kwargs['model']
        focal_criterion = kwargs['criterion']
        total_examples, total_correct, net_loss = 0, 0, []
        un_confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=False)
        confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
        for idx, data in enumerate(val_loader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, softmaxed = model.forward(test_x[:,bands_for_testing,:,:])
            pred = torch.argmax(softmaxed, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            not_one_hot_target_for_loss = not_one_hot_target.clone()
            not_one_hot_target_for_loss[not_one_hot_target_for_loss == 0] = 1
            not_one_hot_target_for_loss -= 1
            loss = focal_criterion(softmaxed, not_one_hot_target_for_loss)  # dice_criterion(softmaxed, label) #
            label_valid_indices = (not_one_hot_target.view(-1) != 0)
            # mind the '-1' fix please. This is to convert Forest and Non-Forest labels from 1, 2 to 0, 1
            valid_label = not_one_hot_target.view(-1)[label_valid_indices] - 1
            valid_pred = pred.view(-1)[label_valid_indices]
            # Eliminate NULL pixels from testing
            accurate = (valid_pred == valid_label).sum().item()
            numerator = float(accurate)
            denominator = float(valid_pred.view(-1).size(0))
            total_correct += numerator
            total_examples += denominator
            net_loss.append(loss.item())
            # NULL elimination
            un_confusion_meter.add(predicted=valid_pred.view(-1), target=valid_label.view(-1))
            confusion_meter.add(predicted=valid_pred.view(-1), target=valid_label.view(-1))
            all_predictions = np.concatenate((all_predictions, valid_pred.view(-1).cpu()), axis=0)
            all_ground_truth = np.concatenate((all_ground_truth, valid_label.view(-1).cpu()), axis=0)
            #################################
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        # writer.add_scalar(tag='eval accuracy', scalar_value=mean_accuracy, global_step=step)
        # writer.add_scalar(tag='eval loss', scalar_value=mean_loss, global_step=step)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('LOG: validation: total loss = {:.5f}, total accuracy = ({}/{}) = {:.5f}%'.format(mean_loss, total_correct, total_examples, mean_accuracy))
        print('Log: Confusion matrix')
        print(confusion_meter.value())
        confusion = confusion_matrix(all_ground_truth, all_predictions)
        print('Confusion Matrix from Scikit-Learn\n')
        print(confusion)
        print('\nClassification Report\n')
        print(classification_report(all_ground_truth, all_predictions, target_names=classes))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    else:
        # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
        pre_model = kwargs['pre_model']
        batch_size = kwargs['batch_size']
        un_confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=False)
        confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
        model_path = os.path.join(kwargs['save_dir'], pre_model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        print('log: resumed model {} successfully!'.format(pre_model))
        weights = torch.Tensor([10, 10])  # forest has ___ times more weight
        weights = weights.cuda(device=device) if cuda else weights
        focal_criterion = FocalLoss2d(weight=weights)
        loaders = get_dataloaders_generated_data(generated_data_path=kwargs['generated_data_path'], data_split_lists_path=kwargs['data_split_lists'],
                                                 bands=kwargs['bands'], model_input_size=kwargs['input_dim'], num_classes=num_classes, train_split=0.8,
                                                 one_hot=True, batch_size=batch_size, num_workers=kwargs['workers'])
        net_loss = list()
        train_dataloader, val_dataloader, test_dataloader = loaders
        total_correct, total_examples = 0, 0
        print("(LOG): Evaluating performance on test data...")
        for idx, data in enumerate(test_dataloader):
            test_x, label = data['input'], data['label']
            test_x = test_x.cuda(device=device) if cuda else test_x
            label = label.cuda(device=device) if cuda else label
            out_x, softmaxed = model.forward(test_x[:,bands_for_testing,:,:])
            pred = torch.argmax(softmaxed, dim=1)
            not_one_hot_target = torch.argmax(label, dim=1)
            #######################################################
            not_one_hot_target_for_loss = not_one_hot_target.clone()
            not_one_hot_target_for_loss[not_one_hot_target_for_loss == 0] = 1
            not_one_hot_target_for_loss -= 1
            loss = focal_criterion(softmaxed, not_one_hot_target_for_loss)
            label_valid_indices = (not_one_hot_target.view(-1) != 0)
            # mind the '-1' fix please. This is to convert Forest and Non-Forest labels from 1, 2 to 0, 1
            valid_label = not_one_hot_target.view(-1)[label_valid_indices] - 1
            valid_pred = pred.view(-1)[label_valid_indices]
            # NULL elimination
            accurate = (valid_pred == valid_label).sum().item()
            numerator = float(accurate)
            denominator = float(valid_pred.view(-1).size(0))
            total_correct += numerator
            total_examples += denominator
            net_loss.append(loss.item())
            ########################################
            # with NULL elimination
            un_confusion_meter.add(predicted=valid_pred.view(-1), target=valid_label.view(-1))
            confusion_meter.add(predicted=valid_pred.view(-1), target=valid_label.view(-1))
            all_predictions = np.concatenate((all_predictions, valid_pred.view(-1).cpu()), axis=0)
            all_ground_truth = np.concatenate((all_ground_truth, valid_label.view(-1).cpu()), axis=0)
            if idx % 10 == 0:
                print('log: on test sample: {}/{}'.format(idx, len(test_dataloader)))
            #################################
        mean_accuracy = total_correct*100/total_examples
        mean_loss = np.asarray(net_loss).mean()
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('---> Confusion Matrix:')
        print(confusion_meter.value())
        confusion = confusion_matrix(all_ground_truth, all_predictions)
        print('Confusion Matrix from Scikit-Learn\n')
        print(confusion)
        print('\nClassification Report\n')
        print(classification_report(all_ground_truth, all_predictions, target_names=classes))
        with open('normalized.pkl', 'wb') as this:
            pkl.dump(confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)
        with open('un_normalized.pkl', 'wb') as this:
            pkl.dump(un_confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)
            pass
        pass
    pass


@torch.no_grad()
def generate_error_maps(**kwargs):
    model = kwargs['model']
    classes = kwargs['classes']
    num_classes = len(classes)
    cuda = kwargs['cuda']
    device = kwargs['device']
    model.eval()
    all_predictions = np.array([])  # empty all predictions
    all_ground_truth = np.array([])
    # special variables
    all_but_chitral_and_upper_dir_predictions = np.array([])  # empty all predictions
    all_but_chitral_and_upper_dir_ground_truth = np.array([])
    if cuda:
        model.cuda(device=device)
    # model, images, labels, pre_model, save_dir, sum_dir, batch_size, lr, log_after, cuda
    pre_model = kwargs['pre_model']
    batch_size = kwargs['batch_size']
    un_confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=False)
    confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
    model_path = os.path.join(kwargs['save_dir'], pre_model)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    print('[LOG] Resumed model {} successfully!'.format(pre_model))
    destination_path = os.path.join(kwargs['error_maps_path'], pre_model.split('.')[0])
    if not os.path.exists(os.path.join(destination_path)):
        os.mkdir(destination_path)
    weights = torch.Tensor([10, 10])  # forest has ___ times more weight
    weights = weights.cuda(device=device) if cuda else weights
    focal_criterion = FocalLoss2d(weight=weights)
    loaders = get_dataloaders_generated_data(generated_data_path=kwargs['generated_data_path'], data_split_lists_path=kwargs['data_split_lists'],
                                             bands=kwargs['bands'], model_input_size=kwargs['input_dim'], num_classes=num_classes+1, train_split=0.8,
                                             one_hot=True, batch_size=batch_size, num_workers=kwargs['workers'])
    net_loss = list()
    train_dataloader, val_dataloader, test_dataloader = loaders
    total_correct, total_examples = 0, 0
    print("[LOG] Evaluating performance on test data...")
    forest_cmap = ListedColormap(["yellow", "green"])
    true_false_cmap = ListedColormap(['red', 'blue'])
    bands_for_testing = [x - 1 for x in kwargs['bands']]
    accuracy_per_district = defaultdict(lambda: [0, 0])
    for idx, data in enumerate(test_dataloader):
        test_x, label, sample_identifiers = data['input'], data['label'], data['sample_identifier']
        test_x = test_x.cuda(device=device) if cuda else test_x
        label = label.cuda(device=device) if cuda else label
        out_x, softmaxed = model.forward(test_x[:,bands_for_testing,:,:])
        pred = torch.argmax(softmaxed, dim=1)
        not_one_hot_target = torch.argmax(label, dim=1)
        for i in range(not_one_hot_target.shape[0]):
            image_name = sample_identifiers[0][i].split('/')[-1].split('.')[0]
            district_name = image_name.split('_')[0]
            if district_name == 'upper' or district_name == 'lower':
                district_name += ' dir'
            # print(district_name)
            rgb_image = (255*(test_x.numpy()[i].transpose(1, 2, 0)[:,:,[3, 2, 1]])).astype(np.uint8)
            district_ground_truth = not_one_hot_target[i,:,:].clone()
            ground_truth = not_one_hot_target[i,:,:] - 1
            ground_truth[ground_truth < 0] = 0
            district_prediction = pred[i,:,:]
            error_map = np.array(ground_truth == district_prediction).astype(np.uint8)
            # calculate accuracy for this district image (below)
            district_label_valid_indices = (district_ground_truth.view(-1) != 0)
            district_valid_label = district_ground_truth.view(-1)[district_label_valid_indices] - 1
            district_valid_pred = district_prediction.view(-1)[district_label_valid_indices]
            district_accurate = (district_valid_pred == district_valid_label).sum().item()
            district_total_pixels = float(district_valid_pred.view(-1).size(0))
            accuracy_per_district[district_name][0] += district_accurate
            accuracy_per_district[district_name][1] += district_total_pixels
            # special variables
            if district_name != "upper dir" and district_name != "chitral":
                all_but_chitral_and_upper_dir_predictions = np.concatenate((all_but_chitral_and_upper_dir_predictions,
                                                                            district_valid_pred.view(-1).cpu()), axis=0)
                all_but_chitral_and_upper_dir_ground_truth = np.concatenate((all_but_chitral_and_upper_dir_ground_truth,
                                                                             district_valid_label.view(-1).cpu()), axis=0)
            # # calculate accuracy for this district image (above)
            # fig = plt.figure(figsize=(12,3))
            # fig.suptitle("[Non-Forest: Yellow; Forest: Green;] Error: [Correct: Blue, In-correct: Red]", fontsize="x-large")
            # ax1 = fig.add_subplot(1, 4, 1)
            # ax1.imshow(rgb_image)
            # ax1.set_title('Image')
            # ax2 = fig.add_subplot(1, 4, 2)
            # ax2.imshow(ground_truth, cmap=forest_cmap, vmin=0, vmax=2)
            # ax2.set_title('Ground Truth')
            # ax3 = fig.add_subplot(1, 4, 3)
            # ax3.imshow(district_prediction, cmap=forest_cmap, vmin=0, vmax=2)
            # ax3.set_title('Prediction')
            # ax4 = fig.add_subplot(1, 4, 4)
            # ax4.imshow(error_map, cmap=true_false_cmap, vmin=0, vmax=1)
            # ax4.set_title('Error')
            # fig.savefig(os.path.join(destination_path, '{}.png'.format(image_name)))
            # plt.close()
        #######################################################
        not_one_hot_target_for_loss = not_one_hot_target.clone()
        not_one_hot_target_for_loss[not_one_hot_target_for_loss == 0] = 1
        not_one_hot_target_for_loss -= 1
        loss = focal_criterion(softmaxed, not_one_hot_target_for_loss)
        label_valid_indices = (not_one_hot_target.view(-1) != 0)
        # mind the '-1' fix please. This is to convert Forest and Non-Forest labels from 1, 2 to 0, 1
        valid_label = not_one_hot_target.view(-1)[label_valid_indices] - 1
        valid_pred = pred.view(-1)[label_valid_indices]
        # NULL elimination
        accurate = (valid_pred == valid_label).sum().item()
        numerator = float(accurate)
        denominator = float(valid_pred.view(-1).size(0))
        total_correct += numerator
        total_examples += denominator
        net_loss.append(loss.item())
        ########################################
        # with NULL elimination
        un_confusion_meter.add(predicted=valid_pred.view(-1), target=valid_label.view(-1))
        confusion_meter.add(predicted=valid_pred.view(-1), target=valid_label.view(-1))
        all_predictions = np.concatenate((all_predictions, valid_pred.view(-1).cpu()), axis=0)
        all_ground_truth = np.concatenate((all_ground_truth, valid_label.view(-1).cpu()), axis=0)
        if idx % 10 == 0:
            print('log: on test sample: {}/{}'.format(idx, len(test_dataloader)))
        #################################
    mean_accuracy = total_correct*100/total_examples
    mean_loss = np.asarray(net_loss).mean()
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('---> Confusion Matrix:')
    print(confusion_meter.value())
    confusion = confusion_matrix(all_ground_truth, all_predictions)
    print('Confusion Matrix from Scikit-Learn\n')
    print(confusion)
    print('\nClassification Report\n')
    print(classification_report(all_ground_truth, all_predictions, target_names=classes))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # Get per district test scores without Upper Dir and Chitral districts
    print('[LOG] Per District Test Accuracies')
    print(accuracy_per_district)
    numerator_sum, denominator_sum = 0, 0
    for idx, (this_district, [true, total]) in enumerate(accuracy_per_district.items(), 1):
        print("{}: {} -> {}/{} = {:.2f}%".format(idx, this_district, true, total, 100*true/total))
        if this_district != 'upper dir' and this_district != 'chitral':
            numerator_sum += true
            denominator_sum += total
        else:
            print("[LOG] Skipping {} district for performance testing".format(this_district))
    print("[LOG] Net Test Accuracy Without Chitral and Upper Dir: {:.2f}%".format(100*numerator_sum/denominator_sum))
    print('---> Confusion Matrix:')
    print(confusion_meter.value())
    confusion = confusion_matrix(all_but_chitral_and_upper_dir_ground_truth, all_but_chitral_and_upper_dir_predictions)
    print('Confusion Matrix from Scikit-Learn\n')
    print(confusion)
    print('\nClassification Report\n')
    print(classification_report(all_but_chitral_and_upper_dir_ground_truth, all_but_chitral_and_upper_dir_predictions,
                                target_names=classes))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    pass
