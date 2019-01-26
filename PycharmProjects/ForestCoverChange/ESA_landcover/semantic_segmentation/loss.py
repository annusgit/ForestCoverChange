

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import os,sys,random,time
import argparse


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)                        # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class FocalLoss2d(nn.Module):
    # output : NxCxHxW Variable of float tensor
    # target :  NxHxW long tensor
    # weights : C float tensor
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs, dim=1)) ** self.gamma * F.log_softmax(inputs, dim=1), targets)


class DiceLoss(nn.Module):

    def __init__(self, weights=None, ignore_index=None, device='cpu'):
        super(DiceLoss, self).__init__()
        self.weights = weights
        self.ignore_index = ignore_index
        self.device = device
        pass

    def forward(self, output, target):
        # output : NxCxHxW Variable of float tensor
        # target :  NxHxW long tensor
        # weights : C float tensor
        # ignore_index : int value to ignore from loss
        smooth = 1.
        loss = 0.

        output = output.exp()  # computes the exponential of each element ie. for 0 it finds 10
        encoded_target = output.data.clone().zero_()  # make output size array and initialize with zeros
        # ignore_index=1

        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            unseq = target.long()  # here
            unseq = unseq.data  # here
            encoded_target.scatter_(1, unseq, 1)

        # encoded_target = Variable(encoded_target).device(self.device)

        if self.weights is None:
            self.weights = Variable(torch.ones(output.size(1)).type_as(output.data))

        intersection = output * encoded_target
        numerator = (2*intersection.sum(3).sum(2).sum(0) + smooth)
        denominator = ((output+encoded_target).sum(3).sum(2).sum(0) + smooth)
        loss_per_channel = self.weights*(1 -(numerator/denominator))  # weights may be directly multiplied

        return loss_per_channel.sum() / output.size(1)


# class DiceLoss(nn.Module):
#     def __init__(self, weights):
#         super(DiceLoss, self).__init__()
#         self.weights = weights
#
#     def forward(self, input, target):
#         smooth = 1.
#         iflat = input.view(-1)
#         tflat = target.view(-1)
#         intersection = (iflat * tflat).sum()
#         # return 1-((2.*intersection+smooth)/(iflat.sum()+tflat.sum()+smooth))
#         loss = 1-((2.*intersection+smooth)/(iflat.sum()+tflat.sum()+smooth))
#         # weighted from here onwards
#         loss_per_channel = self.weights * loss
#         return loss_per_channel / input.size(1)


class tversky_loss(nn.Module):
    def __init__(self, num_c):
        super(tversky_loss, self).__init__()
        self.alpha = 0.5
        self.beta = 0.5
        self.num_c = num_c

    def forward(self, y_pred, y_true):
        ones = torch.ones(y_true.shape)
        p0 = y_pred  # proba that voxels are class i
        p1 = ones - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = ones - y_true
        num = torch.sum(p0 * g0, (0, 1, 2, 3))
        # num = torch.reduce_sum(p0 * g0, dim=0)

        den = num + self.alpha * torch.sum(p0 * g1, (0, 1, 2, 3)) + self.beta * torch.sum(p1 * g0, (0, 1, 2, 3))
        T = torch.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]
        Ncl = torch.FloatTensor(self.num_c)
        return (Ncl - T).mean()


def check_focal_loss():
    num_c = 16
    weights = torch.Tensor([7, 2, 241, 500, 106, 5, 319, 0.06, 0.58, 0.125, 0.045, 0.18, 0.026, 0.506, 0.99, 0.321])
    out_x_np = np.random.randint(0, num_c, size=(16*64*64*num_c)).reshape((16, num_c, 64, 64))
    target_np = np.random.randint(0, num_c, size=(16*64*64*1)).reshape((16, 64, 64))
    logits = torch.Tensor(out_x_np)
    target = torch.LongTensor(target_np)
    weighted_loss = FocalLoss()
    loss_val = weighted_loss(logits, target)
    print(loss_val.item())


def check_focal_loss2d():
    num_c = 16
    weights = torch.Tensor([7, 2, 241, 500, 106, 5, 319, 0.06, 0.58, 0.125, 0.045, 0.18, 0.026, 0.506, 0.99, 0.321])
    out_x_np = np.random.randint(0, num_c, size=(16*64*64*num_c)).reshape((16, num_c, 64, 64))
    target_np = np.random.randint(0, num_c, size=(16*64*64*1)).reshape((16, 64, 64))

    logits = torch.Tensor(out_x_np)
    target = torch.LongTensor(target_np)
    weighted_loss = FocalLoss2d(weight=weights)
    loss_val = weighted_loss(logits, target)
    print("Focalloss2d: ", loss_val.item())


def check_dice_loss():
    num_c = 16
    weights = torch.Tensor([7, 2, 241, 500, 106, 5, 319, 0.06, 0.58, 0.125, 0.045, 0.18, 0.026, 0.506, 0.99, 0.321])
    weights.to('cpu')
    logits_np = np.random.randint(0, 16, size=(64*64*16*num_c)).reshape((16, num_c, 64, 64))
    target_np = np.random.randint(0, 16, size=(64*64*16*1)).reshape((16, 1, 64, 64))

    logits = torch.Tensor(logits_np)
    target = torch.LongTensor(target_np)
    weighted_loss = DiceLoss(weights=weights)
    loss_val = weighted_loss(output=logits, target=target)
    print("Diceloss: ", loss_val.item())


def check_tversky_loss():
    num_c = 16
    logits_np = np.random.randint(0, 16, size=(64 * 64 * 16 * num_c)).reshape((16, num_c, 64, 64))
    target_np = np.random.randint(0, 16, size=(64 * 64 * 16 * 1)).reshape((16, 1, 64, 64))

    logits = torch.Tensor(logits_np)
    target = torch.LongTensor(target_np)
    criterion = tversky_loss(num_c=num_c)
    loss_val = criterion(logits, target.float())
    print("Tversky: ", loss_val.item())


if __name__ == '__main__':
    # check_focal_loss()
    check_dice_loss()
    check_focal_loss2d()
    check_tversky_loss()




