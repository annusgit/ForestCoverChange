

"""
    Use Unet model (pretrained in Matlab) and evaluate it
"""

from __future__ import print_function
from __future__ import division
import torch
import os
import sys
import cv2
import numpy as np
from model import UNet
from scipy import ndimage
from skimage.measure import block_reduce


def convert_labels(label_im):
    # 29(0): background/clutter, 76(1): buildings, 150(2): trees,
    # 179(3): cars, 226(4): low_vegetation, 255(5): impervious_surfaces
    # conversions = {29:0, 76:1, 150:2, 179:3, 226:4, 255:5}
    # we are removing the cars class because of excessive downsampling used in the dataset, but not now...
    conversions = {29:0, 76:1, 150:2, 179:3, 226:4, 255:5}
    gray = cv2.cvtColor(label_im, cv2.COLOR_RGB2GRAY)
    for k in conversions.keys():
        gray[gray == k] = conversions[k]
    # print(np.unique(gray))
    return gray

if __name__ == '__main__':
    net = UNet(model_dir_path=sys.argv[1], input_channels=3)
    test_model = sys.argv[2]
    image_path = sys.argv[3]
    label_path = sys.argv[4]
    patch = int(sys.argv[5])
    net.load_state_dict(torch.load(test_model))
    net.cuda(device=0)

    image_read = cv2.imread(image_path)
    label_read = cv2.imread(label_path)
    small_patch = patch // 4
    full_i = image_read.shape[0]//small_patch
    full_j = image_read.shape[1]//small_patch
    # full_image = np.empty(shape=(small_patch*full_i+small_patch, full_j*small_patch+small_patch, 3))
    # full_label = np.empty(shape=(small_patch*full_i+small_patch, full_j*small_patch+small_patch))
    # full_pred = np.empty(shape=(small_patch*full_i+small_patch, full_j*small_patch+small_patch))
    x, y = image_read.shape[0]//2, image_read.shape[1]//2
    full_image = np.empty(shape=(x,y,3))
    full_label = np.empty(shape=(x,y))
    full_pred = np.empty(shape=(x,y))
    print(image_read.shape)

    net_accuracy = []
    for i in range(image_read.shape[0]//patch):
        for j in range(image_read.shape[1]//patch):
            image = image_read[patch*i:patch*i+patch, j*patch:j*patch+patch, :]
            image = np.dstack((image[:,:,2], image[:,:,1], image[:,:,0]))
            label = label_read[patch*i:patch*i+patch, j*patch:j*patch+patch, :]
            label = np.expand_dims(convert_labels(label_im=label), axis=2)

            for l in range(2):
                image = block_reduce(image=image, block_size=(2, 2, 1), func=np.max)
                label = block_reduce(image=label, block_size=(2, 2, 1), func=np.max)
            label = np.squeeze(label, axis=2)

            # save them
            # small_patch = patch
            image_copy = image.copy()
            label_copy = label.copy()
            full_image[small_patch*i:small_patch*i+small_patch,j*small_patch:j*small_patch+small_patch, :] = image_copy
            full_label[small_patch*i:small_patch*i+small_patch,j*small_patch:j*small_patch+small_patch] = label_copy

            print('log: w = {}-{}, h = {}-{}'.format(patch*i, patch*i+patch, j*patch, j*patch+patch), end='')

            test_x = torch.FloatTensor(image.transpose(2,0,1)).unsqueeze(0).cuda()
            out_x, pred = net(test_x)
            pred = pred.squeeze(0).cpu().numpy()
            pred_copy = pred.astype(np.uint8)
            full_pred[small_patch*i:small_patch*i+small_patch,j*small_patch:j*small_patch+small_patch] = pred_copy

            # fix predictions
            pred[pred == 0] = 8
            pred[pred == 1] = 0
            pred[pred == 8] = 1

            pred[pred == 3] = 8
            pred[pred == 4] = 3
            pred[pred == 8] = 4
            pred = ndimage.median_filter(pred, 3)

            # get accuracy metric
            accuracy = (pred == label).sum() * 100 / pred.reshape(-1).shape[0]
            print(': accuracy = {:.5f}%'.format(accuracy))
            net_accuracy.append(accuracy)
    mean_accuracy = np.asarray(net_accuracy).mean()
    print('total accuracy = {:.5f}%'.format(mean_accuracy))
    cv2.imwrite('image1.png', full_image)
    cv2.imwrite('label1.png', full_label)
    cv2.imwrite('pred1.png', full_pred)






