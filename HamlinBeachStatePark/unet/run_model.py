

"""
    Model for Unet (pretrained in Matlab)
"""

from __future__ import print_function
from __future__ import division
import torch
import os
import numpy as np
from model import UNet


if __name__ == '__main__':

    dir_path = '/home/annuszulfiqar/forest_cover/'
    batch_size, channels, patch = 1, 6, 256
    if True:
        net = UNet(model_dir_path=os.path.join(dir_path, 'Unet_pretrained_model.pkl'), input_channels=channels).cuda()
        val_data = np.load('rit18_data/val_data.npy', mmap_mode='r')
        val_label = np.load('rit18_data/val_labels.npy', mmap_mode='r')
        val_data = val_data.transpose((1, 2, 0))
        print('val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
        # pad the data to be divisible by patch size
        val_data = np.pad(val_data, ((0,patch-val_data.shape[0]%patch),
                                     (0,patch-val_data.shape[1]%patch),
                                     (0, 0)), 'constant')
        val_label = np.pad(val_label, ((0, patch - val_label.shape[0] % patch),
                                       (0, patch - val_label.shape[1] % patch)), 'constant')
        net_accuracy = []
        mean = np.load('mean.npy')
        zero_count = 0
        veg_count = 0
        print('val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
        for i in range(val_data.shape[0]//patch):
            for j in range(val_data.shape[1]//patch):
                image = val_data[patch*i:patch*i+patch, j*patch:j*patch+patch, :6]
                # normalize using pretrained mean
                image = image - mean
                label = val_label[patch*i:patch*i+patch, j*patch:j*patch+patch].astype('int32')
                seg_map = (val_data[patch*i:patch*i+patch,
                           j*patch:j*patch+patch, 6]*1/65535).astype(np.int32)
                print('w = {}-{}, h = {}-{}'.format(patch*i, patch*i+patch, j*patch, j*patch+patch))
                test_x = torch.FloatTensor(image.transpose(2,1,0)).unsqueeze(0).cuda()
                out_x, pred = net(test_x)
                pred = (pred.squeeze(0).cpu().numpy().transpose(1,0)+1) * seg_map
                accuracy = (pred == label).sum() * 100 /(256**2)
                print('accuracy = {:.5f}%'.format(accuracy))
                net_accuracy.append(accuracy)

        #get vegetation count
        unique, counts = np.unique(pred, return_counts=True)
        dict_c = dict(zip(unique, counts))
        #print(dict_c)
        if 0 in dict_c.keys():
            zero_count += dict_c[0]
        if 2 in dict_c.keys():
            veg_count += dict_c[2]
        if 13 in dict_c.keys():
            veg_count += dict_c[13]
        if 14 in dict_c.keys():
            veg_count += dict_c[14]
        print()

        mean_accuracy = np.asarray(net_accuracy).mean()
        print('total accuracy = {:.5f}%'.format(mean_accuracy))
        print('percentage vegetation = {:.5f}%'.format(veg_count*100.0/(val_data.shape[0]*val_data.shape[1]-zero_count)))

