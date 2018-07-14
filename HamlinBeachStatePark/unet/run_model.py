

"""
    Use Unet model (pretrained in Matlab) and evaluate it
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
    net = UNet(model_dir_path=os.path.join(dir_path, 'Unet_pretrained_model.pkl'), input_channels=channels).cuda()
    val_data = np.load(os.path.join(dir_path, 'rit18_data/val_data.npy'), mmap_mode='r')
    val_label = np.load(os.path.join(dir_path, 'rit18_data/val_labels.npy'), mmap_mode='r')
    val_data = val_data.transpose((1, 2, 0))
    print('before padding: val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
    # pad the data to be divisible by patch size
    val_data = np.pad(val_data, ((0, patch - val_data.shape[0] % patch),
                                 (0, patch - val_data.shape[1] % patch),
                                 (0, 0)), 'constant')
    val_label = np.pad(val_label, ((0, patch - val_label.shape[0] % patch),
                                   (0, patch - val_label.shape[1] % patch)), 'constant')
    # divide it into patches
    print('after padding: val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
    val_data = np.reshape(val_data, newshape=(-1, 256, 256, 7))
    seg_map_full = (1/65535*val_data[:,:,:,6]).astype(np.int32); val_data = val_data[:,:,:,:6]
    val_label = np.reshape(val_label, newshape=(-1, 256, 256))
    print('after reshaping: val_data shape = {}, val_label shape = {}'.format(val_data.shape, val_label.shape))
    net_accuracy = []
    # take everything onto the gpu
    mean = torch.FloatTensor(np.load(os.path.join(dir_path, 'mean.npy')).astype(np.float32)).cuda()
    zero_count = 0
    veg_count = 0
    val_data_gpu = torch.FloatTensor(val_data.astype(np.float32)).cuda()
    # val_label = torch.Tensor(val_label).long()
    print('final array on gpu = ', val_data_gpu.size())
    if True:
        num_batches = val_data.shape[0]//batch_size
        for i in range(num_batches): ## val_data.shape[0] is the number of available images
                image = val_data_gpu[batch_size*i:batch_size*i+batch_size, :, :, :] # get a batch of "batch_size" images
                label = val_label[batch_size*i:batch_size*i+batch_size, :, :]
                seg_map = seg_map_full[batch_size*i:batch_size*i+batch_size, :, :]
                # normalize using pretrained mean
                image = image - mean
                # reshape as (batches, channels, w, h)
                test_x = image.permute(0,3,2,1)
                out_x, pred = net(test_x)
                pred = (pred.cpu().numpy().transpose(0,2,1)+1) * seg_map
                #get vegetation count
                unique, counts = np.unique(pred, return_counts=True)
                dict_c = dict(zip(unique, counts))
                # so [2, 13, 14] labels indicate vegetation, find which ones occurred here
                veg_indices = set([2,13,14]).intersection(set(dict_c.keys()))
                for x in veg_indices:
                    veg_count += dict_c[x]
                zero_count += dict_c[0] if 0 in dict_c.keys() else 0
                # get accuracy metric
                accuracy = (pred == label).sum() * 100 / (batch_size*256**2)
                print('batch {}/{}: accuracy = {:.5f}%'.format(i+1, num_batches, accuracy))
                net_accuracy.append(accuracy)
        mean_accuracy = np.asarray(net_accuracy).mean()
        print('total accuracy = {:.5f}%'.format(mean_accuracy))
        print('percentage vegetation = {:.5f}%'.format(veg_count*100.0/(val_data.shape[0]*val_data.shape[1]-zero_count)))





