

"""
    Test a pretrained patch network on a real sentinel image
"""

from __future__ import print_function
from __future__ import division
import torch
import sys
import cv2
import gdal
import numpy as np
from model import VGG
from scipy import misc


if __name__ == '__main__':
    net = VGG(in_channels=3)
    test_model = sys.argv[1] # '/home/annus/Desktop/test.tif'
    image_path = sys.argv[2] # image_pred = np.zeros_like(image_read[:,:,0])
    net.load_state_dict(torch.load(test_model))
    net.cuda(device=0)

    bands = [3,2,1]
    patch = 64 # model input is of fixed size, 64, so...
    this = gdal.Open(image_path)
    image_read = this.GetRasterBand(bands[0]).ReadAsArray()
    for i in bands[1:]:
        image_read = np.dstack((image_read,
                                this.GetRasterBand(i).ReadAsArray())).astype(np.int16)
    image_read = (image_read.astype(np.float)/255).astype(np.float)
    # image_read = (image_read.astype(np.float) * 255 / 4095).astype(np.uint8)
    image_read = image_read[:512,:512,:]
    image_pred = np.zeros_like(image_read[:,:,0])
    print(image_read)


    def toTensor(image):
        "converts a single input image to tensor"
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float().unsqueeze(0)
    # image_read = toTensor(image_read)
    print(image_read.shape)

    if False:
        net.eval()
        for i in range(image_read.shape[0]//patch):
            for j in range(image_read.shape[1]//patch):
                image = image_read[patch*i:patch*i+patch, j*patch:j*patch+patch, :]
                # print(np.unique(image))
                print('log: w = {}-{}, h = {}-{}'.format(patch*i, patch*i+patch, j*patch, j*patch+patch), end=' ')
                # test_x = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).float().cuda()
                test_x = toTensor(image).cuda(device=0)
                # print(test_x.shape)
                out_x, pred = net(test_x)
                pred = pred.squeeze(0).cpu().numpy()
                pred_copy = pred.astype(np.uint8)
                image_pred[patch*i:patch*i+patch, j*patch:j*patch+patch] = pred_copy
                print(pred)

        cv2.imwrite('pred_sentinel.png', image_pred)






