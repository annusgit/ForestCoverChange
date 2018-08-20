

"""
    Get the data in npy files in ~/Desktop/rit18_data and show as png files
"""

from __future__ import print_function
from __future__ import division

import os
import cv2
import numpy as np
from PIL import Image
from scipy.misc import imsave
import matplotlib.pyplot as plt


def resize(arr):
    return np.resize(arr, new_shape=(800, 800)).astype(np.uint16)

def rescale(arr):
    return (255 / 65536 * arr).astype(np.uint16)

def expand(arr):
    return np.expand_dims(arr, axis=2)


def process(np_arr, name):
    print('================================================')
    print('shape of array ==> ', np_arr.shape)
    c, w,h = np_arr.shape
    arr = np_arr.reshape((w, h, c))
    print('reshaped array ==> ', arr.shape)
    r, g, b = arr[:, :, 4], arr[:, :, 5], arr[:, :, 6]
    # r, g, b = map(rescale, [r,g,b])
    r, g, b = map(resize, [r,g,b])
    r, g, b = map(expand, [r,g,b])
    r, g, b = map(cv2.equalizeHist, [r, g, b])
    # print(map(np.shape, (r,g,b)))
    rgb = np.asarray(cv2.merge((r,g,b)))
    print('rgb shape ==> ', rgb.shape, rgb.dtype)
    cv2.imwrite(name, rgb)
    print('{} saved!'.format(name))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
    print(np.mean(np.mean(rgb)), np.mean(np.mean(rgb)))
    print('================================================')
    pass


def read_():
    # set new wd
    os.chdir('/home/annus/Desktop/rit18_data/')
    # read the data files
    # process(np.load('train_data.npy'))
    process(np.load('val_data.npy'), name='train_rgb.png')
    # process(np.load('test_val.npy'))
    pass


def main():
    read_()
    pass


if __name__ == '__main__':
    main()



