

"""
    Test anamoly removal using average of images
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
import os


def main():
    images = [os.path.join('/home/annus/Desktop', x) for x in os.listdir('/home/annus/Desktop') if x.endswith('.jpg')]
    # print(len(images))
    read_images = []
    for x in images:
        read_images.append(np.asarray(cv2.imread(x)))
    img = np.asarray(read_images)
    img = np.mean(img, axis=0, dtype=np.int32)
    print('used {} images'.format(len(read_images)))
    cv2.imwrite('/home/annus/Desktop/resultant.png', img)
    pass


if __name__ == '__main__':
    main()
