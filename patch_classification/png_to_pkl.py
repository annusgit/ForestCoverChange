

from __future__ import print_function
from __future__ import division
import numpy as np
import pickle as p
import sys
import cv2


def main():
    image_file = sys.argv[1]
    save_name = sys.argv[2]
    image = cv2.imread(image_file, -1)[:,:,:3]
    print(image.max())
    save_image = {
        'pixels': image,
        'size': image.size,
        'mode': None,
    }
    with open(save_name, 'wb') as this_file:
        p.dump(save_image, this_file, protocol=p.HIGHEST_PROTOCOL)
    pass


if __name__ == '__main__':
    main()