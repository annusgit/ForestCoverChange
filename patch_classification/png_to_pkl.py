

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import pickle as p
import gdal
import sys
import cv2


def tif_to_png(image_path):
    image = gdal.Open(image_path)
    print(image.RasterCount)
    r = image.GetRasterBand(3).ReadAsArray()
    g = image.GetRasterBand(2).ReadAsArray()
    b = image.GetRasterBand(1).ReadAsArray()
    rgb = np.dstack((r,g,b))
    pl.imshow((rgb*255/4096).astype(np.uint8))
    pl.show()
    return rgb


def png_to_pickle(image_file, pkl_file):
    image = tif_to_png(image_path=image_file)
    # image = cv2.imread(image_file, -1)[:,:,:3]
    save_name = pkl_file
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
    png_to_pickle(image_file=sys.argv[1], pkl_file=sys.argv[2])