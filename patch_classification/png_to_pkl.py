

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import pickle as p
import gdal
import sys
import cv2


def tif_to_png(image_path, bands):
    # bands is the list of bands to load from the image rasters
    # bands = [4,3,2,5,8]
    # bands = [8,5,2,3,4]
    image = gdal.Open(image_path)
    all_bands = [] #
    for i in bands: #range(1, 1+image.RasterCount):
        all_bands.append(image.GetRasterBand(i).ReadAsArray())
    image = np.dstack(all_bands)
    # print(image.max())
    # show_image = (image[:,:,:3].astype(np.float32)*255/4096).astype(np.uint8)
    # show_image = np.dstack((show_image[:,:,2], show_image[:,:,1], show_image[:,:,0]))
    # print(show_image.shape, np.unique(show_image))
    # pl.imshow(show_image)
    # pl.show()
    return image


def png_to_pickle(image_file, pkl_file, bands):
    image = tif_to_png(image_path=image_file, bands=bands)
    save_name = pkl_file
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


