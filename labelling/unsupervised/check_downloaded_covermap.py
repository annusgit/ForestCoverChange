

from __future__ import print_function
from __future__ import division
import os
import sys
import gdal
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as pl


def get_info(ds):
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    bands = ds.RasterCount
    projection = ds.GetProjection()
    return (rows, cols, bands, projection)


def convert_lat_lon_to_x_y(ds, coordinates):
    lon_in, lat_in = coordinates
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def main(path_to_covermap):
    covermap = gdal.Open(path_to_covermap, gdal.GA_ReadOnly)
    (rows, cols, bands, projection) = get_info(ds=covermap)
    print('(rows, cols, bands, projection) = ', rows, cols, bands, projection)
    channel = covermap.GetRasterBand(1)

    # testsite coordinates
    min_coords = (34.46484326132815, 73.30923379854437)
    max_coords = (34.13584821210507, 73.76516641573187)

    min_x, min_y = convert_lat_lon_to_x_y(ds=covermap, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_x_y(ds=covermap, coordinates=max_coords)
    print(min_x, min_y, max_x, max_y)

    #######3 magic!
    image = channel.ReadAsArray(min_x, min_y, abs(max_x-min_x), abs(max_y-min_y))
    print(np.unique(image))
    print(image.shape)
    # let's reshape it to match our actual image
    site_size = (3663, 5077)
    image = misc.imresize(image, size=site_size, interp='nearest')
    print(np.unique(image))
    print(image.shape)
    rgb_image = misc.imread('/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                            'full-test-site-pakistan/rgb/2015.png')[:,:,0:3]
    this_label = np.asarray(image, dtype=np.uint8)
    pl.subplot(1,2,1)
    pl.imshow(rgb_image)
    pl.subplot(1,2,2)
    pl.imshow(this_label)
    pl.show()
    pass


if __name__ == '__main__':
    main(path_to_covermap='/home/annus/Desktop/available cover map 2015/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif')





