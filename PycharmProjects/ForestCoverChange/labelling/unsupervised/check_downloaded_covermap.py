

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


def convert_lat_lon_to_x_y(ds, coordinates, max_rows, max_cols):
    lon_in, lat_in = coordinates
    lon_min, lon_max = -180, 180 # these are coming from the actual data's geotransform information
    lat_min, lat_max = 90, -90
    lon_half, lat_half = 180, 90
    lon_span = 360 # because we only want to deal with positive numbers
    lat_span = 180 # because we only want to deal with positive numbers
    # longitudes correspond to x-coordinates
    # latitudes correspond to y-coordinates
    # basically the mapping is as following
    # (lat_min-lat_max) --> (0-max_rows)
    # (lon_min-lon_max) --> (0-max_cols)
    # x = int(max_cols * (lon_in + lon_half) / lon_span)
    # y = int(max_rows * (lat_in + lat_half) / lat_span)
    # x = int(64799*(lat_in/360 + 1/2))
    # y = int(129599*(-lon_in/180 + 1/2))
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    # print(ds.GetGeoTransform())
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def main(path_to_covermap):
    covermap = gdal.Open(path_to_covermap, gdal.GA_ReadOnly)
    (rows, cols, bands, projection) = get_info(ds=covermap)
    print('(rows, cols, bands, projection) = ', rows, cols, bands, projection)
    channel = covermap.GetRasterBand(1)
    layer = covermap.GetLayer()
    # print(covermap.GetGeoTransform())
    # print('GetGeometryRef' in dir(covermap))
    # print('GetGeometryRef' in dir(channel))
    # print('GetGeometryRef' in dir(layer))
    # print(covermap.GetMetadata())
    # return
    blocksize = channel.GetBlockSize() # gives information of the available block size in the image

    # testsite coordinates
    min_coords = (34.46484326132815, 73.30923379854437)
    max_coords = (34.13584821210507, 73.76516641573187)

    # px = int((mx - gt[0]) / gt[1])  # x pixel
    # py = int((my - gt[3]) / gt[5])  # y pixel

    min_x, min_y = convert_lat_lon_to_x_y(ds=covermap, coordinates=min_coords, max_rows=rows, max_cols=cols)
    max_x, max_y = convert_lat_lon_to_x_y(ds=covermap, coordinates=max_coords, max_rows=rows, max_cols=cols)
    print(min_x, min_y, max_x, max_y)

    #######3 magic!
    # min_x, min_y = int(2.1/3*cols), 10000
    dimension = 1000
    addition = 500
    # india1 = [73.33658522109943, 30.818134308172535]
    # india2 = [89.33267897109943, 5.693429028978402]

    # image = channel.ReadAsArray(min_x, min_y, dimension, dimension)
    # image = channel.ReadAsArray(min_x-addition, min_y-addition, abs(max_x-min_x)+addition, abs(max_y-min_y)+addition)
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
    # this_label = np.dstack((this_label,this_label,this_label))
    # rgb_image = rgb_image * this_label #100*(image==190)
    pl.imshow(this_label)
    pl.show()
    pl.imshow(image)
    pl.show()

    pass


if __name__ == '__main__':
    main(path_to_covermap='/home/annus/Desktop/available cover map 2015/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif')









