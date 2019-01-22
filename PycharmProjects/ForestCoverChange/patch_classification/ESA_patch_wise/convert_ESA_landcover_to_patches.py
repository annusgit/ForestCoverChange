

from __future__ import print_function
from __future__ import division
import os
import gdal
import pickle
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage
import matplotlib.pyplot as pl


def convert_lat_lon_to_xy(ds, coordinates):
    lon_in, lat_in = coordinates
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def convert(image_path, bands, label_path, site_size, min_coords, max_coords, destination, stride):
    covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
    channel = covermap.GetRasterBand(1)
    min_x, min_y = convert_lat_lon_to_xy(ds=covermap, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_xy(ds=covermap, coordinates=max_coords)
    # read the corresponding label at 360m per pixel resolution
    label = channel.ReadAsArray(min_x, min_y, abs(max_x - min_x), abs(max_y - min_y))
    print(label.shape)
    # let's reshape it to match our actual image
    # label = misc.imresize(label, size=site_size, interp='nearest')
    # label = ndimage.median_filter(label, size=7)

    # let's get the actual image now
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    x_size, y_size = site_size #image_ds.RasterXSize, image_ds.RasterYSize
    all_raster_bands = [image_ds.GetRasterBand(x) for x in bands]

    count = -1
    # stride = 3000  # for testing only
    error_pixels = 50  # add this to remove the error pixels at the boundary of the test image
    for i in range(y_size//stride):
        for j in range(x_size//stride):
            count += 1
            # read the raster band by band for this subset
            # example_subset = np.nan_to_num(all_raster_bands[0].ReadAsArray(j*stride+error_pixels,
            #                                                        i*stride+error_pixels,
            #                                                        stride, stride))
            # for band in all_raster_bands[1:]:
            #     example_subset = np.dstack((example_subset , np.nan_to_num(band.ReadAsArray(j*stride+error_pixels,
            #                                                                i*stride+error_pixels,
            #                                                                stride,
            #                                                                stride))))
            # show_image = np.asarray(255*(example_subset [:,:,[4,3,2]]/4096.0).clip(0,1), dtype=np.uint8)
            label_subset = label[i*stride+error_pixels:(i+1)*stride+error_pixels,
                                j*stride+error_pixels:(j+1)*stride+error_pixels]
            these_labels, their_frequency = np.unique(label_subset, return_counts=True)
            if len(these_labels) > 0:
                max_frequent_label = these_labels[np.argmax(their_frequency)]
                # label[i * stride + error_pixels:(i + 1) * stride + error_pixels,
                # j * stride + error_pixels:(j + 1) * stride + error_pixels] = max_frequent_label
            pass
    these_labels, their_frequency = np.unique(label, return_counts=True)
    for i in range(len(these_labels)):
        print(these_labels[i], int(their_frequency[i]/64**2))
    pl.imshow(label)
    pl.show()
    pass


def main():
    convert(image_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                       'ESA_landcover_dataset/raw/full_test_site_2015.tif',
            bands=range(1,14),
            label_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                       'ESA_landcover_dataset/raw/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif',
            # site_size=(3663, 5077),
            # min_coords=(34.46484326132815, 73.30923379854437),
            # max_coords=(34.13584821210507, 73.76516641573187),
            site_size=(7000, 7000),
            # training area 1
            # min_coords=(34.57553970003938, 72.7322769139239),
            # max_coords=(34.044958204383086, 73.42166900376765),
            # training area 2
            # min_coords=(35.01253061124624, 71.6466289969444),
            # max_coords=(34.44343646423781, 72.36123193836568),
            # training area 3
            # min_coords=(34.34351204840675, 71.00509512834856),
            # max_coords=(33.83347503600184, 71.6951904394084),
            # reduced region 1
            min_coords=(35.197641666672425, 71.71160207097978),
            max_coords=(33.850742431378535, 73.45671012555636),
            destination='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                        'ESA_landcover_dataset/divided',
            stride=64)
    pass

# 71.72240936837943, 35.197641666672425],
#           [71.71160207097978, 33.85074243704372],
#           [73.45671012555636, 33.850742431378535],
#           [73.44590446052473, 35.20666927156783

if __name__ == '__main__':
    main()

























