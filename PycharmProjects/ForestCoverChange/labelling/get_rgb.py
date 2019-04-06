

from __future__ import print_function, division
import os
import sys
import png
import cv2
import gdal
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as pl


def write_png(z, name):
    # Use pypng to write z as a color PNG.
    with open(name, 'wb') as f:
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16)
        # Convert z to the Python list of lists expected by
        # the png writer.
        z2list = z.reshape(-1, z.shape[1] * z.shape[2]).tolist()
        writer.write(f, z2list)
    pass


def get_combination(example, bands):
    example_array = example.GetRasterBand(bands[0]).ReadAsArray()
    # print(example_array.shape)
    for i in bands[1:]:
        example_array = np.dstack((example_array,
                                   example.GetRasterBand(i).ReadAsArray()))
    return example_array


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def main(this_example):
    palsar_mean = [8116.269912828, 3419.031791692, 40.270058337]
    palsar_std = [6136.70160067, 2201.432263753, 19.38761076]
    example = gdal.Open(this_example)
    # print(example.GetMetadata())
    # print(example.RasterCount)
    example_array = get_combination(example=example, bands=[1])
    example_array = np.nan_to_num(example_array)
    # print(example_array.mean(axis=(0,1)), example_array.std(axis=(0,1)))
    # example_array = (example_array-palsar_mean).astype(np.float)/palsar_std
    # print(example_array)
    # example_array = (255*example_array/example_array.max()).astype(np.uint8)
    # example_array = histogram_equalize(example_array)
    show_image = example_array
    # show_image = histogram_equalize(show_image)
    # show_image = np.asarray(255*show_image, dtype=np.uint8)
    # print(np.unique(show_image))
    # show_image[show_image < 1] = 4
    # show_image[show_image > 3] = 4
    # show_image = np.asarray(np.clip(example_array/4096, 0, 1)*255, dtype=np.uint8)
    uniq_labels, counts = np.unique(show_image, return_counts=True)
    forest_position = np.argmax(uniq_labels == 0)
    forest_pixels = float(counts[forest_position])
    total_pixels = float(counts.sum())
    forest_percentage = forest_pixels * 100 / total_pixels
    print('original forest: {:.3f}%'.format(forest_percentage))
    forest_cmap = colors.ListedColormap(['green', 'black'])
    bounds = [-0.5, 0.5, 1.5]  # between each two numbers is one corresponding color
    forest_norm = colors.BoundaryNorm(bounds, forest_cmap.N)
    print(show_image.shape)
    pl.imshow(show_image, norm=forest_norm, cmap=forest_cmap)
    pl.show()
    # if show_image is not None:
    #     pl.imsave(save_as, show_image) # save_as is the file_name you want to save with
    pass


def savesinglebands(this_example, dest_path):
    example = gdal.Open(this_example)
    bands = range(1,14)
    for b in bands:
        print('log: on band : {}'.format(b))
        np.save(os.path.join(dest_path, '{}.npy'.format(b)), example.GetRasterBand(b).ReadAsArray())
    pass


def check_single_bands(r, g, b):
    full_test_site_shape = (3663, 5077)
    red = np.load(r, mmap_mode='r')
    green = np.load(g, mmap_mode='r')
    blue = np.load(b, mmap_mode='r')
    red_band = red[1000:3000, 1000:3000]
    green_band = green[1000:3000, 1000:3000]
    blue_band = blue[1000:3000, 1000:3000]
    print(blue_band.shape)
    show_image = np.dstack((red_band,green_band,blue_band))
    show_image = np.asarray(np.clip(show_image/ 4096, 0, 1) * 255, dtype=np.uint8)
    pl.imshow(show_image)
    pl.show()


if __name__ == '__main__':
    main(this_example=sys.argv[1])#, save_as=sys.argv[2])
    # savesinglebands(this_example=sys.argv[1], dest_path=sys.argv[2])
    # check_single_bands('/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/full-test-site-pakistan/numpy_sums/2015/4.npy',
    #                    '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/full-test-site-pakistan/numpy_sums/2015/3.npy',
    #                    '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/full-test-site-pakistan/numpy_sums/2015/2.npy')







