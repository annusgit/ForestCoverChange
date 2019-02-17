

from __future__ import print_function
from __future__ import division
import os
import sys
import gdal
import argparse
import numpy as np
from scipy.ndimage import median_filter, binary_dilation, binary_erosion, morphology
import matplotlib.pyplot as pl
from matplotlib.widgets import Slider, RadioButtons


def convert_to_gamma(args):
    '''
        Given an image filled with digital numbers (DN), convert it to gamma_naught using the following relation
        But first, we need to separate the three bands [HH, HV, angle]
        Polarization data are stored as 16-bit digital numbers (DN). The DN values can be converted to gamma naught values in decibel unit (dB) using the following equation:

                gamma_naught = 10*log_base_10(DN**2) - 83.0 dB

    :param image: a 3d image containing HH, HV and angle bands
    :return: two bands, gamma_naught for HH and gamma_naught for HV
    '''
    main_path = '/home/annus/Desktop/palsar/palsar_dataset_full/palsar_dataset/'
    image_path = os.path.join(main_path, 'palsar_{}_region_{}.tif'.format(args.year, args.region))
    label_path = os.path.join(main_path, 'fnf_{}_region_{}.tif'.format(args.year, args.region))

    full_label = gdal.Open(label_path)
    binary_map = np.nan_to_num(full_label.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    # reset the labels
    binary_map[binary_map == 0] = 1  # only once!
    binary_map[binary_map == 3] = 2
    binary_map -= 1

    full_image_ = gdal.Open(image_path)
    HV = full_image_.GetRasterBand(2)
    HV = HV.ReadAsArray()
    # convert them to gamma_naught
    HV_g_naught = np.nan_to_num(10*np.log10(HV**2+1e-7)-83.0)
    HV_g_naught = median_filter(HV_g_naught, size=5)
    glob_th_HV_g_naught = np.asarray(HV_g_naught < args.threshold).astype(np.uint8)
    match = 100 * (glob_th_HV_g_naught == binary_map).sum() / binary_map.size
    pl.figure()
    pl.suptitle('Percentage match: {:.3f}%'.format(match), fontsize=16)
    pl.subplot(121)
    pl.title('Original Map')
    pl.imshow(binary_map)
    pl.subplot(122)
    pl.imshow(glob_th_HV_g_naught)
    pl.title('HV')
    pl.show()
    pass


def distribution_function(args):
    main_path = '/home/annus/Desktop/palsar/palsar_dataset_full/palsar_dataset/'
    image_path = os.path.join(main_path, 'palsar_{}_region_{}.tif'.format(args.year, args.region))
    label_path = os.path.join(main_path, 'fnf_{}_region_{}.tif'.format(args.year, args.region))
    full_label = gdal.Open(label_path)
    binary_map = np.nan_to_num(full_label.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    full_image_ = gdal.Open(image_path)
    HV = full_image_.GetRasterBand(2).ReadAsArray()
    # convert them to gamma_naught
    HV_g_naught = np.nan_to_num(10 * np.log10(HV ** 2 + 1e-7) - 83.0)
    HV_g_naught = median_filter(HV_g_naught, size=5)
    forest = HV_g_naught[binary_map == 1]
    not_forest = HV_g_naught[binary_map == 2]
    print('forest pixels:', len(forest))
    print('not-forest pixels:', len(not_forest))
    # forest_hist = np.histogram(forest)[0]
    # not_forest_hist = np.histogram(not_forest)[0]
    pl.hist(not_forest, color='b', bins='auto', label='not forest')
    pl.hist(forest, color='r', bins='auto', label='forest')
    pl.legend(loc='upper right')
    pl.title('HV gamma_naught histogram')
    pl.ylabel('frequency')
    pl.xlabel('gamma_naught (dB)')
    pl.show()
    pass


def get_means(args):
    main_path = '/home/annus/Desktop/palsar/palsar_dataset_full/palsar_dataset/'
    means = []
    stds = []
    for year in range(2007, 2011):
        for region in range(1, 5):
            image_path = os.path.join(main_path, 'palsar_{}_region_{}.tif'.format(year, region))
            print('On {}'.format(image_path))
            full_image_ = gdal.Open(image_path)
            HH = full_image_.GetRasterBand(1).ReadAsArray()
            HV = full_image_.GetRasterBand(2).ReadAsArray()
            angle = full_image_.GetRasterBand(3).ReadAsArray()
            # convert them to gamma_naught
            HH_g_naught = np.nan_to_num(10 * np.log10(HH ** 2 + 1e-7) - 83.0)
            HV_g_naught = np.nan_to_num(10 * np.log10(HV ** 2 + 1e-7) - 83.0)
            angle = np.nan_to_num(angle)
            # get means
            means.append((np.mean(HH_g_naught), np.mean(HV_g_naught), np.mean(angle)))
            stds.append((np.std(HH_g_naught), np.std(HV_g_naught), np.std(angle)))
    mean_array = np.asarray(means)
    mean = mean_array.mean(axis=0)
    std_array = np.asarray(stds)
    std = std_array.mean(axis=0)
    print(mean, std)
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', dest='year', type=int)
    parser.add_argument('--region', dest='region', type=int)
    parser.add_argument('--thresh', dest='threshold', type=float)
    args = parser.parse_args()
    # convert_to_gamma(args)
    # distribution_function(args)
    get_means(args)
    pass


if __name__ == '__main__':
    main()









