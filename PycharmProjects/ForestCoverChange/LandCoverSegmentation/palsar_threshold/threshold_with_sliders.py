

from __future__ import print_function
from __future__ import division
import os
import sys
import gdal
import argparse
import numpy as np
from scipy.ndimage import median_filter
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
    full_image_ = gdal.Open(args.image)
    HH = full_image_.GetRasterBand(1)
    HH = HH.ReadAsArray()
    HV = full_image_.GetRasterBand(2)
    HV = HV.ReadAsArray()

    full_label = gdal.Open(args.label)
    binary_map = np.nan_to_num(full_label.GetRasterBand(1).ReadAsArray()).astype(np.uint8)
    # reset the labels
    binary_map[binary_map == 0] = 1  # only once!
    binary_map[binary_map == 3] = 2
    binary_map -= 1

    # convert them to gamma_naught
    # HH_g_naught = np.nan_to_num(10*np.log10(HH**2+1e-7)-83.0)
    HV_g_naught = np.nan_to_num(10*np.log10(HV**2+1e-7)-83.0)
    HV_g_naught = median_filter(HV_g_naught, size=5)

    # glob_th_HH_g_naught = np.asarray(HH_g_naught>-7.5).astype(np.uint8)
    glob_th_HV_g_naught = np.asarray(HV_g_naught < args.threshold).astype(np.uint8)
    # glob_th_HV_g_naught = median_filter(glob_th_HV_g_naught, size=10)
    red = np.zeros_like(glob_th_HV_g_naught)
    green = 255*glob_th_HV_g_naught
    blue = np.zeros_like(glob_th_HV_g_naught)
    g_n_colored = np.dstack((red, green, blue))

    red = np.zeros_like(binary_map)
    green = 255 * binary_map
    blue = np.zeros_like(binary_map)
    binary_colored = np.dstack((red, green, blue))
    match = 100 * (glob_th_HV_g_naught == binary_map).sum() / binary_map.size

    fig, ax = pl.subplots()
    pl.suptitle('Percentage match: {:.3f}%'.format(match), fontsize=16)
    pl.subplot(121)
    pl.title('Original Map')
    pl.imshow(binary_colored)
    l = pl.subplot(122)
    pl.imshow(g_n_colored)
    pl.title('HV')

    axcolor = 'lightgoldenrodyellow'
    # ax_thresh = pl.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # ax_med_filt_size = pl.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    #
    # sthresh = Slider(ax_thresh, 'threshold', -25.0, 0.0, valinit=args.threshold, valstep=0.1)
    # sfilter = Slider(ax_med_filt_size, 'filter_size', 1, 10, valinit=3)
    #
    # def update(val):
    #     updated_thresh = sthresh.val
    #     updated_filter = sfilter.val
    #
    #     HV_g_naught = np.nan_to_num(10*np.log10(HV**2+1e-7)-83.0)
    #     HV_g_naught = median_filter(HV_g_naught, size=int(updated_filter))
    #     glob_th_HV_g_naught = np.asarray(HV_g_naught < updated_thresh).astype(np.uint8)
    #     # glob_th_HV_g_naught = median_filter(glob_th_HV_g_naught, size=10)
    #     red = np.zeros_like(glob_th_HV_g_naught)
    #     green = 255 * glob_th_HV_g_naught
    #     blue = np.zeros_like(glob_th_HV_g_naught)
    #     g_n_colored = np.dstack((red, green, blue))
    #     match = 100 * (glob_th_HV_g_naught == binary_map).sum() / binary_map.size
    #     pl.suptitle('Percentage match: {:.3f}%'.format(match), fontsize=16)
    #
    #     l.imshow(g_n_colored)
    #     fig.canvas.draw_idle()
    #
    # sthresh.on_changed(update)
    # sfilter.on_changed(update)

    # add radio buttons here
    # ['left_x', 'top_y', 'width', 'height'] in percentages of screen size
    current_threshold = args.threshold
    current_filter = 5
    threshold_rax = pl.axes([0.01, 0.35, 0.05, 0.4], facecolor=axcolor)
    thresholds_list = ('-12.1', '-12.3', '-11.9', '-14.3', '-13.9', '-13.5', '-12.2', '-15.1', '-14.8', '-20.1',
                       '-14.4', '-15.5', '-15.6', '-14.2', '-14.1')
    threshold_radio = RadioButtons(threshold_rax, thresholds_list, active=0)

    def threshold_radio_func(label):
        fig.canvas.draw_idle()
        updated_thresh = float(label)
        current_threshold = updated_thresh
        HV_g_naught = np.nan_to_num(10 * np.log10(HV ** 2 + 1e-7) - 83.0)
        HV_g_naught = median_filter(HV_g_naught, size=int(current_threshold))
        glob_th_HV_g_naught = np.asarray(HV_g_naught < updated_thresh).astype(np.uint8)
        # glob_th_HV_g_naught = median_filter(glob_th_HV_g_naught, size=10)
        red = np.zeros_like(glob_th_HV_g_naught)
        green = 255 * glob_th_HV_g_naught
        blue = np.zeros_like(glob_th_HV_g_naught)
        g_n_colored = np.dstack((red, green, blue))
        match = 100 * (glob_th_HV_g_naught == binary_map).sum() / binary_map.size
        pl.suptitle('Percentage match: {:.3f}%'.format(match), fontsize=16)

        l.imshow(g_n_colored)
        fig.canvas.draw_idle()

    threshold_radio.on_clicked(threshold_radio_func)

    # add radio buttons here
    # ['left_x', 'top_y', 'width', 'height'] in percentages of screen size
    filter_rax = pl.axes([0.95, 0.35, 0.04, 0.2], facecolor=axcolor)
    filters_list = [str(x) for x in range(1,11)]
    filter_radio = RadioButtons(filter_rax, filters_list, active=0)

    def filter_radio_func(label):
        fig.canvas.draw_idle()
        updated_filter = int(label)
        current_filter = updated_filter
        HV_g_naught = np.nan_to_num(10 * np.log10(HV ** 2 + 1e-7) - 83.0)
        HV_g_naught = median_filter(HV_g_naught, size=int(updated_filter))
        glob_th_HV_g_naught = np.asarray(HV_g_naught < current_threshold).astype(np.uint8)
        # glob_th_HV_g_naught = median_filter(glob_th_HV_g_naught, size=10)
        red = np.zeros_like(glob_th_HV_g_naught)
        green = 255 * glob_th_HV_g_naught
        blue = np.zeros_like(glob_th_HV_g_naught)
        g_n_colored = np.dstack((red, green, blue))
        match = 100 * (glob_th_HV_g_naught == binary_map).sum() / binary_map.size
        pl.suptitle('Percentage match: {:.3f}%'.format(match), fontsize=16)

        l.imshow(g_n_colored)
        fig.canvas.draw_idle()

    filter_radio.on_clicked(filter_radio_func)

    pl.show()
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image', type=str)
    parser.add_argument('--label', dest='label', type=str)
    parser.add_argument('--thresh', dest='threshold', type=float)
    args = parser.parse_args()
    convert_to_gamma(args)
    pass


if __name__ == '__main__':
    main()

