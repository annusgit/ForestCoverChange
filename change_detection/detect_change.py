

from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as pl
from scipy.ndimage import median_filter


def check_difference_map(args):
    first_image_path = os.path.join(args.parent_dir_path, args.region, 'generated_map_{}_{}.npy'.format(args.first_year,
                                                                                                        args.region))
    second_image_path = os.path.join(args.parent_dir_path, args.region, 'generated_map_{}_{}.npy'.format(args.second_year,
                                                                                                         args.region))
    with open(first_image_path, 'rb') as first_generated:
        first_year_generated_label = np.load(first_generated).astype(np.uint8)
        first_year_generated_label = median_filter(first_year_generated_label, size=3)
    with open(second_image_path, 'rb') as second_generated:
        second_year_generated_label = np.load(second_generated).astype(np.uint8)
        second_year_generated_label = median_filter(second_year_generated_label, size=3)
    uniq_labels, counts = np.unique(first_year_generated_label, return_counts=True)
    forest_position = np.argmax(uniq_labels == 0)
    num_forest_pixels = float(counts[forest_position])
    num_total_pixels = float(counts.sum())
    first_forest_percentage = num_forest_pixels * 100 / num_total_pixels
    # print('-> First year forest: {:.3f}%'.format(first_forest_percentage))
    print(first_forest_percentage)
    uniq_labels, counts = np.unique(second_year_generated_label, return_counts=True)
    forest_position = np.argmax(uniq_labels == 0)  # forest is labeled as class 0
    num_forest_pixels = float(counts[forest_position])
    num_total_pixels = float(counts.sum())
    second_forest_percentage = num_forest_pixels * 100 / num_total_pixels
    # print('-> Second year forest: {:.3f}%'.format(second_forest_percentage))
    print(second_forest_percentage)
    match = 100 * (first_year_generated_label == second_year_generated_label).sum() / num_total_pixels
    # print('-> Percentage match: {:.3f}%'.format(match))
    print(match)
    # get the difference image
    diff_image = np.asarray(second_year_generated_label.astype(np.float)-first_year_generated_label.astype(np.float))
    # get the change statistics, forest gain, forest loss
    uniq_labels, counts = np.unique(diff_image, return_counts=True)
    gain_position = np.argmax(uniq_labels == -1)  # because -1 indicates gain in forest in our case
    loss_position = np.argmax(uniq_labels == +1)  # because +1 indicates loss in forest in our case
    num_gain_pixels = float(counts[gain_position])
    num_loss_pixels = float(counts[loss_position])
    num_total_pixels = float(counts.sum())
    gain_percentage = num_gain_pixels * 100 / num_total_pixels
    loss_percentage = num_loss_pixels * 100 / num_total_pixels
    # change_percentage = (num_gain_pixels-num_loss_pixels) * 100 / num_total_pixels
    change_percentage = (second_forest_percentage-first_forest_percentage) * 100 / first_forest_percentage
    ####################
    # print('-> Relative Percentage Forest Gain: {:.3f}%'.format(gain_percentage))
    # print('-> Relative Percentage Forest Loss: {:.3f}%'.format(loss_percentage))
    # print('-> Relative Percentage Effective Change: {:.3f}%'.format(change_percentage))
    ####################
    print(gain_percentage)
    print(loss_percentage)
    print(change_percentage)
    # print('-> Forest gain: {:.3f}%'.format(forest_gain))
    # print('-> Forest loss: {:.3f}%'.format(forest_loss))
    # make a color map of fixed colors for difference image
    # pixel mapping in the actual images 0 -> forest (green), 1 -> non-forest (black)
    # pixel mapping in difference image (second_year - first_year)
    # 0-0 = 1-1 = 0 -> persistent land (black) no change
    # 0-1 = -1 -> (non-forest->forest) (green) increase
    # 1-0 = +1 -> (forest->non-forest) (red) decrease
    forest_cmap = colors.ListedColormap(['green', 'black'])
    bounds = [-0.5, 0.5, 1.5]  # between each two numbers is one corresponding color
    forest_norm = colors.BoundaryNorm(bounds, forest_cmap.N)
    pl.subplot(131)
    pl.title('{}_region_{}'.format(args.first_year, args.region))
    pl.imshow(first_year_generated_label, cmap=forest_cmap, norm=forest_norm)
    pl.axis('off')
    pl.subplot(132)
    pl.title('{}_region_{}'.format(args.second_year, args.region))
    pl.imshow(second_year_generated_label, cmap=forest_cmap, norm=forest_norm)
    pl.axis('off')
    diff_cmap = colors.ListedColormap(['blue', 'black', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5]  # between each two numbers is one corresponding color
    diff_norm = colors.BoundaryNorm(bounds, diff_cmap.N)
    pl.subplot(133)
    pl.title('Change Image')
    pl.imshow(diff_image, cmap=diff_cmap, norm=diff_norm)
    pl.axis('off')
    pl.show()
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', dest='parent_dir_path', type=str, default='/home/annus/Desktop/'
                                                                              'final_defense_generated_maps/')
    parser.add_argument('-f_y', dest='first_year', type=int)
    parser.add_argument('-s_y', dest='second_year', type=int)
    parser.add_argument('-r', dest='region', type=str)
    args = parser.parse_args()
    check_difference_map(args)
    pass


if __name__ == '__main__':
    main()



