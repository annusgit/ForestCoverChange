

from __future__ import print_function
from __future__ import division
import os
import cv2
import argparse
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as pl
from scipy.ndimage import median_filter
from tkinter import *
import tkMessageBox

total_buttons = 0
bt_regions = ['Battagram', 'Abbottabad', 'Haripur', 'Kohistan', 'Tor Ghar', 'Mansehra', 'Buner', 'Chitral',
              'Lower Dir', 'Malakand', 'Shangla', 'Swat', 'Upper Dir', 'Hangu', 'Karak', 'Kohat', 'Nowshehra']
real_names = ['battagram', 'abbottabad', 'haripur_region', 'kohistan', 'tor_ghar', 'mansehra', 'buner', 'chitral',
              'lower_dir', 'malakand', 'shangla', 'swat', 'upper_dir', 'hangu', 'karak', 'kohat', 'nowshehra']
bt_names_to_real_names = dict(zip(bt_regions, real_names))
first_row_btn = [(0.05, 0.2), (0.20, 0.2), (0.35, 0.20), (0.50, 0.2), (0.65, 0.2), (0.80, 0.2)]
second_row_btn = [(0.05, 0.28), (0.20, 0.28), (0.35, 0.28), (0.50, 0.28), (0.65, 0.28), (0.80, 0.28)]
third_row_btn = [(0.10, 0.36), (0.25, 0.36), (0.40, 0.36), (0.55, 0.36), (0.70, 0.36)]
button_locs = first_row_btn + second_row_btn + third_row_btn
parent_dir_path = '/home/annus/Desktop/final_defense_generated_maps/'
graphs_dir = '/home/annus/Desktop/final_defense_graphs/'
gif_dir = '/home/annus/Desktop/final_defense_change_images/'
first_year, second_year = 2014, 2018


def check_difference_map(region_name):
    first_image_path = os.path.join(parent_dir_path, region_name, 'generated_map_{}_{}.npy'.format(first_year, region_name))
    second_image_path = os.path.join(parent_dir_path, region_name, 'generated_map_{}_{}.npy'.format(second_year, region_name))
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
    print('-> First year forest: {:.3f}%'.format(first_forest_percentage))
    print(first_forest_percentage)
    uniq_labels, counts = np.unique(second_year_generated_label, return_counts=True)
    forest_position = np.argmax(uniq_labels == 0)  # forest is labeled as class 0
    num_forest_pixels = float(counts[forest_position])
    num_total_pixels = float(counts.sum())
    second_forest_percentage = num_forest_pixels * 100 / num_total_pixels
    print('-> Second year forest: {:.3f}%'.format(second_forest_percentage))
    print(second_forest_percentage)
    match = 100 * (first_year_generated_label == second_year_generated_label).sum() / num_total_pixels
    print('-> Percentage match: {:.3f}%'.format(match))
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
    print('-> Relative Percentage Forest Gain: {:.3f}%'.format(gain_percentage))
    print('-> Relative Percentage Forest Loss: {:.3f}%'.format(loss_percentage))
    print('-> Relative Percentage Effective Change: {:.3f}%'.format(change_percentage))
    forest_cmap = colors.ListedColormap(['green', 'black'])
    bounds = [-0.5, 0.5, 1.5]  # between each two numbers is one corresponding color
    forest_norm = colors.BoundaryNorm(bounds, forest_cmap.N)

    pl.figure()
    files = [os.path.join(gif_dir, region_name, x) for x in os.listdir(os.path.join(gif_dir, region_name))]
    count = 0
    years = [2013, 2014, 2016, 2017, 2018]
    for f in files:
        im = pl.imread(f)
        # print(im)
        # im = cv2.resize(im, (800,800,3))
        pl.imshow(im)
        pl.title('{}'.format(years[count]))
        pl.pause(.1)
        count += 1
        pl.draw()

    pl.figure()
    pl.subplot(131)
    pl.title('{}, {}'.format(first_year, region_name))
    pl.imshow(first_year_generated_label, cmap=forest_cmap, norm=forest_norm)
    pl.axis('off')
    pl.subplot(132)
    pl.title('{}, {}'.format(second_year, region_name))
    pl.imshow(second_year_generated_label, cmap=forest_cmap, norm=forest_norm)
    pl.axis('off')
    diff_cmap = colors.ListedColormap(['blue', 'black', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5]  # between each two numbers is one corresponding color
    diff_norm = colors.BoundaryNorm(bounds, diff_cmap.N)
    pl.subplot(133)
    pl.title('Change 14-18')
    pl.imshow(diff_image, cmap=diff_cmap, norm=diff_norm)
    pl.axis('off')

    # now get its corresponding graph
    pl.figure()
    full_stats = pl.imread(os.path.join(graphs_dir, '{}.png'.format(region_name)))
    pl.imshow(full_stats)
    pl.axis('off')

    pl.show()
    pass

def main():
    window = Tk()
    window.geometry('750x420')
    window.title("Welcome to LikeGeeks app")
    main_title = Label(window, text=" Billion Tree Tsunami Forest Cover Change Detection", font=('Calibri', 20))
    sub_title = Label(window, text="   Select a Region for Running Analysis", font=('Calibri', 15))
    main_title.pack()
    sub_title.pack()

    def callback(*args):
        tkMessageBox.showinfo("Billion Tree Analytics", "Getting Analytics for {} Now...".format(args[0]))
        check_difference_map(region_name=args[0])
        pass

    def get_button(label, colors, command, count=0):
        global total_buttons
        btn = Button(window, text=label, bg=colors[0], fg=colors[1], height=1, width=10,
                     command=lambda: command(bt_names_to_real_names[label]))
        btn.pack()
        btn.place(relx=button_locs[count][0], rely=button_locs[count][1])
        total_buttons = total_buttons + 1
        return btn

    all_buttons = []
    btn_color = ('white', 'blue')
    for region in bt_regions:
        all_buttons.append(get_button(label=region, colors=btn_color, command=callback, count=len(all_buttons)))
    window.mainloop()
    pass


if __name__ == '__main__':
    main()