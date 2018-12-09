

from __future__ import print_function
from __future__ import division
import os
import cv2
import sys
import random
import numpy as np
import matplotlib as mt
import PIL.Image as Image
import matplotlib.pyplot as pl
import matplotlib.ticker as plticker
from skimage.measure import block_reduce
import scipy.ndimage as nd


German_labels = {
            'Annual\nCrop'           : 0,
            'Forest'                 : 1,
            'Herbaceous\nVegetation' : 2,
            'Highway'                : 3,
            'Industrial'             : 4,
            'Pasture'                : 5,
            'Permanent\nCrop'        : 6,
            'Residential'            : 7,
            'River'                  : 8,
            'SeaLake'                : 9
            }

Pakistani_labels = {
                    'forest': 0,
                    'residential': 1,
                    'pastures': 2,
                    'vegetation': 3,
                    'plainland': 4,
                    'snow': 5
                   }


def plot_separately():
    input_image = pl.imread(sys.argv[1])
    pred_image = pl.imread(sys.argv[2])
    pl.subplot(121)
    pl.imshow(input_image)
    pl.subplot(122)
    pl.imshow(pred_image)
    pl.show()
    pass


# color_set = [0, 255, 70, 128]
# possible_colors = [(x, y, z) for x in color_set for y in color_set for z in color_set]
possible_colors = [(127,255,212),
                   (0,255,0),
                   (240,230,140),
                   (211,211,211),
                   (139,136,120),
                   (0,0,128),
                   (127,255,0),
                   (0,0,0),
                   (0,191,255),
                   (135,206,250)]
color_bank = {l:c for l,c in zip(range(10), possible_colors)}

def convert_to_colors(image_arr, label, flag=None):
    # just forests?
    new_image = np.zeros(shape=(image_arr.shape[0], image_arr.shape[1], 3))
    if flag == 'forest':
        new_image[image_arr == label] = color_bank[1]
        return new_image.astype(np.uint8)
    # or all of the labels?
    unique = np.unique(image_arr)
    for idx, pix in enumerate(unique):
        new_image[image_arr == pix] = color_bank[idx]
    return new_image.astype(np.uint8)


def direct_overlay():
    background = Image.open(sys.argv[2])
    foreground = Image.open(sys.argv[1])
    background.paste(foreground, (0, 0), foreground)
    background.show()


def overlayed_output():
    image = cv2.imread('image_test.png')[4000:8000, 4000:8000, :]
    label = cv2.imread('pred_sentinel.png')[4000:8000, 4000:8000, 0]
    label = convert_to_colors(label)
    background = Image.fromarray(image)
    overlay = Image.fromarray(label)
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    generated = Image.blend(background, overlay, 0.4)
    background.save('image.png', 'PNG')
    overlay.save('label.png', 'PNG')
    generated.save('gen.png', 'PNG')
    generated = np.asarray(generated)
    # new_img.save("this.png", "PNG")
    pl.imshow(generated)
    pl.show()
    pass


def overlay_with_grid(image_path, pred_path, data_type, image_save_path, downsampled_image_save_path,
                      label_save_path, shape, show=False, type='germany'):
    # use one of the following based on the size of the image; if image is huge, go with the first one!
    ##########################################################################
    (H, W, C) = shape
    full_image = np.memmap(image_path, dtype=np.uint16, mode='r', shape=(H, W, C))#.transpose(1,0,2)
    full_label = np.memmap(pred_path, dtype=np.uint8, mode='r', shape=(H, W))#.transpose(1,0)
    x_start = 64 * 3
    y_start = 64 * 3
    x_end = x_start + 64 * 10
    y_end = y_start + 64 * 10
    image = full_image.copy()[y_start:y_end,x_start:x_end,:]
    if type == 'pakistan':
        all_labels = Pakistani_labels
    elif type == 'germany':
        all_labels = German_labels
    all_labels_inverted = {v: k for k, v in all_labels.iteritems()}
    # ex_array = []
    # for t in range(4, -1, -1):
    #     temp = np.expand_dims(image[:, :, t], 2)
    #     ex_array.append(temp)
    # image = np.dstack(ex_array)
    # do this for more than 3 channels
    show_image = image[:, :, :3]
    image = np.dstack((show_image[:, :, 2], show_image[:, :, 1], show_image[:, :, 0]))
    #################################
    label = full_label.copy()[y_start:y_end,x_start:x_end]
    # image = np.load(image_path, mmap_mode='r')
    # label = np.load(pred_path, mmap_mode='r')
    # print(image)
    ###########################################################################
    # x_start = 64 * 140
    # y_start = 64 * 10
    # x_end = x_start + 64 * 10
    # y_end = y_start + 64 * 10
    # image = image[y_start:y_end,x_start:x_end,:]
    # label = label[y_start:y_end,x_start:x_end]
    ###########################################################################
    # colored_label = convert_to_colors(label)
    my_dpi = 300

    # Set up figure
    fig = pl.figure(figsize=(float(image.shape[0])/my_dpi, float(image.shape[1])/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Set the gridding interval: here we use the major tick interval
    myInterval = 64.
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-', color='g')

    # Add the image
    ax.imshow(image)

    # Find number of gridsquares in x and y direction
    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(myInterval)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(myInterval)))

    # Add some labels to the gridsquares
    mt.rcParams.update({'font.size': 2})
    for j in range(ny):
        y = myInterval / 2 + j * myInterval
        for i in range(nx):
            x = myInterval / 2. + float(i) * myInterval
            # ax.text(x, y, '{:d}'.format(i + j * nx), color='w', ha='center', va='center').set_color('red')
            # find the label at this point
            this_label = label[int(y),int(x)]
            ax.text(x, y, '{}'.format(all_labels_inverted[this_label]),
                    color='w', ha='center', va='center').set_color('yellow')

    # Save the figure
    fig.savefig(image_save_path, dpi=my_dpi)

    forest_label = 0 #if data_type == 'pakistan' else 1
    # print('data type = ', data_type)
    ############ we will save downsampled images to show how it relates to pixel-wise results ##########
    # save colored label as well
    # 1. reduce 64*64 blocks, 2. apply filter to remove segmentation noise, 3. convert labels to colors
    colored_labels = block_reduce(full_label, block_size=(64,64), func=np.max)
    # colored_labels = cv2.medianBlur(colored_labels, ksize=3)
    # print(forest_label)
    # print(colored_labels)
    filtered_forest = colored_labels[colored_labels == forest_label].sum()/colored_labels.reshape(-1).shape[0]*100
    print('forest count = ', colored_labels[colored_labels == 0].sum())
    colored_labels = convert_to_colors(colored_labels, forest_label, flag='forest')
    # print(set( tuple(v) for m2d in colored_labels for v in m2d )) # if you want to check unique colors
    pl.imsave(label_save_path, colored_labels)
    # downsample and save input image
    downsampled_input_image = block_reduce(full_image, block_size=(64,64,1), func=np.max)
    # downsampled_image_path = os.path.join(os.path.splitext(image_save_path)[0]+'_down.png')
    pl.imsave(downsampled_image_save_path, downsampled_input_image)
    if show:
        pl.show()
    return filtered_forest


def check_predictions():
    # image = cv2.imread('/home/annus/Desktop/forest_images/image_test.png')[4000:8000,4000:8000,:]
    pred_path = sys.argv[1] #'../numerical_results/german_sentinel_ee/test_images_and_predictions/image_pred_6.npy'
    label = np.memmap(pred_path, dtype=np.uint8, mode='r', shape=(2048, 3840))#.transpose(1,0)
    # pl.imshow(label)
    # pl.show()
    label = convert_to_colors(label)
    from skimage.measure import block_reduce
    # label[label != 1] = 0
    # print(np.unique(label), label.shape)
    # pl.subplot(121)
    # pl.imshow(image)
    # pl.subplot(122)
    # print(label.shape)
    # label = label[:2048,:2048,:]
    print(label.shape)
    label = block_reduce(label, block_size=(64,64,1), func=np.max)
    print(label.shape)
    pl.imshow(label)
    pl.axis('off')
    pl.show()
    pass


if __name__ == '__main__':
    overlay_with_grid(image_path='test_image_tmp.npy',
                      pred_path='image_pred_tmp.npy',
                      image_save_path='testing',
                      label_save_path='testing2',
                      shape=(2048, 3840, 3))
    # check_predictions()






