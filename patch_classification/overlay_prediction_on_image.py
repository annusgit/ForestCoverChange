

from __future__ import print_function
from __future__ import division
import cv2
import sys
import random
import numpy as np
import matplotlib as mt
import PIL.Image as Image
import matplotlib.pyplot as pl
import matplotlib.ticker as plticker


all_labels = {
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

all_labels_inverted = {v:k for k,v in all_labels.iteritems()}

def plot_separately():
    input_image = pl.imread(sys.argv[1])
    pred_image = pl.imread(sys.argv[2])
    pl.subplot(121)
    pl.imshow(input_image)
    pl.subplot(122)
    pl.imshow(pred_image)
    pl.show()
    pass


color_set = [0, 255, 70, 128]
possible_colors = [(x, y, z) for x in color_set for y in color_set for z in color_set]

def convert_to_colors(image_arr):
    # possible_colors = {x:y for x in range(len(possible_colors)) for y in }
    random.shuffle(possible_colors)
    # while possible_colors[0] != (0, 0, 0) and possible_colors[1] != (0, 255, 0):
    #     random.shuffle(possible_colors)
    # print(possible_colors)

    unique = np.unique(image_arr)
    new_image = np.zeros(shape=(image_arr.shape[0], image_arr.shape[1], 3))
    for idx, pix in enumerate(unique):
        new_image[image_arr == pix] = possible_colors[idx]
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


def overlay_with_grid():
    # Open image file
    x_start = 64*40
    y_start = 64*40
    x_end = x_start+64*10
    y_end = y_start+64*10
    image = cv2.imread('image_test_german.png')[y_start:y_end,x_start:x_end,:]
    cv2.imwrite('small_image_test_german.png', image)
    label = cv2.imread('pred_sentinel_german.png')[y_start:y_end,x_start:x_end,0]
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
    # fig.savefig('myImageGrid.tiff', dpi=my_dpi)
    # pl.axis('off')
    pl.show()


def check_predictions():
    # image = cv2.imread('/home/annus/Desktop/forest_images/image_test.png')[4000:8000,4000:8000,:]
    label = cv2.imread('pred_sentinel_german.png')[:,:,0]
    # label[label != 1] = 0
    # label = convert_to_colors(label)
    # print(np.unique(label), label.shape)
    # pl.subplot(121)
    # pl.imshow(image)
    # pl.subplot(122)
    # print(label.shape)
    pl.imshow(label)
    pl.show()
    pass


if __name__ == '__main__':
    overlay_with_grid()







