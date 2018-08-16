

from __future__ import print_function
from __future__ import division
import cv2
import sys
import random
import numpy as np
import matplotlib as mt
import PIL.Image as Image
import matplotlib.pyplot as pl


def plot_separately():
    input_image = pl.imread(sys.argv[1])
    pred_image = pl.imread(sys.argv[2])
    pl.subplot(121)
    pl.imshow(input_image)
    pl.subplot(122)
    pl.imshow(pred_image)
    pl.show()
    pass


all_labels = {
            'AnnualCrop'           : 0,
            'Forest'               : 1,
            'HerbaceousVegetation' : 2,
            'Highway'              : 3,
            'Industrial'           : 4,
            'Pasture'              : 5,
            'PermanentCrop'        : 6,
            'Residential'          : 7,
            'River'                : 8,
            'SeaLake'              : 9
            }

def convert_to_colors(image_arr):
    color_set = [0, 255, 128]
    possible_colors = [(x,y,z) for x in color_set for y in color_set for z in color_set]
    while possible_colors[0] != (0, 0, 0) and possible_colors[1] != (0, 255, 0):
        random.shuffle(possible_colors)
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


def check_predictions():

    image = cv2.imread('image_test.png')[4000:8000,4000:8000,:]
    label = cv2.imread('pred_sentinel.png')[4000:8000,4000:8000,0]
    label = convert_to_colors(label)
    print(np.unique(label), label.shape)
    pl.subplot(121)
    pl.imshow(image)
    pl.subplot(122)
    print(label.shape)
    pl.imshow(label)
    pl.show()
    pass


if __name__ == '__main__':
    overlayed_output()







