

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as pl
import numpy as np
import sys
import os


def convert(this_one, dest, name):
    array = np.load(this_one)
    new_array = 1 - array
    green = 255*new_array
    red = blue = np.zeros_like(green)
    full_image = np.dstack((red, green, blue))
    save_path = os.path.join(dest, '{}.png'.format(name))
    pl.imsave(save_path, full_image)
    pass


if __name__ == '__main__':
    folder_path = sys.argv[1]
    files = [f for f in os.listdir(folder_path)]
    files.sort(key=lambda f: int(filter(str.isdigit, f)))
    print(files)
    count = 0
    for np_arr in files:
        convert(os.path.join(folder_path, np_arr), dest=folder_path, name=count)
        count += 1










