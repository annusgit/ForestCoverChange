

from __future__ import print_function, division
import sys
import gdal
import png
import numpy as np
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
    for i in bands[1:]:
        example_array = np.dstack((example_array,
                                   example.GetRasterBand(i).ReadAsArray())).astype(np.int16)
    return example_array


def main(this_example):
    example = gdal.Open(this_example)
    example_array = get_combination(example=example, bands=[4,3,2])
    show_image = np.asarray(np.clip(example_array/4096, 0, 1)*255, dtype=np.uint8)
    print(show_image.shape)
    pl.imshow(show_image)
    pl.show()
    pass


if __name__ == '__main__':
    main(this_example=sys.argv[1])




