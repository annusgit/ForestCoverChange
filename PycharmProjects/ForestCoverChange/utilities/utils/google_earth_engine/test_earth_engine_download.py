

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import sys


def main():
    bands = [4, 3, 2] # create rgb raster
    dataset = gdal.Open(sys.argv[1], gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(bands[0])
    image = band.ReadAsArray()
    # for x in range(2, dataset.RasterCount + 1):
    for x in bands[1:]:
        band = dataset.GetRasterBand(x)
        image = np.dstack((image, band.ReadAsArray()))
        # print(image.shape)
    # print(image.dtype)
    img = (image/4096*255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    pass


if __name__ == '__main__':
    main()