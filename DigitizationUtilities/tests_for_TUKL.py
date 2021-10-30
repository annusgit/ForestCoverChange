import matplotlib.pyplot as plt
import numpy as np
import gdal


def min_max_stretch(this_image, null_mask):
    # https://theailearner.com/tag/min-max-stretching/
    min_max_stretched_bands = []
    for i in range(3):
        this_band = this_image[:,:,i]
        sorted_pixel_values = np.unique(np.sort(this_band.flatten()))
        min_pixel, max_pixel = sorted_pixel_values[1], sorted_pixel_values[-1]
        this_band_stretched = 255*(this_band-min_pixel)/(max_pixel-min_pixel)
        this_band_stretched[null_mask] = 0
        min_max_stretched_bands.append(this_band_stretched)
    return np.dstack(min_max_stretched_bands).astype(dtype=np.uint8)


def get_clipped_image(this_path):
    ds = gdal.Open(this_path)
    rgb = [ds.GetRasterBand(x).ReadAsArray() for x in [1, 2, 3]]  # 4, 3, 2 are the red, green, blue bands
    null_pixel_mask = (rgb[0] == 0)
    stacked_image = np.dstack(rgb)
    stretched_image = min_max_stretch(this_image=stacked_image, null_mask=null_pixel_mask)
    return (stacked_image/255).astype(np.float), stretched_image


original_image, stretched_image = get_clipped_image(this_path='2019_0-2.tif')
plt.imshow(original_image)
plt.show()