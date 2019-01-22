

from __future__ import print_function, division
import os
import sys
import png
import cv2
import gdal
import numpy as np
import PIL.Image as Image
import scipy.misc as misc
import matplotlib.pyplot as pl


all_coordinates = {
    'reduced_region_1': [[35.197641666672425, 71.72240936837943],
                         [33.85074243704372, 71.71160207097978],
                         [33.850742431378535, 73.45671012555636],
                         [35.20666927156783, 73.44590446052473]],
    'reduced_region_2': [[33.85103422591486, 71.71322076522074],
                         [32.48625596915296, 71.68572000739414],
                         [32.490880463032994, 73.48098348625706],
                         [33.84190120007257, 73.46449281249886]],
    'reduced_region_3': [[33.814697704520704, 69.92654032072937],
                         [32.480979689325956, 69.89907450041687],
                         [32.47171147795466, 71.64590067229187],
                         [33.80556931756046, 71.64590067229187]],
    'reduced_region_4': [[33.829815338339664, 73.52583857548143],
                         [32.491695768256115, 73.51485224735643],
                         [32.505594640301354, 75.05293818485643],
                         [33.825252073333274, 75.03096552860643]],
    'reduced_region_5': [[32.411916100234734, 69.54339120061195],
                         [30.972378337165992, 69.51043221623695],
                         [30.972378337166045, 71.29021737248695],
                         [32.38872602390184, 71.30120370061195]],
    'reduced_region_6': [[32.38872602390184, 71.36162850529945],
                         [30.972378337165992, 71.37261483342445],
                         [30.995925051879148, 73.00408455998695],
                         [32.407278561516435, 72.98760506779945]],
    'reduced_region_7': [[32.407278561516435, 73.05352303654945],
                         [31.010050291052025, 73.06450936467445],
                         [31.024173437313156, 74.65752694279945],
                         [32.40727856151646, 74.64104745061195]]
}


def convert_lat_lon_to_xy(ds, coordinates):
    lon_in, lat_in = coordinates
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def get_combination(example, bands):
    example_array = np.nan_to_num(example.GetRasterBand(bands[0]).ReadAsArray())
    example_array = misc.imresize(example_array, (500, 500))
    for i in bands[1:]:
        next_band = np.nan_to_num(example.GetRasterBand(i).ReadAsArray())
        next_band = misc.imresize(next_band, (500, 500))
        example_array = np.dstack((example_array, next_band))
    return example_array


def get_index_band(example, bands):
    # important fixes before calculating ndvi
    band_0 = np.nan_to_num(example.GetRasterBand(bands[0]).ReadAsArray())
    band_1 = np.nan_to_num(example.GetRasterBand(bands[1]).ReadAsArray())
    band_0[band_0 == 0] = 1
    band_1[band_1 == 0] = 1
    # this turns the float arrays into uint8 images, bad!!!
    # band_0 = np.resize(band_0, (500, 500))
    # band_1 = np.resize(band_1, (500, 500))
    diff = band_0-band_1
    sum = band_0+band_1
    # pl.subplot(121)
    # pl.imshow(diff)
    # pl.subplot(122)
    # pl.imshow(sum)
    # pl.show()
    return np.nan_to_num(diff/sum)


def check_ndvi_clusters_in_image(example_path):
    example = gdal.Open(example_path)
    ndvi_band = get_index_band(example, bands=[5,4])  # bands 5 and 4, (counting starts from 1)
    print(ndvi_band.max(), ndvi_band.min())
    all_ndvis = ndvi_band.reshape(-1)
    pl.scatter(all_ndvis, range(len(all_ndvis)))
    pl.show()
    pass


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def get_rgb_from_landsat8(this_image):
    example = gdal.Open(this_image)
    example_array = get_combination(example=example, bands=[4,3,2])
    example_array = np.nan_to_num(example_array)
    example_array = (255*example_array).astype(np.uint8)
    example_array = histogram_equalize(example_array)
    return example_array


def check_image_against_label(this_example, full_label_file, this_region):
    show_image = get_rgb_from_landsat8(this_image=this_example)
    label_map = gdal.Open(full_label_file)
    channel = label_map.GetRasterBand(1)
    min_coords, _, max_coords, _ = all_coordinates[this_region]
    min_x, min_y = convert_lat_lon_to_xy(ds=label_map, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_xy(ds=label_map, coordinates=max_coords)
    label_image = channel.ReadAsArray(min_x, min_y, abs(max_x - min_x), abs(max_y - min_y))
    # resize if needed
    # show_image = np.resize(show_image, new_shape=(label_image.shape[1], label_image.shape[0], 3))
    show_image = misc.imresize(show_image, (500, 500, 3))
    label_image = misc.imresize(label_image, (500, 500))
    pl.subplot(131)
    pl.title('actual image: {}'.format(show_image.shape[:2]))
    pl.imshow(show_image)
    pl.subplot(132)
    pl.title('label image: {}'.format(label_image.shape[:2]))
    pl.imshow(label_image)

    # background = Image.fromarray(show_image)
    # foreground = Image.fromarray(label_image)
    # background.paste(foreground, (0, 0), foreground)
    # background = Image.blend(background, foreground, alpha=0.5)
    # background.show()
    merged = show_image.copy()  # np.asarray(background)
    merged[label_image == 20] = (255, 0, 0)
    pl.subplot(133)
    pl.title('merged image: {}'.format(merged.shape[:2]))
    pl.imshow(merged)
    pl.show()
    pass


if __name__ == '__main__':
    # check_image_against_label(this_example=sys.argv[1], full_label_file=sys.argv[2], this_region=sys.argv[3])
    check_ndvi_clusters_in_image(example_path='/home/annus/PycharmProjects/'
                                              'ForestCoverChange_inputs_and_numerical_results/reduced_landsat_images/'
                                              'reduced_landsat_images/2013/reduced_regions_landsat_2013_5.tif')







