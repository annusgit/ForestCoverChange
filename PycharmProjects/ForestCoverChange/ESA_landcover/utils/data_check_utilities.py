

from __future__ import print_function, division
import os
import sys
import png
import cv2
import gdal
import pickle
import numpy as np
import PIL.Image as Image
import scipy.misc as misc
import matplotlib.pyplot as pl
# for image registration
from dipy.data import get_fnames
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import dipy.align.imwarp as imwarp
from dipy.viz import regtools



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
    for i in bands[1:]:
        next_band = np.nan_to_num(example.GetRasterBand(i).ReadAsArray())
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
    return example_array


def check_image_against_label(this_example, full_label_file, this_region):
    register = False
    show_image = get_rgb_from_landsat8(this_image=this_example) # unnormalized ofcourse
    label_map = gdal.Open(full_label_file)
    channel = label_map.GetRasterBand(1)
    min_coords, _, max_coords, _ = all_coordinates[this_region]
    min_x, min_y = convert_lat_lon_to_xy(ds=label_map, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_xy(ds=label_map, coordinates=max_coords)
    label_image = channel.ReadAsArray(min_x-18, min_y-9, abs(max_x - min_x), abs(max_y - min_y))
    # label_image = channel.ReadAsArray(min_x, min_y, abs(max_x - min_x), abs(max_y - min_y))
    # show_image = show_image[17:, 13:, :]
    # now extract a minimum image
    min_r = min(show_image.shape[0], label_image.shape[0])
    min_c = min(show_image.shape[1], label_image.shape[1])
    # show_image = show_image[:min_r, :min_c, :]
    # label_image = label_image[:min_r, :min_c]

    if register:
        registered = np.zeros((label_image.shape[0], label_image.shape[1], 3))
        moving = show_image.copy()[:,:,0]
        static = label_image.copy()
        dim = static.ndim
        metric = SSDMetric(dim)
        level_iters = [200, 100, 50, 25]
        # level_iters = [5, 5]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50)
        mapping = sdr.optimize(static, moving)
        registered[:, :, 0] = mapping.transform(show_image[:, :, 0], 'linear')
        registered[:, :, 1] = mapping.transform(show_image[:, :, 1], 'linear')
        registered[:, :, 2] = mapping.transform(show_image[:, :, 2], 'linear')
    else:
        pass

    # convert to rgb 255 images now
    show_image = histogram_equalize((255*show_image).astype(np.uint8))
    if register:
        registered = histogram_equalize((255*registered).astype(np.uint8))
        # show_image[label_image == 190] = (255, 0, 0)
        registered[label_image == 190] = (255, 0, 0)
    else:
        # registered
        registered = show_image
        pass
    pl.subplot(131)
    pl.title('label image: {}'.format(label_image.shape[:2]))
    pl.imshow(label_image)
    pl.subplot(132)
    pl.title('actual image: {}'.format(show_image.shape[:2]))
    pl.imshow(show_image)
    pl.subplot(133)
    pl.title('registered image: {}'.format(registered.shape[:2]))
    pl.imshow(registered)
    pl.show()
    pass


def make_dataset_numpy_from_image(this_example, full_label_file, this_region, pickle_path):
    this_ds = gdal.Open(this_example)
    example_array = get_combination(example=this_ds, bands=range(1,12))
    label_map = gdal.Open(full_label_file)
    channel = label_map.GetRasterBand(1)
    min_coords, _, max_coords, _ = all_coordinates[this_region]
    min_x, min_y = convert_lat_lon_to_xy(ds=label_map, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_xy(ds=label_map, coordinates=max_coords)
    label_image = channel.ReadAsArray(min_x, min_y, abs(max_x - min_x), abs(max_y - min_y))
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump((example_array, label_image), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    pass


def generate_dataset(year):
    examples_path = '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/' \
                    'reduced_landsat_images/reduced_landsat_images/{}/'.format(year)
    single_example_name = 'reduced_regions_landsat_{}_'.format(year)
    full_label_path = '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/' \
                      'land_cover_maps/ESACCI-LC-L4-LCCS-Map-300m-P1Y-{}-v2.0.7.tif'.format(year)
    this_region_name = 'reduced_region_'
    pickle_main_path = '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/' \
                       'reduced_landsat_images/reduced_dataset_for_segmentation/'
    single_pickle_name = 'reduced_regions_landsat_{}_'.format(year)

    for i in range(1,8):
        make_dataset_numpy_from_image(this_example=os.path.join(examples_path, single_example_name+'{}.tif'.format(i)),
                                      full_label_file=full_label_path,
                                      this_region=this_region_name+str(i),
                                      pickle_path=os.path.join(pickle_main_path, single_pickle_name+'{}.pkl'.format(i)))

    pass


if __name__ == '__main__':
    check_image_against_label(this_example=sys.argv[1], full_label_file=sys.argv[2], this_region=sys.argv[3])
    # check_ndvi_clusters_in_image(example_path='/home/annus/PycharmProjects/'
    #                                           'ForestCoverChange_inputs_and_numerical_results/reduced_landsat_images/'
    #                                           'reduced_landsat_images/2013/reduced_regions_landsat_2013_5.tif')

    # generate_dataset(year='2015')







