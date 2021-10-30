

from __future__ import print_function, division
import os
import sys
# import png
import cv2
import gdal
import pickle
import imageio
import numpy as np
import PIL.Image as Image
import scipy.misc as misc
from scipy.ndimage import median_filter
import matplotlib.pyplot as pl
from matplotlib import colors

# for image registration
# from dipy.data import get_fnames
# from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
# from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
# import dipy.align.imwarp as imwarp
# from dipy.viz import regtools


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


offset = {
    'reduced_region_1': [17, 10],
    'reduced_region_2': [11, 5],
    'reduced_region_3': [5, 8],
    'reduced_region_4': [7, 8],
    'reduced_region_5': [28, 5],
    'reduced_region_6': [4, 14],
    'reduced_region_7': [15, 9]
}


homography_matrices = {
    'reduced_region_7': [[9.44992008e-01, -6.67993218e-03, 1.82962384e+01],
                         [-1.05305680e-02, 9.58651416e-01, 9.69074852e+00],
                         [-3.96235575e-05, -1.11081603e-05, 1.00000000e+00]],
    'reduced_region_6': [[9.74339147e-01, -1.21107019e-02, 5.02780567e+00],
                         [-2.96768712e-03, 9.61484910e-01, 8.45501196e+00],
                         [1.33400519e-05, -2.75922200e-05, 1.00000000e+00]],
    # 'reduced_region_5': [[1.00000000e+00, -3.06683539e-16, 9.84556895e-14],
    #                      [-1.99543640e-16, 1.00000000e+00, 7.38417672e-14],
    #                      [-9.30764774e-19, -6.44616120e-19, 1.00000000e+00]],
    'reduced_region_4': [[9.62296206e-01, -3.49573968e-03, 4.52386190e+00],
                         [-1.05782159e-02, 9.77340640e-01, 7.61574557e+00],
                         [-2.48444321e-05, 2.17193338e-05, 1.00000000e+00]],
    'reduced_region_3': [[1.00000000e+00, -3.06683539e-16, 9.84556895e-14],
                         [-1.99543640e-16, 1.00000000e+00, 7.38417672e-14],
                         [-9.30764774e-19, -6.44616120e-19, 1.00000000e+00]],
    'reduced_region_2': [[9.57686454e-01, -9.59738783e-03, 4.45568094e+00],
                         [-3.47093771e-03, 9.58133199e-01, 7.35584353e+00],
                         [-1.55444703e-05, -1.52156507e-05, 1.00000000e+00]],
    'reduced_region_1': [[9.78661750e-01, 5.03822813e-03, 1.00863737e+01],
                         [-1.41885904e-03, 9.90638796e-01, 4.10874813e+00],
                         [-8.50612875e-06, 3.66387912e-05, 1.00000000e+00]]
}


# pts_src -> pts_dest
homography_similarity_coordinates = {
    'reduced_region_7': ([[421, 98], [456, 293], [18, 369], [20, 195]],
                         [[423, 101], [457, 292], [33, 365], [36, 197]]),
    'reduced_region_6': ([[421, 98], [456, 293], [18, 369], [20, 195]],
                         [[423, 101], [457, 292], [33, 365], [36, 197]]),
    'reduced_region_5': ([[486, 116], [499, 280], [106, 252], [178, 123]],
                         [[486, 116], [499, 280], [106, 252], [178, 123]]),
    'reduced_region_4': ([[91, 235], [361, 436], [486, 401], [474, 283], [77, 442], [276, 464]],
                         [[91, 235], [348, 426], [474, 396], [462, 282], [77, 436], [268, 459]]),
    'reduced_region_3': ([[486, 116], [499, 280], [106, 252], [178, 123]],
                         [[486, 116], [499, 280], [106, 252], [178, 123]]),
    'reduced_region_2': ([[15, 305], [10, 188], [434, 340], [556, 41], [384, 23]],
                         [[16, 300], [12, 189], [422, 336], [541, 45], [375, 28]]),
    'reduced_region_1': ([[392, 253], [367, 411], [247, 163], [113, 363], [558, 334]],
                         [[392, 254], [367, 406], [252, 164], [121, 359], [554, 331]])
}


cropping_regions_after_homography = {
    'reduced_region_1': ([17, 15], [611, 474]),
    'reduced_region_2': ([16, 11], [618, 480]),
    'reduced_region_3': ([13, 4], [616, 480]),
    'reduced_region_4': ([12, 12], [531, 465]),
    'reduced_region_5': ([], []),
    'reduced_region_6': ([], []),
    'reduced_region_7': ([], []),
}


def convert_lat_lon_to_xy(ds, coordinates):
    lon_in, lat_in = coordinates
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    # print(1/px_w, 1/px_h)
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def get_combination(example, bands):
    example_array = np.nan_to_num(example.GetRasterBand(bands[0]).ReadAsArray())
    for i in bands[1:]:
        next_band = np.nan_to_num(example.GetRasterBand(i).ReadAsArray())
        example_array = np.dstack((example_array, next_band))
    return example_array


def get_rgb_from_landsat8(this_image):
    example = gdal.Open(this_image)
    example_array = get_combination(example=example, bands=[4,3,2])
    example_array = np.nan_to_num(example_array)
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


def convert_labels(label_im):
    label_im = np.asarray(label_im/10, dtype=np.uint8)
    return label_im


def check_image_against_label(this_example, full_label_file, this_region):
    show_image = get_rgb_from_landsat8(this_image=this_example) # unnormalized ofcourse
    # convert to rgb 255 images now
    show_image = histogram_equalize((255 * show_image).astype(np.uint8))
    label_map = gdal.Open(full_label_file)
    channel = label_map.GetRasterBand(1)
    min_coords, _, max_coords, _ = all_coordinates[this_region]
    min_x, min_y = convert_lat_lon_to_xy(ds=label_map, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_xy(ds=label_map, coordinates=max_coords)
    label_image = channel.ReadAsArray(min_x-offset[this_region][0], min_y-offset[this_region][1],
                                      abs(max_x - min_x), abs(max_y - min_y))
    # label_image = channel.ReadAsArray(min_x, min_y, abs(max_x - min_x), abs(max_y - min_y))
    # show_image = show_image[17:, 13:, :]
    # now extract a minimum image
    min_r = min(show_image.shape[0], label_image.shape[0])
    min_c = min(show_image.shape[1], label_image.shape[1])
    show_image = show_image[:min_r, :min_c, :]
    label_image = label_image[:min_r, :min_c]
    registered = show_image.copy()
    registered[label_image == 130] = (255, 0, 0) #(0, 255, 150)
    registered[label_image == 190] = (255, 0, 0)
    registered[label_image == 210] = (255, 0, 0) #(150, 255, 0)
    # print(registered[label_image == 210].shape)

    mng = pl.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
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


def make_CCI_dataset_numpy_from_image(this_example, full_label_file, this_region, pickle_path):
    this_ds = gdal.Open(this_example)
    example_array = get_combination(example=this_ds, bands=range(1,12))
    label_map = gdal.Open(full_label_file)
    channel = label_map.GetRasterBand(1)
    min_coords, _, max_coords, _ = all_coordinates[this_region]
    min_x, min_y = convert_lat_lon_to_xy(ds=label_map, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_xy(ds=label_map, coordinates=max_coords)
    label_image = channel.ReadAsArray(min_x - offset[this_region][0], min_y - offset[this_region][1],
                                      abs(max_x - min_x), abs(max_y - min_y))
    min_r = min(example_array.shape[0], label_image.shape[0])
    min_c = min(example_array.shape[1], label_image.shape[1])
    example_array = example_array[:min_r, :min_c, :]
    label_image = label_image[:min_r, :min_c]
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump((example_array, label_image), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    pass


def make_MODIS_dataset_numpy_from_image(this_example, this_label, this_region, pickle_path):
    this_ds = gdal.Open(this_example)
    example_array = get_combination(example=this_ds, bands=range(1, 12))
    label_image = -1 # indicating no label
    if os.path.exists(this_label):
        label_map = gdal.Open(this_label)
        label_image = label_map.GetRasterBand(1).ReadAsArray()
        min_r = min(example_array.shape[0], label_image.shape[0])
        min_c = min(example_array.shape[1], label_image.shape[1])
        example_array = example_array[:min_r, :min_c, :]
        label_image = label_image[:min_r, :min_c]
    else:
        print('LOG: No labeling passed.')
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump((example_array, label_image), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved {}'.format(pickle_path))
    pass


def generate_dataset_CCI(year):
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
        make_CCI_dataset_numpy_from_image(this_example=os.path.join(examples_path, single_example_name+'{}.tif'.format(i)),
                                      full_label_file=full_label_path,
                                      this_region=this_region_name+str(i),
                                      pickle_path=os.path.join(pickle_main_path, single_pickle_name+'{}.pkl'.format(i)))

    pass


def generate_dataset_MODIS(year):
    examples_path = '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/' \
                    'reduced_landsat_images/reduced_landsat_images/{}/'.format(year)
    # examples_path = '/home/annuszulfiqar/forest_cover/forestcoverUnet/ESA_landcover/reduced_regions_landsat/{}/'\
    #                 .format(year)
    single_example_name = 'reduced_regions_landsat_{}_'.format(year)
    labels_path = '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/' \
                  'modis_land_covermaps/{}'.format(year)
    # labels_path = '/home/annuszulfiqar/forest_cover/forestcoverUnet/ESA_landcover/reduced_regions_landsat/' \
    #               'modis_land_covermaps/{}'.format(year)
    single_label_name = 'covermap_{}_reduced_region_'.format(year)
    this_region_name = 'reduced_region_'
    pickle_main_path = '/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/' \
                       'reduced_landsat_images/reduced_dataset_for_segmentation_MODIS/'
    # pickle_main_path = '/home/annuszulfiqar/forest_cover/forestcoverUnet/ESA_landcover/reduced_regions_landsat/dataset'
    single_pickle_name = 'reduced_regions_landsat_{}_'.format(year)

    for region in range(1,13):
        make_MODIS_dataset_numpy_from_image(this_example=os.path.join(examples_path, single_example_name+'{}.tif'
                                                                      .format(region)),
                                      this_label=os.path.join(labels_path, single_label_name+'{}.tif'.format(region)),
                                      this_region=this_region_name+str(region),
                                      pickle_path=os.path.join(pickle_main_path, single_pickle_name+'{}.pkl'
                                                               .format(region)))
    pass


def check_temporal_map_difference(label_1, label_2):
    label_map_1 = gdal.Open(label_1)
    label_map_2 = gdal.Open(label_2)
    channel_1 = label_map_1.GetRasterBand(1)
    channel_2 = label_map_2.GetRasterBand(1)
    label_image_1 = channel_1.ReadAsArray()
    label_image_2 = channel_2.ReadAsArray()
    label_image_1 = np.nan_to_num(label_image_1).astype(np.float)
    label_image_2 = np.nan_to_num(label_image_2).astype(np.float)
    # convert to binary image, 0-> noise, 3-> water, map both to non-forest->2
    label_image_1[label_image_1 == 0] = 2
    label_image_1[label_image_1 == 3] = 2
    label_image_2[label_image_2 == 0] = 2
    label_image_2[label_image_2 == 3] = 2
    # convert 1,2 to 0,1
    label_image_1 -= 1
    label_image_2 -= 1

    # positive valued differencing by adding an offset of
    diff_1 = label_image_2 - label_image_1
    sum_1 = label_image_2 + label_image_1
    mng = pl.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    pl.subplot(221)
    pl.title('first image {}'.format(label_image_1.shape[:2]))
    pl.imshow(label_image_1) # , cmap='Paired'
    pl.subplot(222)
    pl.title('second image {}'.format(label_image_1.shape[:2]))
    pl.imshow(label_image_1)
    pl.subplot(223)
    pl.title('difference image {}'.format(diff_1.shape[:2]))
    pl.imshow(diff_1)
    pl.subplot(224)
    pl.title('sum image {}'.format(sum_1.shape[:2]))
    pl.imshow(sum_1)
    pl.show()
    pass


def check_generated_numpy(pathtonumpy):
    with open(pathtonumpy, 'rb') as dataset:
        example_array, label_array = pickle.load(dataset)
    this = histogram_equalize(np.asarray(255*(example_array[:,:,[3,2,1]]), dtype=np.uint8))

    print(this.shape, label_array.shape)
    registered = this.copy()
    registered[label_array == 0] = (255, 0, 0) #(0, 255, 150)
    registered[label_array == 13] = (255, 0, 0) #(0, 255, 150)
    registered[label_array == 15] = (255, 0, 0)

    pl.subplot(131)
    pl.imshow(this)
    pl.subplot(132)
    pl.imshow(label_array)
    pl.subplot(133)
    pl.imshow(registered)
    pl.show()
    pass


# def register_label_on_image(pathtonumpy):
#     with open(pathtonumpy, 'rb') as dataset:
#         moving, static = pickle.load(dataset)
#     moving_single = moving[:,:,4]
#     moving = histogram_equalize(np.asarray(255*(moving[:,:,[4,3,2]]), dtype=np.uint8))
#     overlaid = moving.copy()
#     overlaid[static == 130] = (255, 0, 0)  # (0, 255, 150)
#     overlaid[static == 190] = (255, 0, 0)
#     overlaid[static == 210] = (255, 0, 0)  # (150, 255, 0)
#
#     # pl.imshow(moving_single)
#     # pl.show()
#     regtools.overlay_images(static, moving_single, 'Static', 'Overlay', 'Moving')
#     dim = static.ndim
#     metric = SSDMetric(dim)
#     level_iters = [200, 100, 50, 25]
#     # level_iters = [1, 1]
#     sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50)
#     mapping = sdr.optimize(static, moving_single)
#     warped_moving_0 = mapping.transform(moving[:,:,0], 'linear')
#     warped_moving_1 = mapping.transform(moving[:,:,1], 'linear')
#     warped_moving_2 = mapping.transform(moving[:,:,2], 'linear')
#     warped_moving = np.dstack((warped_moving_0, warped_moving_1, warped_moving_2))
#     # print(warped_moving)
#     warped_moving = np.asarray(warped_moving, dtype=np.uint8)
#     overlaid_again = warped_moving.copy()
#     overlaid_again[static == 130] = (255, 0, 0)  # (0, 255, 150)
#     overlaid_again[static == 190] = (255, 0, 0)
#     overlaid_again[static == 210] = (255, 0, 0)  # (150, 255, 0)
#
#     pl.subplot(131)
#     pl.imshow(static)
#     pl.subplot(132)
#     pl.imshow(overlaid)
#     pl.subplot(133)
#     pl.imshow(overlaid_again)
#     pl.show()
#     pass


def label_image_homography(pathtonumpy, this_region):
    with open(pathtonumpy, 'rb') as dataset:
        all_bands, static = pickle.load(dataset)
    moving_single = all_bands[:,:,4]
    # print(all_bands)
    moving = histogram_equalize(np.asarray(255*(all_bands[:,:,[3,2,1]]), dtype=np.uint8))
    overlaid = moving.copy()
    # overlaid[static == 130] = (255, 0, 0)  # (0, 255, 150)
    overlaid[static == 190] = (255, 0, 0)
    overlaid[static == 210] = (255, 0, 0)  # (150, 255, 0)
    overlaid[static == 200] = (255, 0, 0)  # (150, 255, 0)
    # overlaid[static == 20] = (255, 0, 0)  # (0, 255, 150)

    # Four corners in source image
    pts_src = np.array([[553, 75], [496, 329], [614, 189], [55, 226]])

    # Four corners in destination image.
    pts_dst = np.array([[551, 91], [494, 317], [614, 189], [68, 224]])

    # Calculate Homography
    if this_region in homography_matrices.keys():
        h = np.asarray(homography_matrices[this_region])
        print('using previously saved homography matrix')
    else:
        h, status = cv2.findHomography(pts_src, pts_dst)
        print('finding new homography matrix')
        print(h, status)

    # Warp source image to destination based on homography
    original = all_bands[:,:,[3,2,1]]
    im_out = cv2.warpPerspective(original, h, (static.shape[1], static.shape[0]))
    im_out = histogram_equalize(np.asarray(255*im_out, dtype=np.uint8))

    overlaid_reg = im_out.copy()
    # overlaid_reg[static == 130] = (255, 0, 0)  # (0, 255, 150)
    overlaid_reg[static == 190] = (255, 0, 0)
    overlaid_reg[static == 210] = (255, 0, 0)  # (150, 255, 0)
    overlaid_reg[static == 200] = (255, 0, 0)  # (150, 255, 0)
    # overlaid_reg[static == 20] = (150, 100, 200)  # (0, 255, 150)

    mng = pl.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    pl.subplot(131)
    pl.imshow(moving)
    pl.subplot(132)
    pl.imshow(static)
    pl.subplot(133)
    pl.imshow(overlaid)
    pl.figure()
    mng = pl.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    pl.imshow(overlaid_reg)
    pl.show()

    # Display images
    # cv2.imshow("Source Image", im_src)
    # cv2.imshow("Destination Image", im_dst)
    # cv2.imshow("Warped Source Image", im_out)
    #
    # cv2.waitKey(0)
    pass


def check_MODIS_earth_engine_label(this_example, full_label_file):
    # gdal_ds = gdal.Open(this_example)
    show_image = get_rgb_from_landsat8(this_image=this_example)
    # convert to rgb 255 images now
    show_image = histogram_equalize((255 * show_image).astype(np.uint8))
    label_map = gdal.Open(full_label_file)
    print(full_label_file)
    channel = label_map.GetRasterBand(1)
    label_image = channel.ReadAsArray()
    label_image[np.isnan(label_image)] = -1
    min_r = min(show_image.shape[0], label_image.shape[0])
    min_c = min(show_image.shape[1], label_image.shape[1])
    safe_pixels = 20
    show_image = show_image[safe_pixels:min_r-safe_pixels, safe_pixels:min_c-safe_pixels, :]
    label_image = label_image[safe_pixels:min_r-safe_pixels, safe_pixels:min_c-safe_pixels]
    print(np.unique(label_image, return_counts=True))
    registered = show_image.copy()
    registered[label_image == 0] = (255, 0, 0)  # (0, 255, 150)
    # registered[label_image == 12] = (255, 0, 0)
    registered[label_image == 13] = (255, 0, 0)
    registered[label_image == 15] = (255, 0, 0)  # (150, 255, 0)
    registered[np.where(np.logical_and(label_image >= 1, label_image <= 5))] = (0, 255, 0)  # (150, 255, 0)

    # print(registered[label_image == 210].shape)

    # print(label_image[label_image==-1].shape)
    mng = pl.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    pl.subplot(131)
    pl.title('actual image: {}'.format(show_image.shape[:2]))
    pl.imshow(show_image)
    pl.subplot(132)
    pl.title('label image: {}'.format(label_image.shape[:2]))
    pl.imshow(label_image)
    pl.subplot(133)
    pl.title('registered image: {}'.format(registered.shape[:2]))
    pl.imshow(registered)
    pl.show()
    pass


def check_generated_map_palsar(label_path, generated_path):
    with open(generated_path, 'rb') as generated:
        generated_label = np.load(generated).astype(np.uint8)
        generated_label = median_filter(generated_label, size=3)
    rows, cols = generated_label.shape
    covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
    channel = covermap.GetRasterBand(1)
    inference_label = np.nan_to_num(channel.ReadAsArray()).astype(np.uint8)[:rows, :cols]
    inference_label[inference_label == 0] = 2
    inference_label[inference_label == 3] = 2
    inference_label -= 1
    uniq_labels, counts = np.unique(inference_label, return_counts=True)
    forest_position = np.argmax(uniq_labels==0)
    forest_pixels = float(counts[forest_position])
    total_pixels = float(counts.sum())
    forest_percentage = forest_pixels*100/total_pixels
    print('original forest: {:.3f}%'.format(forest_percentage))
    uniq_labels, counts = np.unique(generated_label, return_counts=True)
    forest_position = np.argmax(uniq_labels == 0)
    forest_pixels = float(counts[forest_position])
    total_pixels = float(counts.sum())
    forest_percentage = forest_pixels * 100 / total_pixels
    print('generated forest: {:.3f}%'.format(forest_percentage))
    accuracy = 100 * (inference_label == generated_label).sum() / (inference_label.shape[0] * inference_label.shape[1])
    print('accuracy: {:.3f}%'.format(accuracy))
    # for better visualization, we define our own cmaps
    forest_cmap = colors.ListedColormap(['green', 'black'])
    bounds = [-0.5, 0.5, 1.5]  # between each two numbers is one corresponding color
    forest_norm = colors.BoundaryNorm(bounds, forest_cmap.N)
    pl.subplot(121)
    pl.title('original')
    pl.imshow(inference_label, cmap=forest_cmap, norm=forest_norm)
    pl.subplot(122)
    pl.title('generated')
    pl.imshow(generated_label, cmap=forest_cmap, norm=forest_norm)
    pl.show()
    pass


def generate_palsar_series_gif(region, destination):
    images = []
    years = [2007, 2008, 2009, 2010, 2015, 2016, 2017]
    parent_folder = '/home/annus/Desktop/palsar/generated_maps/using_separate_models'
    filenames = [os.path.join(parent_folder, 'generated_{}_{}.npy'.format(year, region)) for year in years]
    for filename in filenames:
        with open(filename, 'rb') as generated:
            generated_label = np.load(generated).astype(np.uint8)
            # since forest is 0 and non-forest is 1, we'll have to swap their labels for better visualization
            generated_label = 1 - generated_label
            generated_label = 255*median_filter(generated_label, size=3)
            generated_covermap = np.dstack((np.zeros_like(generated_label), generated_label,
                                            np.zeros_like(generated_label)))
        images.append(generated_covermap)
    imageio.mimsave(destination, images)


if __name__ == '__main__':
    # check_image_against_label(this_example=sys.argv[1], full_label_file=sys.argv[2], this_region=sys.argv[3])

    # check_ndvi_clusters_in_image(example_path='/home/annus/PycharmProjects/'
    #                                           'ForestCoverChange_inputs_and_numerical_results/reduced_landsat_images/'
    #                                           'reduced_landsat_images/2013/reduced_regions_landsat_2013_5.tif')

    # generate_dataset(year='2015')

    # generate_dataset_MODIS(year=sys.argv[1])

    # label_image_homography(pathtonumpy='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                   'reduced_landsat_images/reduced_dataset_for_segmentation/2015/'
    #                                   'reduced_regions_landsat_2015_{}.pkl'.format(sys.argv[1]),
    #                        this_region='reduced_region_{}'.format(sys.argv[1]))

    # check_generated_numpy(pathtonumpy='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                   'reduced_landsat_images/reduced_dataset_for_segmentation_MODIS/'
    #                                   'reduced_regions_landsat_2016_12.pkl')

    # register_label_on_image(pathtonumpy='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                     'reduced_landsat_images/reduced_dataset_for_segmentation/2015/'
    #                                     'reduced_regions_landsat_2015_7.pkl')

    # check_temporal_map_difference(label_1='/home/annus/Desktop/palsar/palsar_dataset_full/palsar_dataset/'
    #                                       'fnf_2007_region_1.tif',
    #                               label_2='/home/annus/Desktop/palsar/palsar_dataset_full/palsar_dataset/'
    #                                       'fnf_2008_region_1.tif')

    # check_MODIS_earth_engine_label(this_example='/home/annus/PycharmProjects/'
    #                                             'ForestCoverChange_inputs_and_numerical_results/reduced_landsat_images/'
    #                                             'reduced_landsat_images/{}/reduced_regions_landsat_{}_{}.tif'
    #                                .format(sys.argv[1], sys.argv[1], sys.argv[2]),
    #                                full_label_file='/home/annus/PycharmProjects/'
    #                                                'ForestCoverChange_inputs_and_numerical_results/'
    #                                                'modis_land_covermaps/{}/covermap_{}_reduced_region_{}.tif'
    #                                .format(sys.argv[1], sys.argv[1], sys.argv[2]))

    # check_MODIS_earth_engine_label(this_example='/home/annus/Desktop/new_regions_images/'
    #                                             'reduced_regions_landsat_{}_{}.tif'
    #                                .format(sys.argv[1], sys.argv[2]),
    #                                full_label_file='/home/annus/Desktop/new_regions_labels/'
    #                                                'covermap_{}_reduced_region_{}.tif'
    #                                .format(sys.argv[1], sys.argv[2]))

    # watch -n2 rsync -avh /home/annus/PycharmProjects/ForestCoverChange/ESA_landcover/  -a annuszulfiqar@111.68.101.28:forest_cover/forestcoverUnet/ESA_landcover/
    # watch -n2 rsync -avh /home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/modis_land_covermaps/   -a annuszulfiqar@111.68.101.28:forest_cover/forestcoverUnet/ESA_landcover/reduced_regions_landsat/modis_land_covermaps

    check_generated_map_palsar(label_path=sys.argv[1], generated_path=sys.argv[2])
    # generate_palsar_series_gif(region=sys.argv[1], destination=sys.argv[2])




