

from __future__ import print_function
from __future__ import division
import os
import gdal
import pickle
import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt


def get_band_names(list_of_contents):
    """
        Will return a dictionary of names of bands
    :param list_of_contents:
    :return: dict of bands
    """
    bands = {}
    for x in list_of_contents:
        if x.endswith('.tif'):
            if 'band4' in x:
                bands['red'] = x
            if 'band3' in x:
                bands['green'] = x
            if 'band2' in x:
                bands['blue'] = x
            if 'band5' in x:
                bands['near_ir'] = x
            if 'band7' in x:
                bands['sh_ir2'] = x
            if 'ndvi' in x:
                bands['ndvi'] = x
    return bands


def convert(arr):
    return np.asarray(255/4096*arr).astype(np.int16)


def get_data_from_single_folder(path, analyse_thresh=False):
    rgb_file = os.path.join(path, 'rgb.png')
    enh_veg_file = os.path.join(path, 'enh_veg.png')
    fal_col_file = os.path.join(path, 'fal_col.png')
    ndvi_file = os.path.join(path, 'ndvi.png')
    directory = os.listdir(path)
    bands = get_band_names(directory)

    band_images = {}
    for band in bands.keys():
        content = gdal.Open(os.path.join(path, bands[band]))
        band_data = content.GetRasterBand(1)
        arr = np.asarray(band_data.ReadAsArray())
        band_images[band] = arr
    pass

    rgb = np.dstack((band_images['red'], band_images['green'], band_images['blue']))
    enhanced_veg = np.dstack((band_images['sh_ir2'], band_images['near_ir'], band_images['green']))
    false_color = np.dstack((band_images['near_ir'], band_images['red'], band_images['green']))
    rgb, enhanced_veg, false_color = map(convert, [rgb, enhanced_veg, false_color])
    scipy.misc.imsave(rgb_file, rgb)
    scipy.misc.imsave(enh_veg_file, enhanced_veg)
    scipy.misc.imsave(fal_col_file, false_color)

    # threshold the ndvi image for segmentation target
    thresh = 0.75
    ndvi_ = np.asarray(1/20000*(band_images['ndvi']+10000)).astype(np.float16)
    ndvi = np.zeros_like(ndvi_)
    ndvi[ndvi_>thresh] = 255
    scipy.misc.imsave(ndvi_file, ndvi)
    if analyse_thresh:
        read_images = map(Image.open, [rgb_file, enh_veg_file, fal_col_file, ndvi_file])
        evaluate_threshold(read_images)


def evaluate_threshold(images):
    # rgb, false, enhanced, ndvi = images
    # w = 2; h = 2
    fig = plt.figure(figsize=(2, 2))
    columns = 2
    rows = 2
    for i in range(columns * rows):
        images[i] = np.asarray(images[i])
        fig.add_subplot(rows, columns, i+1)
        if images[i].ndim == 2:
            plt.gray()
        plt.axis('off')
        plt.imshow(images[i])
    # fig.set_size_inches(np.array(fig.get_size_inches()) * len(images))
    plt.show()
    pass


def directory_interface(path):
    "basically calls get_data_from_single_folder on each folder in a directory"
    folders = os.listdir(path)
    for folder in folders:
        this_path = os.path.join(path, folder)
        get_data_from_single_folder(path=this_path)
        print('log: on {}'.format(folder))


def collect_similar_images(path, image_name, destination_folder):
    "Collect similar images from all the folders and place them in one folder"
    os.mkdir(destination_folder)
    folders = os.listdir(path)
    for folder in folders:
        this_path = os.path.join(path, folder)
        image = Image.open(os.path.join(this_path, image_name))
        scipy.misc.imsave(os.path.join(destination_folder, '{}_{}'.format(folder, image_name)), image)


def generate_dataset(images_path, labels_path, dest_path, test=False):
    "Save the dataset in the form of a numpy array that we can train on..."
    examples = os.listdir(images_path)
    dataset = []
    labels = []
    os.mkdir(dest_path)
    new_width = 60
    new_height = 60
    for example in examples:
        this_example = os.path.join(images_path, example)
        this_label = os.path.join(labels_path, this_example.split('/')[-1].split('_')[0]+'_ndvi.png')
        # better reshape for training...
        image = np.asarray(Image.open(this_example).resize((new_width, new_height), Image.ANTIALIAS))
        label = np.asarray(Image.open(this_label).resize((new_width, new_height), Image.ANTIALIAS))
        dataset.append(image)
        labels.append(label)
        # print('on example {}'.format(this_example))
    dataset = np.asarray(dataset)
    labels = np.asarray(labels)
    data_file = open(os.path.join(dest_path, 'data.pkl'), 'wb')
    labels_file = open(os.path.join(dest_path, 'labels.pkl'), 'wb')
    pickle.dump(dataset, data_file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels, labels_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('log: dataset size = {}, labels size = {}'.format(dataset.shape, labels.shape))
    if test:
        test_example = dataset[0,:]
        test_label = labels[0,:]
        fig = plt.figure(figsize=(1,2))
        fig.add_subplot(1, 2, 1)
        if test_example.ndim == 2:
            plt.gray()
        plt.imshow(test_label)
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        if test_label.ndim == 2:
            plt.gray()
        plt.imshow(test_example)
        plt.axis('off')
        plt.show()


def load_data(path):
    with open(os.path.join(path, 'data.pkl'), 'rb') as data:
        examples = pickle.load(data)
    with open(os.path.join(path, 'labels.pkl'), 'rb') as labels:
        targets = pickle.load(labels)
    print('examples shape = {}, labels shape = {}'.format(examples.shape, targets.shape))
    this_one = np.random.randint(0,examples.shape[0])
    test_example = examples[this_one,:,:]
    test_label = targets[this_one,:,:]
    fig = plt.figure(figsize=(1,2))
    fig.add_subplot(1, 2, 1)
    if test_example.ndim == 2:
        plt.gray()
    plt.imshow(test_example)
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    if test_label.ndim == 2:
        plt.gray()
    plt.imshow(test_label)
    plt.axis('off')
    plt.show()


def main():
    path = '/home/annus/Desktop/forest_cover_change/region_3_data/malakand/data/' \
           'espa-annuszulfiqar@gmail.com-07012018-001522-723/untars/LC081510362013041401T1-SC20180701002950'
    get_data_from_single_folder(path=path, analyse_thresh=True)
    # path = '/home/annus/Desktop/forest_cover_change/region_3_data/malakand/data/' \
    #        'espa-annuszulfiqar@gmail.com-07012018-001522-723/untars/'
    # directory_interface(path=path)
    # generate_dataset(images_path='/home/annus/Desktop/forest_cover_change/region_3_data/malakand/'
    #                              'data/espa-annuszulfiqar@gmail.com-07012018-001522-723/false_color',
    #                  labels_path='/home/annus/Desktop/forest_cover_change/region_3_data/malakand/'
    #                              'data/espa-annuszulfiqar@gmail.com-07012018-001522-723/ndvi',
    #                  dest_path='/home/annus/Desktop/forest_cover_change/region_3_data/malakand/'
    #                              'data/espa-annuszulfiqar@gmail.com-07012018-001522-723/training_data',
    #                  test=False)
    # load_data('/home/annus/Desktop/forest_cover_change/region_3_data/malakand/'
    #           'data/espa-annuszulfiqar@gmail.com-07012018-001522-723/training_data/')


if __name__ == '__main__':
    main()









