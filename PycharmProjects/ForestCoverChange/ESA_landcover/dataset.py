


from __future__ import print_function
from __future__ import division
import os
import cv2
import time
import gdal
import torch
import random
import numpy as np
random.seed(int(time.time()))
import matplotlib.pyplot as pl
import scipy.misc as misc
from scipy.ndimage import rotate
import scipy.ndimage as ndimage
from skimage.measure import block_reduce
from torch.utils.data import Dataset, DataLoader


def convert_lat_lon_to_xy(ds, coordinates):
    lon_in, lat_in = coordinates
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def get_images_from_large_file(image_path, bands, label_path, site_size, min_coords, max_coords, destination, stride):
    """
        This code generates our training images from our training site
    :param image_path: path to sentinel-2 satellite image
    :param bands: bands required from the test image
    :param label_path: path to ESA land cover map
    :param site_size: the spatial size of the actual image in pixels (rows, cols)
    :param min_coords: top-left coordinates in latitude and longitude
    :param max_coords: bottom right coordinates in latitude and longitude
    :param destination: folder path to save the dataset
    :param stride: step size for cropping out images
    :return: None
    """
    covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
    channel = covermap.GetRasterBand(1)
    min_x, min_y = convert_lat_lon_to_xy(ds=covermap, coordinates=min_coords)
    max_x, max_y = convert_lat_lon_to_xy(ds=covermap, coordinates=max_coords)
    # read the corresponding label at 360m per pixel resolution
    label = channel.ReadAsArray(min_x, min_y, abs(max_x - min_x), abs(max_y - min_y))

    # let's reshape it to match our actual image
    label = misc.imresize(label, size=site_size, interp='nearest')
    label = ndimage.median_filter(label, size=7)
    pl.imshow(label)
    pl.title('Full label Image')
    pl.show()

    # let's get the actual image now
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    x_size, y_size = image_ds.RasterXSize, image_ds.RasterYSize

    stride = 2000
    error_pixels = 50 # add this to remove the error pixels at the boundary of the test image
    for i in range(y_size//stride):
        for j in range(x_size//stride):
            # read the raster band by band for this subset
            subset = np.nan_to_num(image_ds.GetRasterBand(bands[0]).ReadAsArray(j*stride+error_pixels,
                                                                                i*stride+error_pixels,
                                                                                stride, stride))
            for n in bands[1:]:
                subset = np.dstack((subset, np.nan_to_num(image_ds.GetRasterBand(n).ReadAsArray(j*stride+error_pixels,
                                                                                                i*stride+error_pixels,
                                                                                                stride,
                                                                                                stride))))
            image_subset = np.asarray(255*(subset[:,:,[4,3,2]]/4096.0).clip(0,1), dtype=np.uint8)
            label_subset = label[j*stride+error_pixels:(j+1)*stride+error_pixels,
                                i*stride+error_pixels:(i+1)*stride+error_pixels]
            # image_subset[:,:,0] = label_subset
            pl.subplot(1,2,1)
            pl.imshow(image_subset)
            pl.subplot(1,2,2)
            pl.imshow(label_subset)
            pl.show()
            pass
    pass


#####################################################################################################3
# will implement the functionality (data augmentation) for
# 1. random crops,
# 2. random flips,
# 3. random rotations,


# we'll need these methods to generate random images from our dataset
def crop_center(img, crop_size):
    # will be used to crop an image at its center
    shape = img.shape
    if len(shape) == 2:
        x, y = shape
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[startx:startx + crop_size, starty:starty + crop_size]
    x, y, channels = shape
    startx = x // 2 - (crop_size // 2)
    starty = y // 2 - (crop_size // 2)
    return img[startx:startx + crop_size, starty:starty + crop_size, :]

def crop_and_rotate(**kwargs):
    croped_image, croped_label = kwargs['image'], kwargs['label']
    # print(np.unique(croped_label))
    """
        will create an example to train on...
    :param croped_image: np array of image
    :param croped_label: np array of label (colored)
    :return: image and labeled processed and augmented if needed
    """
    # first crop
    crop_size = 512
    x = random.randint(0, croped_image.shape[0] - crop_size)
    y = random.randint(0, croped_image.shape[1] - crop_size)
    croped_image = croped_image[x:x + crop_size, y:y + crop_size, :]
    croped_label = croped_label[x:x + crop_size, y:y + crop_size, :]

    #################################################################333333
    croped_label = convert_labels(label_im=croped_label)
    croped_label = np.expand_dims(fix(target_image=croped_label), axis=2)
    #################################################################333333

    # choice on cropping
    choice = random.randint(0, 2)
    crop_size = 64
    if choice == 0:  # just crop and return
        x = random.randint(0, croped_image.shape[0] - crop_size)
        y = random.randint(0, croped_image.shape[1] - crop_size)
        croped_image = croped_image[x:x + crop_size, y:y + crop_size, :]
        croped_label = croped_label[x:x + crop_size, y:y + crop_size]
        # print('simple crop')
    else:

        angle = random.randint(-179, 180)
        croped_image = crop_center(rotate(croped_image, angle=angle), crop_size=crop_size)
        croped_label = crop_center(rotate(croped_label, angle=angle), crop_size=crop_size)
        # print('fancy crop @ {}'.format(angle))

    # choice on flip
    choice = random.randint(0, 2)
    if choice == 1:  # flip it as well if 1, else no flip!
        second_choice_1 = random.randint(0, 2)
        if second_choice_1 == 0:
            croped_image = np.fliplr(croped_image)
            croped_label = np.fliplr(croped_label)
            # print('flip lr')
            # double-flip?
            second_choice_2 = random.randint(0, 2)
            if second_choice_2 == 1:
                croped_image = np.flipud(croped_image)
                croped_label = np.flipud(croped_label)
                # print('second flip lr')
        else:
            croped_image = np.flipud(croped_image)
            croped_label = np.flipud(croped_label)
            # print('flip ud')
            # double-flip?
            second_choice_2 = random.randint(0, 2)
            if second_choice_2 == 1:
                croped_image = np.fliplr(croped_image)
                croped_label = np.fliplr(croped_label)
                # print('second flip lr')
            pass

    return croped_image.copy(), croped_label.copy()

def convert_labels(label_im):
    # 29(0): background/clutter, 76(1): buildings, 150(2): trees,
    # 179(3): cars, 226(4): low_vegetation, 255(5): impervious_surfaces
    # conversions = {29:0, 76:1, 150:2, 179:3, 226:4, 255:5}
    # we are removing the cars class because of excessive downsampling used in the dataset, but not now...
    conversions = {29:0, 76:1, 150:2, 179:3, 226:4, 255:5}
    gray = cv2.cvtColor(label_im, cv2.COLOR_RGB2GRAY)
    for k in conversions.keys():
        gray[gray == k] = conversions[k]
    # print(np.unique(gray))
    return gray

def fix(target_image):
    # 6 is the noise class, generated during augmentation
    target_image[target_image < 0] = 6
    target_image[target_image > 5] = 6
    return target_image

def toTensor(**kwargs):
    image, label = kwargs['image'], kwargs['label']
    'will convert image and label from numpy to torch tensor'
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).float(), torch.from_numpy(label).long()


def get_dataloaders(images_path, labels_path, batch_size):
    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, data_dictionary, mode='train'):
            super(dataset, self).__init__()
            self.example_dictionary = data_dictionary
            self.mode = mode
            pass

        def __getitem__(self, k):
            example_path, label_path = self.example_dictionary[k]
            example_image = cv2.imread(example_path) # because pl imposed an alpha channel on it...
            target_image = cv2.imread(label_path)
            print(target_image.shape)

            # transforms
            if self.mode == 'train':
                # print('mode is not train')
                example_image, target_image = crop_and_rotate(image=example_image, label=target_image)
                target_image = target_image[:,:,0]


            example_image, target_image = toTensor(image=example_image, label=target_image)
            return {'input': example_image, 'label': target_image}

        def __len__(self):
            return len(self.example_dictionary)

    # train and test examples dirs only
    train_images_dir = os.path.join(images_path, 'train')
    train_labels_dir = os.path.join(labels_path, 'train')
    test_images_dir = os.path.join(images_path, 'test')
    test_labels_dir = os.path.join(labels_path, 'test')

    # create training set examples dictionary
    train_examples_dictionary = {}
    for name in os.listdir(train_images_dir):
        this_image_path = os.path.join(train_images_dir, name)
        this_label_path = os.path.join(train_labels_dir, name) # image because image and label have the same name!
        # for each index as key, we want to have its path and label as its items
        train_examples_dictionary[len(train_examples_dictionary)] = (this_image_path, this_label_path)

    # at this point, we have our train data dictionary mapping file paths to labels
    # we can split it into two dicts if we want, for example
    keys = train_examples_dictionary.keys()
    random.shuffle(keys)
    train_dictionary, val_dictionary = {}, {}
    for l, key in enumerate(keys):
        if l % 15 == 0:
            val_dictionary[len(val_dictionary)] = train_examples_dictionary[key]
            continue
        train_dictionary[len(train_dictionary)] = train_examples_dictionary[key]

    # create test set examples dictionary
    test_dictionary = {}
    for name in os.listdir(test_images_dir):
        this_image_path = os.path.join(test_images_dir, name)
        this_label_path = os.path.join(test_labels_dir, name)
        # for each index as key, we want to have its path and label as its...
        test_dictionary[len(test_dictionary)] = (this_image_path, this_label_path)

    # create dataset class instances
    train_data = dataset(data_dictionary=train_dictionary)
    val_data = dataset(data_dictionary=val_dictionary)
    test_data = dataset(data_dictionary=test_dictionary)
    test_data = dataset(data_dictionary=test_dictionary)
    print('train examples =', len(train_dictionary), 'val examples =', len(val_dictionary),
          'test examples =', len(test_dictionary))

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


def main():
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(images_path='/home/annus/_/'
                                                                                    'ISPRS_BENCHMARK_DATASETS/'
                                                                                    'Vaihingen/datainhere/'
                                                                                    'croped_jpg_images/',
                                                                        labels_path= '/home/annus/_/'
                                                                                    'ISPRS_BENCHMARK_DATASETS/'
                                                                                    'Vaihingen/datainhere/'
                                                                                    'croped_jpg_labels/',
                                                                        batch_size=1)
    # #
    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders(images_path='../../jpg_images/',
    #                                                                     labels_path='../../jpg_labels/',
    #                                                                     batch_size=16)

    # print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    count = 0
    while True:
        count += 1
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            print('{} -> on batch {}/{}, {}'.format(count, idx+1, len(train_dataloader), examples.size()))
            if True:
                this = (examples[0].numpy()).transpose(1,2,0).astype(np.uint8)
                that = labels[0].numpy().astype(np.uint8)
                print(np.unique(that))
                pl.subplot(121)
                pl.imshow(this)
                pl.subplot(122)
                pl.imshow(that)
                pl.show()


if __name__ == '__main__':
    # main()
    get_images_from_large_file(image_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                                          'ESA_landcover_dataset/raw/full_test_site_2015.tif',
                               bands=range(1,14),
                               label_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                                          'ESA_landcover_dataset/raw/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif',
                               site_size=(3663, 5077),
                               min_coords=(34.46484326132815, 73.30923379854437),
                               max_coords=(34.13584821210507, 73.76516641573187),
                               destination='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                                           'ESA_landcover_dataset/divided',
                               stride=256)













