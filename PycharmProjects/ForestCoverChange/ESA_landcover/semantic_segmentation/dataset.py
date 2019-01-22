


from __future__ import print_function
from __future__ import division
import os
import cv2
import time
import gdal
import torch
import random
import pickle
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
    # np.save('/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/ESA_landcover_dataset/raw/'
    #         'label_full_test_site.npy', label)

    # let's reshape it to match our actual image
    label = misc.imresize(label, size=site_size, interp='nearest')
    label = ndimage.median_filter(label, size=7)
    # re_label = np.load('/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/ESA_landcover_dataset/raw/'
    #                    'label_full_test_site.npy')
    # re_label = misc.imresize(re_label, size=site_size, interp='nearest')
    # re_label = ndimage.median_filter(re_label, size=7)
    # print(np.all(label==re_label))

    # pl.imshow(label)
    # pl.title('Full label Image')
    # pl.show()

    # let's get the actual image now
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    x_size, y_size = image_ds.RasterXSize, image_ds.RasterYSize
    all_raster_bands = [image_ds.GetRasterBand(x) for x in bands]

    count = -1
    # stride = 3000  # for testing only
    error_pixels = 50  # add this to remove the error pixels at the boundary of the test image
    for i in range(y_size//stride):
        for j in range(x_size//stride):
            count += 1
            # read the raster band by band for this subset
            example_subset = np.nan_to_num(all_raster_bands[0].ReadAsArray(j*stride+error_pixels,
                                                                   i*stride+error_pixels,
                                                                   stride, stride))
            for band in all_raster_bands[1:]:
                example_subset = np.dstack((example_subset , np.nan_to_num(band.ReadAsArray(j*stride+error_pixels,
                                                                           i*stride+error_pixels,
                                                                           stride,
                                                                           stride))))
            show_image = np.asarray(255*(example_subset [:,:,[4,3,2]]/4096.0).clip(0,1), dtype=np.uint8)
            label_subset = label[i*stride+error_pixels:(i+1)*stride+error_pixels,
                                j*stride+error_pixels:(j+1)*stride+error_pixels]
            # image_subset[:,:,0] = label_subset

            # save this example/label pair of numpy arrays as a pickle file with an index
            this_example_save_path = os.path.join(destination, '{}.pkl'.format(count))
            with open(this_example_save_path, 'wb') as this_pickle:
                pickle.dump((example_subset, label_subset), file=this_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                print('log: Saved {}'.format(this_example_save_path))

            # pl.subplot(1,2,1)
            # pl.imshow(show_image)
            # pl.subplot(1,2,2)
            # pl.imshow(label_subset)
            # pl.show()
            pass
    pass


def check_generated_dataset(path_to_dataset):

    for count in range(266):
        this_example_save_path = os.path.join(path_to_dataset, '{}.pkl'.format(count))
        with open(this_example_save_path, 'rb') as this_pickle:
            print('log: Reading {}'.format(this_example_save_path))
            (example_subset, label_subset) = pickle.load(this_pickle)

        show_image = np.asarray(255 * (example_subset[:, :, [4, 3, 2]] / 4096.0).clip(0, 1), dtype=np.uint8)
        pl.subplot(1,2,1)
        pl.imshow(show_image)
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
    croped_image, croped_label, model_input_size = kwargs['image'], kwargs['label'], kwargs['model_input_size']
    """
        will create an example to train on...
    :param croped_image: np array of image
    :param croped_label: np array of label (colored)
    :return: image and labeled processed and augmented if needed
    """
    # if random.randint(0, 2) == 1:
    #     crop_size = model_input_size
    #     x = random.randint(0, croped_image.shape[0] - crop_size)
    #     y = random.randint(0, croped_image.shape[1] - crop_size)
    #     croped_image = croped_image[x:x + crop_size, y:y + crop_size, :]
    #     croped_label = croped_label[x:x + crop_size, y:y + crop_size]
    #     return croped_image, croped_label

    # first crop
    crop_size = kwargs['first_crop_size']
    x = random.randint(0, croped_image.shape[0] - crop_size)
    y = random.randint(0, croped_image.shape[1] - crop_size)
    croped_image = croped_image[x:x + crop_size, y:y + crop_size, :]
    croped_label = croped_label[x:x + crop_size, y:y + crop_size]

    #################################################################333333
    croped_label = np.expand_dims(croped_label, axis=2)
    #################################################################333333

    # choice on cropping
    choice = random.randint(0, 2)
    crop_size = model_input_size
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
    label_im = np.asarray(label_im/10, dtype=np.uint8)
    return label_im


def fix(target_image, total_labels):
    # 6 is the noise class, generated during augmentation
    target_image[target_image < 0] = 0
    target_image[target_image > total_labels] = 0
    return target_image


def toTensor(**kwargs):
    image, label = kwargs['image'], kwargs['label']
    'will convert image and label from numpy to torch tensor'
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).float(), torch.from_numpy(label).long()


def get_dataloaders_raw(images_path, bands, labels_path, save_data_path, block_size=256, model_input_size=64,
                        train_split=0.8, batch_size=16, num_workers=4, max_label=22):

    # This function is veryyy slow because it reads in the whole image for each example

    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, data_list, image_path, label_image, mode='train'):
            '''
            :param data_list: list of all
            :param raster_bands: raster bands list of examples image
            :param raster_size: raster spatial size
            :param model_input_size: size of image input to the model
            :param label_image: label image numpy array
            :param mode: train or test?
            '''
            super(dataset, self).__init__()
            self.data_list = data_list
            self.image_path = image_path
            # self.image_ds = gdal.Open(images_path)
            # self.raster_bands = [self.image_ds.GetRasterBand(x) for x in bands]
            self.block_size = block_size
            self.label_image = label_image
            self.model_input_size = model_input_size
            # this is just a rough estimate, actual augmentation will result in many more images
            self.images_per_dimension = self.block_size//model_input_size
            self.total_images = self.images_per_dimension * self.images_per_dimension * len(self.data_list)
            self.mode = mode
            # print(len(self), len(data_list))
            pass

        def __getitem__(self, k):
            image_ds = gdal.Open(self.image_path)
            raster_bands = [image_ds.GetRasterBand(x) for x in bands]
            if self.mode == 'test':
                # 1. find out which image subset is it and then crop out that area first
                # no augmentation here, just stride-wise cropping out subset images
                this_block = int(k/self.images_per_dimension**2)
                _, example_indices, label_indices = self.data_list[this_block]
                this_example = np.nan_to_num(raster_bands[0].ReadAsArray(*example_indices))
                for band in raster_bands[1:]:
                    this_example = np.dstack((this_example, np.nan_to_num(band.ReadAsArray(*example_indices))))
                this_label = self.label_image[label_indices[0]:label_indices[1], label_indices[2]:label_indices[3]]

                # 2. next find out which sub-subset is it and crop it out
                subset_sum = k % int(self.images_per_dimension**2)
                this_row = subset_sum // self.images_per_dimension
                this_col = subset_sum % self.images_per_dimension

                this_example_subset = this_example[this_row*self.model_input_size:(this_row+1)*self.model_input_size,
                                                    this_col*self.model_input_size:(this_col+1)*self.model_input_size,:]
                this_label_subset = this_label[this_row*self.model_input_size:(this_row+1)*self.model_input_size,
                                                this_col*self.model_input_size:(this_col+1)*self.model_input_size]
                this_label_subset = fix(convert_labels(this_label_subset), total_labels=max_label)

                # image_subset = np.asarray(255*(this_example_subset[:,:,[4,3,2]]/4096.0).clip(0, 1), dtype=np.uint8)
                # pl.subplot(1, 2, 1)
                # pl.imshow(image_subset)
                # pl.subplot(1, 2, 2)
                # pl.imshow(this_label_subset)
                # pl.show()
                this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset)
                image_ds = None
                return {'input': this_example_subset, 'label': this_label_subset}

            elif self.mode == 'train':
                # total images are 64 and x25 for augmentation
                subset_image_index = k % len(self.data_list)
                _, example_indices, label_indices = self.data_list[subset_image_index]

                # 1. get the whole training block first
                this_example = np.nan_to_num(raster_bands[0].ReadAsArray(*example_indices))
                for band in raster_bands[1:]:
                    this_example = np.dstack((this_example, np.nan_to_num(band.ReadAsArray(*example_indices))))

                this_label = self.label_image[label_indices[0]:label_indices[1], label_indices[2]:label_indices[3]]
                this_label = convert_labels(this_label)

                # 2. Next step, crop out from anywhere and get an augmented image and label
                this_example_subset, this_label_subset = crop_and_rotate(image=this_example, label=this_label,
                                                                         first_crop_size=block_size//2,
                                                                         model_input_size=model_input_size)
                this_label_subset = fix(this_label_subset[:,:,0], total_labels=max_label)

                # image_subset = np.asarray(255*(this_example_subset[:,:,[4,3,2]]/4096.0).clip(0, 1), dtype=np.uint8)
                # pl.subplot(1, 2, 1)
                # pl.imshow(image_subset)
                # pl.subplot(1, 2, 2)
                # pl.imshow(this_label_subset)
                # pl.show()

                this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset)
                image_ds = None
                return {'input': this_example_subset, 'label': this_label_subset}
            else:
                image_ds = None
                return -1

        def __len__(self):
            # x25 for training images because of augmentation
            return 25*self.total_images if self.mode == 'train' else self.total_images
    ######################################################################################
    # generate training and testing lists here
    error_pixels = 50  # add this to remove the error pixels at the boundary of the test image
    image_ds = gdal.Open(images_path)
    x_size, y_size = image_ds.RasterXSize, image_ds.RasterYSize
    if not os.path.exists(save_data_path):
        print('LOG: No Saved Data Found! Generating Now...')
        all_examples = []
        count = -1
        for i in range(y_size//block_size):
            for j in range(x_size//block_size):
                count += 1
                # tuple -> [image_subset_indices, label_subset_indices]
                all_examples.append((count, (j*block_size+error_pixels, i*block_size+error_pixels,
                                             block_size, block_size),
                                     (i*block_size+error_pixels,(i+1)*block_size+error_pixels,
                                      j*block_size+error_pixels,(j+1)*block_size+error_pixels)))
        print('LOG: total examples:', len(all_examples))

        train_test_split = int(train_split*len(all_examples))
        random.shuffle(all_examples)
        train_eval_list = all_examples[:train_test_split]
        test_list = all_examples[train_test_split:]

        train_eval_split = int(0.90*len(train_eval_list))  # 10% of training examples are for validation
        random.shuffle(train_eval_list)
        train_list = train_eval_list[:train_eval_split]
        eval_list = train_eval_list[train_eval_split:]
        with open(save_data_path, 'wb') as save_pickle:
            pickle.dump((train_list, eval_list, test_list), file=save_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        print('LOG: Data saved!')
    else:
        print('LOG: Found Saved Data! Loading Now...')
        with open(save_data_path, 'rb') as save_pickle:
            train_list, eval_list, test_list = pickle.load(save_pickle)

    print('LOG: [train_list, eval_list, test_list] ->', map(len, (train_list, eval_list, test_list)))
    print('LOG: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))
    ######################################################################################
    # print([index for (index, tup1, tup2) in train_list])
    # print([index for (index, tup1, tup2) in eval_list])
    # print([index for (index, tup1, tup2) in test_list])

    # load the label image
    re_label = np.load(labels_path)
    re_label = misc.imresize(re_label, size=(y_size, x_size), interp='nearest')
    re_label = ndimage.median_filter(re_label, size=7)

    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    train_data = dataset(data_list=train_list, image_path=images_path, label_image=re_label, mode='train')
    eval_data = dataset(data_list=eval_list, image_path=images_path, label_image=re_label, mode='test')
    test_data = dataset(data_list=test_list, image_path=images_path, label_image=re_label, mode='test')

    # for j in range(10):
    #     print(train_data[j])
    # for j in range(len(eval_data)):
    #     print(eval_data[j])
    # for j in range(len(test_data)):
    #     print(test_data[j])

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def get_dataloaders_generated_data(generated_data_path, save_data_path, block_size=256, model_input_size=64,
                                   train_split=0.8, batch_size=16, num_workers=4, max_label=22):

    # This function is faster because we have already saved our data as subset pickle files

    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, data_list, mode='train'):
            super(dataset, self).__init__()
            self.data_list = data_list
            self.block_size = block_size
            self.model_input_size = model_input_size
            # this is just a rough estimate, actual augmentation will result in many more images
            self.images_per_dimension = self.block_size//model_input_size
            self.per_row = 500//model_input_size
            self.per_col = 450//model_input_size
            self.total_images = self.per_col*self.per_row*len(self.data_list)
            self.mode = mode
            # print(len(self), len(data_list))
            pass

        def __getitem__(self, k):
            if self.mode == 'train' or self.mode == 'test':
                # 1. find out which image subset is it and then crop out that area first
                # no augmentation here, just stride-wise cropping out subset images
                # we need, first which block image, next which row and column
                this_block = int(k/(self.per_row*self.per_col))
                this_subblock = int(k%(self.per_row*self.per_col))

                # 2. next find out which sub-subset is it and crop it out
                this_row = int(this_subblock/self.per_row)
                this_col = int(this_subblock % self.per_row)
                example_path = self.data_list[this_block]
                # print(this_block, this_subblock, this_row, this_col)
                with open(example_path, 'rb') as this_pickle:
                    # print('log: Reading {}'.format(example_path))
                    (example_subset, label_subset) = pickle.load(this_pickle)
                    example_subset = np.nan_to_num(example_subset)
                    # print(example_subset.shape, label_subset.shape)
                    example_subset = example_subset[:450, :500, :]
                    label_subset = label_subset[:450, :500]

                show_image = np.asarray(255*(example_subset[:,:,[4,3,2]]), dtype=np.uint8)
                pl.subplot(121)
                pl.imshow(show_image)
                pl.subplot(122)
                pl.imshow(label_subset)
                pl.show()

                # print(this_row*self.model_input_size,(this_row+1)*self.model_input_size,
                #                                     this_col*self.model_input_size,(this_col+1)*self.model_input_size)
                this_example_subset = example_subset[this_row*self.model_input_size:(this_row+1)*self.model_input_size,
                                                    this_col*self.model_input_size:(this_col+1)*self.model_input_size,:]
                this_label_subset = label_subset[this_row*self.model_input_size:(this_row+1)*self.model_input_size,
                                                this_col*self.model_input_size:(this_col+1)*self.model_input_size]
                this_label_subset = fix(convert_labels(this_label_subset), total_labels=max_label)

                # image_subset = np.asarray(255*(this_example_subset[:,:,[4,3,2]]/4096.0).clip(0, 1), dtype=np.uint8)
                # pl.subplot(1, 2, 1)
                # pl.imshow(image_subset)
                # pl.subplot(1, 2, 2)
                # pl.imshow(this_label_subset)
                # pl.show()
                this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset)
                return {'input': this_example_subset, 'label': this_label_subset}

            elif self.mode == 'gift':
                # get block image index first
                this_block = k % len(self.data_list)
                example_path = self.data_list[this_block]
                with open(example_path, 'rb') as this_pickle:
                    # print('log: Reading {}'.format(example_path))
                    (this_example, this_label) = pickle.load(this_pickle)

                this_label = convert_labels(this_label)

                # 2. Next step, crop out from anywhere and get an augmented image and label
                this_example_subset, this_label_subset = crop_and_rotate(image=this_example, label=this_label,
                                                                         first_crop_size=block_size//2,
                                                                         model_input_size=model_input_size)
                this_label_subset = fix(this_label_subset[:,:,0], total_labels=max_label)

                # image_subset = np.asarray(255*(this_example_subset[:,:,[4,3,2]]/4096.0).clip(0, 1), dtype=np.uint8)
                # pl.subplot(1, 2, 1)
                # pl.imshow(image_subset)
                # pl.subplot(1, 2, 2)
                # pl.imshow(this_label_subset)
                # pl.show()

                this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset)
                return {'input': this_example_subset, 'label': this_label_subset}
            else:
                return -1

        def __len__(self):
            # x25 for training images because of augmentation
            return self.total_images if self.mode == 'train' else self.total_images
    ######################################################################################


    # this is the list of paths to all of the training data
    all_examples = [os.path.join(generated_data_path, x) for x in os.listdir(generated_data_path)]

    if not os.path.exists(save_data_path):
        print('LOG: No Saved Data Found! Generating Now...')
        train_test_split = int(train_split*len(all_examples))
        random.shuffle(all_examples)
        train_eval_list = all_examples[:train_test_split]
        test_list = all_examples[train_test_split:]

        train_eval_split = int(0.90*len(train_eval_list))  # 10% of training examples are for validation
        random.shuffle(train_eval_list)
        train_list = train_eval_list[:train_eval_split]
        eval_list = train_eval_list[train_eval_split:]
        with open(save_data_path, 'wb') as save_pickle:
            pickle.dump((train_list, eval_list, test_list), file=save_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        print('LOG: Data saved!')
    else:
        print('LOG: Found Saved Data! Loading Now...')
        with open(save_data_path, 'rb') as save_pickle:
            train_list, eval_list, test_list = pickle.load(save_pickle)

    print('LOG: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))
    ######################################################################################

    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    train_data = dataset(data_list=train_list, mode='train')
    eval_data = dataset(data_list=eval_list, mode='test')
    test_data = dataset(data_list=test_list, mode='test')
    print('LOG: [train_data, eval_data, test_data] ->', len(train_data), len(eval_data), len(test_data))

    # for j in range(10):
    #     print(train_data[j])
    # for j in range(len(eval_data)):
    #     print(eval_data[j])
    # for j in range(len(test_data)):
    #     print(test_data[j])

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader



def main():
    # loaders = get_dataloaders(images_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                       'ESA_landcover_dataset/raw/full_test_site_2015.tif',
    #                           bands=range(1,14),
    #                           labels_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                       'ESA_landcover_dataset/raw/label_full_test_site.npy',
    #                           save_data_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                          'ESA_landcover_dataset/raw/pickled_data.pkl',
    #                           block_size=256, model_input_size=64, batch_size=16, num_workers=4)

    # loaders = get_dataloaders_raw(images_path='dataset/full_test_site_2015.tif',
    #                               bands=range(1,14),
    #                               labels_path='dataset/label_full_test_site.npy',
    #                               save_data_path='dataset/pickled_data.pkl',
    #                               block_size=256, model_input_size=64, batch_size=16, num_workers=6)

    loaders = get_dataloaders_generated_data(generated_data_path='/home/annus/PycharmProjects/'
                                                                 'ForestCoverChange_inputs_and_numerical_results/'
                                                                 'reduced_landsat_images/'
                                                                 'reduced_dataset_for_segmentation/2013/',
                                             save_data_path='pickled_data.pkl',
                                             block_size=400, model_input_size=64, batch_size=64,
                                             train_split=0.8, num_workers=4, max_label=22)

    # loaders = get_dataloaders_generated_data(generated_data_path='generated_dataset',
    #                                          save_data_path='pickled_generated_datalist.pkl',
    #                                          block_size=256, model_input_size=64, batch_size=128,
    #                                          train_split=0.8, num_workers=6, max_label=22)

    train_dataloader, val_dataloader, test_dataloader = loaders
    for idx, data in enumerate(train_dataloader):
        examples, labels = data['input'], data['label']
        print('-> on batch {}/{}, {}'.format(idx+1, len(train_dataloader), examples.size()))
        this_example_subset = (examples[0].numpy()).transpose(1,2,0)
        this = np.asarray(255*(this_example_subset[:,:,[4,3,2]]), dtype=np.uint8)
        that = labels[0].numpy().astype(np.uint8)
        # print()
        print(this.shape, that.shape, np.unique(that))
        pl.subplot(121)
        pl.imshow(this)
        pl.subplot(122)
        pl.imshow(that)
        pl.show()


if __name__ == '__main__':
    main()
    # get_images_from_large_file(image_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                       'ESA_landcover_dataset/raw/full_test_site_2015.tif',
    #                            bands=range(1,14),
    #                            label_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                       'ESA_landcover_dataset/raw/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif',
    #                            site_size=(3663, 5077),
    #                            min_coords=(34.46484326132815, 73.30923379854437),
    #                            max_coords=(34.13584821210507, 73.76516641573187),
    #                            destination='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                        'ESA_landcover_dataset/divided',
    #                            stride=256)

    # get_images_from_large_file(image_path='raw_dataset/full_test_site_2015.tif',
    #                            bands=range(1, 14),
    #                            label_path='raw_dataset/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif',
    #                            site_size=(3663, 5077),
    #                            min_coords=(34.46484326132815, 73.30923379854437),
    #                            max_coords=(34.13584821210507, 73.76516641573187),
    #                            destination='generated_dataset',
    #                            stride=256)

    # check_generated_dataset(path_to_dataset='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                        'ESA_landcover_dataset/divided')

    # get_dataloaders(images_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                             'ESA_landcover_dataset/raw/full_test_site_2015.tif',
    #                 bands=range(1,14),
    #                 labels_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                             'ESA_landcover_dataset/raw/label_full_test_site.npy',
    #                 save_data_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                'ESA_landcover_dataset/raw/pickled_data.pkl',
    #                 block_size=1500, model_input_size=500, batch_size=16)
    pass









