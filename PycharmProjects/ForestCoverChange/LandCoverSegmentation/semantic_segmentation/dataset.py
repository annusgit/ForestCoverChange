

from __future__ import print_function
from __future__ import division
import os
# import cv2
import sys
import time
import gdal
import torch
import random
import pickle
import numpy as np
import PIL.Image as Im
np.random.seed(int(time.time()))
random.seed(int(time.time()))
import matplotlib.pyplot as pl
import scipy.misc as misc
from scipy.ndimage import rotate
import scipy.ndimage as ndimage
# from skimage.measure import block_reduce
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def convert_lat_lon_to_xy(ds, coordinates):
    lon_in, lat_in = coordinates
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def histogram_equalize(img):
    # b, g, r = cv2.split(img)
    # red = cv2.equalizeHist(r)
    # green = cv2.equalizeHist(g)
    # blue = cv2.equalizeHist(b)
    # return cv2.merge((blue, green, red))
    pass


def adaptive_resize(array, new_shape):
    # reshape the labels to the size of the image
    single_band = Im.fromarray(array)
    single_band_resized = single_band.resize(new_shape, Im.NEAREST)
    return np.asarray(single_band_resized)


def get_images_from_large_file(bands, year, region, stride):
    # data_directory_path = '/home/annuszulfiqar/palsar_dataset_full/palsar_dataset/'
    data_directory_path = '/home/azulfiqar_bee15seecs/sentinel-2/'
    image_path = os.path.join(data_directory_path, 'sentinel2_{}_region_{}.tif'.format(year, region))
    label_path = os.path.join(data_directory_path, 'fnf_self_{}_region_{}.tif'.format(year, region))
    destination_directory_path = 'sentinel2_all_bt_generated_dataset' #os.path.join(data_directory_path, '{}/generated_dataset/'.format(region))
    # destination = os.path.join(destination_directory_path, '{}'.format(year))
    destination = destination_directory_path
    if not os.path.exists(destination):
        print('Log: Making parent directory: {}'.format(destination))
        os.mkdir(destination)
    print(image_path, label_path)
    # we will use this to divide those fnf images
    covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
    channel = covermap.GetRasterBand(1)
    # big_x_size, big_y_size = covermap.RasterXSize, covermap.RasterYSize
    label = channel.ReadAsArray()
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    x_size, y_size = image_ds.RasterXSize, image_ds.RasterYSize
    # we need the difference of the two raster sizes to do the resizing
    label = adaptive_resize(label, new_shape=(x_size, y_size))
    # print(label.shape, (y_size, x_size))
    all_raster_bands = [image_ds.GetRasterBand(x) for x in bands]

    count = 0
    # stride = 3000  # for testing only
    for i in range(y_size//stride):
        for j in range(x_size//stride):
            count += 1
            # read the raster band by band for this subset
            example_subset = np.nan_to_num(all_raster_bands[0].ReadAsArray(j*stride,
                                                                           i*stride,
                                                                           stride, stride))
            for band in all_raster_bands[1:]:
                example_subset = np.dstack((example_subset, np.nan_to_num(band.ReadAsArray(j*stride,
                                                                                           i*stride,
                                                                                           stride,
                                                                                           stride))))
            label_subset = label[i*stride:(i+1)*stride, j*stride:(j+1)*stride]

            # save this example/label pair of numpy arrays as a pickle file with an index
            this_example_save_path = os.path.join(destination, '{}_{}_{}.pkl'.format(year, region, count))
            with open(this_example_save_path, 'wb') as this_pickle:
                pickle.dump((example_subset, label_subset), file=this_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                print('log: Saved {} '.format(this_example_save_path), end='')
                print(i*stride, (i+1)*stride, j*stride, (j+1)*stride)
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
    # target_image[target_image < 0] = -1
    # target_image = target_image - 1 #[target_image > 2] = 2, convert 1,2,3 to 0,1,2
    return target_image


def toTensor(**kwargs):
    image, label = kwargs['image'], kwargs['label']
    'will convert image and label from numpy to torch tensor'
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    if kwargs['one_hot']:
        label = label.transpose((2, 0, 1))
        return torch.from_numpy(image).float(), torch.from_numpy(label).float()
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


def get_dataloaders_generated_data(generated_data_path, save_data_path, model_input_size=64, num_classes=4,
                                   train_split=0.8, one_hot=False, batch_size=16, num_workers=4, max_label=3):

    # This function is faster because we have already saved our data as subset pickle files
    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, data_list, data_map_path, stride, mode='train', transformation=None):
            super(dataset, self).__init__()
            self.model_input_size = model_input_size
            self.data_list = data_list
            self.all_images = []
            self.total_images = 0
            self.stride = stride
            self.one_hot = one_hot
            self.num_classes = num_classes
            self.mode = mode
            self.transformation = transformation

            if os.path.exists(data_map_path):
                print('LOG: Saved data map found! Loading now...')
                with open(data_map_path, 'rb') as data_map:
                    self.data_list, self.all_images = pickle.load(data_map)
                    self.total_images = len(self.all_images)
            else:
                print('LOG: No data map found! Generating now...')
                for example_path in self.data_list:
                    with open(example_path, 'rb') as this_data:
                        _, label = pickle.load(this_data)
                        row_limit, col_limit = label.shape[0]-model_input_size, label.shape[1]-model_input_size
                        label = None  # clear memory
                        _ = None  # clear memory
                        for i in range(0, row_limit, self.stride):
                            for j in range(0, col_limit, self.stride):
                                self.all_images.append((example_path, i, j))
                                self.total_images += 1

                with open(data_map_path, 'wb') as data_map:
                    pickle.dump((self.data_list, self.all_images), file=data_map, protocol=pickle.HIGHEST_PROTOCOL)
                    print('LOG: {} saved!'.format(data_map_path))
            pass

        def __getitem__(self, k):
            k = k % self.total_images ####
            (example_path, this_row, this_col) = self.all_images[k]
            with open(example_path, 'rb') as this_pickle:
                (example_subset, label_subset) = pickle.load(this_pickle)
                example_subset = np.nan_to_num(example_subset)
                label_subset = np.nan_to_num(label_subset)
            this_example_subset = example_subset[this_row:this_row + self.model_input_size,
                                                 this_col:this_col + self.model_input_size, :]
            # instead of using the Digital Numbers (DN), use the backscattering coefficient
            # HH = this_example_subset[:,:,0]
            # HV = this_example_subset[:,:,1]
            # angle = this_example_subset[:,:,2]
            # HH_gamma_naught = np.nan_to_num(10 * np.log10(HH ** 2 + 1e-7) - 83.0)
            # HV_gamma_naught = np.nan_to_num(10 * np.log10(HV ** 2 + 1e-7) - 83.0)
            # this_example_subset = np.dstack((HH_gamma_naught, HV_gamma_naught, angle))

            # get more indices to add to the example, landsat-8
            # ndvi_band = (this_example_subset[:,:,4]-
            #              this_example_subset[:,:,3])/(this_example_subset[:,:,4]+
            #                                           this_example_subset[:,:,3]+1e-7)
            # evi_band = 2.5*(this_example_subset[:,:,4]-
            #                 this_example_subset[:,:,3])/(this_example_subset[:,:,4]+
            #                                              6*this_example_subset[:,:,3]-
            #                                              7.5*this_example_subset[:,:,1]+1)
            # savi_band = 1.5*(this_example_subset[:,:,4]-
            #                  this_example_subset[:,:,3])/(this_example_subset[:,:,4]+
            #                                               this_example_subset[:,:,3]+0.5)
            # msavi_band = 0.5*(2*this_example_subset[:,:,4]+1-
            #                   np.sqrt((2*this_example_subset[:,:,4]+1)**2-
            #                           8*(this_example_subset[:,:,4]-
            #                              this_example_subset[:,:,3])))
            # ndmi_band = (this_example_subset[:,:,4]-
            #              this_example_subset[:,:,5])/(this_example_subset[:,:,4]+
            #                                           this_example_subset[:,:,5]+1e-7)
            # nbr_band = (this_example_subset[:,:,4]-
            #             this_example_subset[:,:,6])/(this_example_subset[:,:,4]+
            #                                          this_example_subset[:,:,6]+1e-7)
            # nbr2_band = (this_example_subset[:,:,5]-
            #              this_example_subset[:,:,6])/(this_example_subset[:,:,5]+
            #                                           this_example_subset[:,:,6]+1e-7)

            # get more indices to add to the example, landsat-8
            ndvi_band = (this_example_subset[:,:,7] -
                         this_example_subset[:,:,3])/(this_example_subset[:,:,7] +
                                                        this_example_subset[:,:,3] + 1e-7)
            evi_band = 2.5 * (this_example_subset[:,:,7] -
                              this_example_subset[:,:,3]) / ((this_example_subset[:,:,7] +
                                                          6.0 * this_example_subset[:,:,3] -
                                                          7.5 * this_example_subset[:,:,1]) + 1.0 + 1e-7)
            savi_band = (this_example_subset[:,:,7] -
                         this_example_subset[:,:,3]) / (this_example_subset[:,:,7] +
                                                    this_example_subset[:,:,3] + 0.428 + 1e-7) * (1.0 + 0.428)
            msavi_band = None
            ndmi_band = (this_example_subset[:,:,7] -
                          this_example_subset[:,:,10]) / (this_example_subset[:,:,7] +
                                                      this_example_subset[:,:,10] + 1e-7)
            nbr_band = (this_example_subset[:,:,7] -
                          this_example_subset[:,:,11]) / (this_example_subset[:,:,7] +
                                                      this_example_subset[:,:,11] + 1e-7)
            nbr2_band = None

            # print(this_example_subset.shape, ndvi_band.shape, evi_band.shape,
            #       savi_band.shape, ndmi_band.shape, nbr_band.shape)

            ndvi_band = np.nan_to_num(ndvi_band)
            evi_band = np.nan_to_num(evi_band)
            savi_band = np.nan_to_num(savi_band)
            # msavi_band = np.nan_to_num(msavi_band)
            ndmi_band = np.nan_to_num(ndmi_band)
            nbr_band = np.nan_to_num(nbr_band)
            # nbr2_band = np.nan_to_num(nbr2_band)

            this_example_subset = np.dstack((this_example_subset, ndvi_band))
            this_example_subset = np.dstack((this_example_subset, evi_band))
            this_example_subset = np.dstack((this_example_subset, savi_band))
            # this_example_subset = np.dstack((this_example_subset, msavi_band))
            this_example_subset = np.dstack((this_example_subset, ndmi_band))
            this_example_subset = np.dstack((this_example_subset, nbr_band))
            # this_example_subset = np.dstack((this_example_subset, nbr2_band))
            # # re-scale
            this_example_subset = this_example_subset / 1000

            this_label_subset = label_subset[this_row:this_row + self.model_input_size,
                                             this_col:this_col + self.model_input_size, ]
            this_label_subset = fix(this_label_subset, total_labels=max_label).astype(np.uint8)
            # if self.mode != 'train':
            #     these_labels, their_frequency = np.unique(this_label_subset, return_counts=True)
            #     print(these_labels, their_frequency)
            # reject all noisy pixels
            # if 0 in these_labels:
            #     self.__getitem__(np.random.randint(self.__len__()))
            # this_label_subset = this_label_subset//2 # convert 1, 2 to 0, 1
            if self.one_hot:
                this_label_subset = np.eye(self.num_classes)[this_label_subset]

            if self.mode == 'train':
                # augmentation
                if np.random.randint(0, 2) == 0:
                    # print('flipped this')
                    this_example_subset = np.fliplr(this_example_subset).copy()
                    this_label_subset = np.fliplr(this_label_subset).copy()
                if np.random.randint(0, 2) == 1:
                    # print('flipped this')
                    this_example_subset = np.flipud(this_example_subset).copy()
                    this_label_subset = np.flipud(this_label_subset).copy()
                if np.random.randint(0, 2) == 1:
                    # print('flipped this')
                    this_example_subset = np.fliplr(this_example_subset).copy()
                    this_label_subset = np.fliplr(this_label_subset).copy()
                if np.random.randint(0, 2) == 0:
                    # print('flipped this')
                    this_example_subset = np.flipud(this_example_subset).copy()
                    this_label_subset = np.flipud(this_label_subset).copy()
                pass

            # print(this_label_subset.shape, this_example_subset.shape)
            this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset,
                                                              one_hot=self.one_hot)
            # if self.transformation:
            #     this_example_subset = self.transformation(this_example_subset)
            return {'input': this_example_subset, 'label': this_label_subset}

        def __len__(self):
            return 1*self.total_images if self.mode == 'train' else self.total_images
    ######################################################################################

    # palsar_mean = torch.Tensor([8116.269912828, 3419.031791692, 40.270058337])
    # palsar_std = torch.Tensor([6136.70160067, 2201.432263753, 19.38761076])
    # palsar_gamma_naught_mean = [-7.68182243, -14.59668144, 40.44296671]
    # palsar_gamma_naught_std = [3.78577892, 4.27134019, 19.73628546]

    # transformation = transforms.Compose([transforms.Normalize(mean=palsar_gamma_naught_mean,
    #                                                           std=palsar_gamma_naught_std)])
    transformation = None

    train_list, eval_list, test_list = [], [], []
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
        print('LOG: No saved data found. Making new data directory {}'.format(save_data_path))
        # full_list = [os.path.join(generated_data_path, x) for x in os.listdir(generated_data_path)]
        # random.shuffle(full_list)
        # train_list = full_list[:int(len(full_list)*train_split)]
        # eval_list = full_list[int(len(full_list)*train_split):]
        # train_years = ["2015"]
        # eval_years = ["2016", "2017"]
        # train_list_per_year = [os.path.join(generated_data_path, year) for year in train_years]
        # eval_list_per_year = [os.path.join(generated_data_path, year) for year in eval_years]
        # train_list = [os.path.join(per_year_train_folder, x) for per_year_train_folder in train_list_per_year
        #               for x in os.listdir(per_year_train_folder)]
        # eval_list = [os.path.join(per_year_eval_folder, x) for per_year_eval_folder in eval_list_per_year
        #              for x in os.listdir(per_year_eval_folder)]

        year = '2016'
        extended_data_path = generated_data_path #os.path.join(generated_data_path, year)
        full_examples_list = [os.path.join(extended_data_path, x) for x in os.listdir(extended_data_path)]
        # print('we are here', len(full_examples_list))
        random.shuffle(full_examples_list)
        # print(len(full_examples_list))
        train_split = int(train_split*len(full_examples_list))
        # print(train_split)
        train_list = full_examples_list[:train_split]
        eval_list = full_examples_list[train_split:]
    ######################################################################################

    print('LOG: [train_list, eval_list, test_list] ->', map(len, (train_list, eval_list, test_list)))
    print('LOG: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))

    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    train_data = dataset(data_list=train_list, data_map_path=os.path.join(save_data_path, 'train_datamap.pkl'),
                         mode='train', stride=32, transformation=transformation) # more images for training
    eval_data = dataset(data_list=eval_list, data_map_path=os.path.join(save_data_path, 'eval_datamap.pkl'),
                        mode='test', stride=model_input_size, transformation=transformation)
    test_data = dataset(data_list=test_list, data_map_path=os.path.join(save_data_path, 'test_datamap.pkl'),
                        mode='test', stride=model_input_size, transformation=transformation)
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


def check_generated_fnf_datapickle(example_path):
    with open(example_path, 'rb') as this_pickle:
        (example_subset, label_subset) = pickle.load(this_pickle)
        example_subset = np.nan_to_num(example_subset)
        label_subset = np.nan_to_num(label_subset)
    # print(example_subset)
    this = histogram_equalize(np.asarray(255*(example_subset[:,:,[3,2,1]]), dtype=np.uint8))
    that = label_subset
    pl.subplot(121)
    pl.imshow(this)
    pl.subplot(122)
    pl.imshow(that)
    pl.show()


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

    loaders = get_dataloaders_generated_data(generated_data_path='/home/azulfiqar_bee15seecs/fnf_dataset/',
                                             save_data_path='pickled_dataset_lists',
                                             model_input_size=128, batch_size=64, train_split=0.8,
                                             one_hot=True, num_workers=4, max_label=16)

    # loaders = get_dataloaders_generated_data(generated_data_path='/home/annuszulfiqar/forest_cover/forestcoverUnet/'
    #                                                              'ESA_landcover/semantic_segmentation/'
    #                                                              'reduced_regions_landsat/dataset',
    #                                          save_data_path='pickled_MODIS_dataset',
    #                                          model_input_size=64, batch_size=128, train_split=0.8,
    #                                          num_workers=10, max_label=22)

    train_dataloader, val_dataloader, test_dataloader = loaders
    while True:
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            print('-> on batch {}/{}, {}'.format(idx+1, len(train_dataloader), examples.size()))
            this_example_subset = (examples[0].numpy()).transpose(1,2,0)
            this = histogram_equalize(np.asarray(255*(this_example_subset[:,:,[3,2,1]]), dtype=np.uint8))
            that = labels[0].numpy().astype(np.uint8)
            ndvi = this_example_subset[:,:,11]
            # print(this.shape, that.shape, np.unique(that))
            that = np.argmax(that, axis=0)
            # print()
            # for j in range(7):
            #     pl.subplot(4,3,j+1)
            #     pl.imshow(this_example_subset[:,:,11+j])
            # pl.show()


if __name__ == '__main__':
    # main()

    # check_generated_fnf_datapickle('/home/annus/Desktop/1_12.pkl')

    get_images_from_large_file(bands=range(1,13),
                               year=sys.argv[1],
                               region=sys.argv[2],
                               stride=256)
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

# It's time to sync this ship!







