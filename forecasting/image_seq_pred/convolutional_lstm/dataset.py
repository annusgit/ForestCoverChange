


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
np.random.seed(int(time.time()))
random.seed(int(time.time()))
import matplotlib.pyplot as pl
import scipy.misc as misc
from scipy.ndimage import rotate
import scipy.ndimage as ndimage
from torch.utils.data import Dataset, DataLoader


def convert_lat_lon_to_xy(ds, coordinates):
    lon_in, lat_in = coordinates
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    x = int((lat_in-xoffset)/px_w)
    y = int((lon_in-yoffset)/px_h)
    return x, y


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


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


def convert_labels(label_im):
    label_im = np.asarray(label_im/10, dtype=np.uint8)
    return label_im


def fix(target_image, total_labels):
    # target_image[target_image < 0] = -1
    # target_image[target_image > total_labels] = -1
    return target_image


def toTensor(**kwargs):
    image, label = kwargs['image'], kwargs['label']
    'will convert image and label from numpy to torch tensor'
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).float(), torch.from_numpy(label).long()


def get_dataloaders_generated_data(generated_data_path, save_data_path, model_input_size=64, stride=10,
                                   train_split=0.8, batch_size=16, num_workers=4, max_label=16):

    # This function is faster because we have already saved our data as subset pickle files
    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, data_list, data_map_path, mode='train'):
            super(dataset, self).__init__()
            self.examples_list = data_list
            self.model_input_size = model_input_size
            self.all_images = []
            self.total_images = 0
            self.stride = stride
            self.mode = mode
            if os.path.exists(data_map_path):
                print('LOG: Saved data map found! Loading now...')
                with open(data_map_path, 'rb') as data_map:
                    self.total_images, self.all_images = pickle.load(data_map)
            else:
                print('LOG: No data map found! Generating now...')
                self.get_full_data_map()
                with open(data_map_path, 'wb') as data_map:
                    pickle.dump((self.total_images, self.all_images), file=data_map,
                                protocol=pickle.HIGHEST_PROTOCOL)
                    print('LOG: {} saved!'.format(data_map_path))
            pass

        def get_full_data_map(self):

            pass

        def __getitem__(self, k):
            (example_path, this_row, this_col) = self.all_images[k]
            with open(example_path, 'rb') as this_pickle:
                (example_subset, label_subset) = pickle.load(this_pickle)
                example_subset = np.nan_to_num(example_subset)
            this_example_subset = example_subset[
                                  this_row:this_row + self.model_input_size,
                                  this_col:this_col + self.model_input_size, :]
            this_label_subset = label_subset[
                                this_row:this_row + self.model_input_size,
                                this_col:this_col + self.model_input_size,]
            this_label_subset = fix(this_label_subset, total_labels=max_label)

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
            this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset)
            return {'input': this_example_subset, 'label': this_label_subset}

        def __len__(self):
            return self.total_images #if self.mode == 'train' else self.total_images
    ######################################################################################

    # this is the list of paths to all of the training data
    # all_examples = [os.path.join(generated_data_path, x) for x in os.listdir(generated_data_path)]

    if not os.path.exists(save_data_path):
        full_list = [os.path.join(generated_data_path, 'reduced_regions_landsat_2013_1.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_2.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_3.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_4.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_5.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_6.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_7.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_8.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_9.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_10.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_11.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2013_12.pkl'),

                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_1.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_2.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_3.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_4.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_5.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_6.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_7.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_8.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_9.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_10.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_11.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2014_12.pkl'),

                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_1.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_2.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_3.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_4.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_5.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_6.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_7.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_8.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_9.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_10.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_11.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2015_12.pkl'),

                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_1.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_2.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_3.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_4.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_5.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_6.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_7.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_8.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_9.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_10.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_11.pkl'),
                     os.path.join(generated_data_path, 'reduced_regions_landsat_2016_12.pkl'),
                     ]

        train_list = full_list
        eval_list = full_list
        test_list = full_list

        with open(save_data_path, 'wb') as save_pickle:
            pickle.dump((full_list, train_list, eval_list, test_list), file=save_pickle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print('LOG: Data saved!')
    else:
        print('LOG: Found Saved Data! Loading Now...')
        with open(save_data_path, 'rb') as save_pickle:
            full_list, train_list, eval_list, test_list = pickle.load(save_pickle)

    print('LOG: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))

    train_images, eval_images, test_images = [], [], []
    total_train_images, total_eval_images = 0, 0
    for data in full_list:
        with open(data, 'rb') as this_data:
            _, label = pickle.load(this_data)
            row_limit, col_limit = label.shape[0] - model_input_size, label.shape[1] - model_input_size
            # use 1/4th of all images for testing
            for i in range(0, row_limit, stride):
                for j in range(0, col_limit, stride):
                    if i > 3*row_limit//4 and j > 3*col_limit//4:
                        eval_images.append((data, i, j))
                        total_eval_images += 1
                    else:
                        train_images.append((data, i, j))
                        total_train_images += 1
    ######################################################################################

    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    train_data = dataset(data_list=train_list, data_map_path='train_datamap.pkl', mode='train')
    eval_data = dataset(data_list=eval_list, data_map_path='eval_datamap.pkl', mode='test')
    test_data = dataset(data_list=test_list, data_map_path='test_datamap.pkl', mode='test')
    train_data.all_images = train_images
    train_data.total_images = total_train_images
    eval_data.all_images = eval_images
    eval_data.total_images = total_eval_images
    print('LOG: [train_data, eval_data, test_data] ->', len(train_data), len(eval_data), len(test_data))

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def main():
    loaders = get_dataloaders_generated_data(generated_data_path='/home/annus/PycharmProjects/'
                                                                 'ForestCoverChange_inputs_and_numerical_results/'
                                                                 'reduced_landsat_images/'
                                                                 'reduced_dataset_for_segmentation_MODIS/',
                                             save_data_path='pickled_data_check.pkl',
                                             model_input_size=64, batch_size=64, train_split=0.8,
                                             num_workers=4, max_label=22)
    #
    # loaders = get_dataloaders_generated_data(generated_data_path='/home/annuszulfiqar/forest_cover/forestcoverUnet/'
    #                                                              'ESA_landcover/reduced_regions_landsat/dataset',
    #                                          save_data_path='pickled_MODIS_dataset.pkl',
    #                                          model_input_size=64, batch_size=64, train_split=0.8,
    #                                          num_workers=6, max_label=22)

    train_dataloader, val_dataloader, test_dataloader = loaders
    for idx, data in enumerate(train_dataloader):
        examples, labels = data['input'], data['label']
        print('-> on batch {}/{}, {}'.format(idx+1, len(train_dataloader), examples.size()))
        this_example_subset = (examples[0].numpy()).transpose(1,2,0)
        this = histogram_equalize(np.asarray(255*(this_example_subset[:,:,[3,2,1]]), dtype=np.uint8))
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









