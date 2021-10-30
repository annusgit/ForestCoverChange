


from __future__ import print_function
from __future__ import division
import os
import sys
# import cv2
import gdal
# import json
import torch
import random
import pickle as p
import numpy as np
random.seed(74)
import matplotlib.pyplot as pl
from torch.utils.data import Dataset, DataLoader
import imgaug as ia
from imgaug import augmenters as iaa


# will implement all functionality (data augmentation) of doing
# 1. random crops,
# 2. random flips,
# 3. random rotations,

German_labels = {
                'AnnualCrop'           : 0,
                'Forest'               : 1,
                'HerbaceousVegetation' : 2,
                'Highway'              : 3,
                'Industrial'           : 4,
                'Pasture'              : 5,
                'PermanentCrop'        : 6,
                'Residential'          : 7,
                'River'                : 8,
                'SeaLake'              : 9
                }

Pakistani_labels = {
                    'forest': 0,
                    'residential': 1,
                    'pastures': 2,
                    'vegetation': 3,
                    'plainland': 4,
                    'snow': 5
                   }

all_labels = Pakistani_labels

def toTensor(image):
    "converts a single input image to tensor"
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    # print(image.shape)a
    return torch.from_numpy(image).float()

######################################################################################################
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 50% of all images

        # crop some of the images by 0-20% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.2))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate=(-180, 175),
            mode=ia.ALL
        )),
    ],
    # do all of the above augmentations in random order
    random_order=True
)
######################################################################################################

def get_dataloaders(base_folder, batch_size, one_hot=False, custom_size=False):
    print('inside dataloading code...')

    class dataset(Dataset):
        def __init__(self, data_dictionary, bands, mode='train'):
            super(dataset, self).__init__()
            self.example_dictionary = data_dictionary
            self.custom_size = custom_size # incase the images are not all the same size
            self.bands = bands # bands are a list bands to use as data, pass them as a list []
            self.mode = mode
            self.max = 0
            pass

        def __getitem__(self, k):
            example_path, label_name = self.example_dictionary[k]
            # print(example_path, label_name)
            # example is a tiff image, need to use gdal
            this_example = gdal.Open(example_path)
            this_label = all_labels[label_name]
            if one_hot:
                label_arr = np.zeros(10)
                label_arr[this_label] = 1
            # print(this_label, label_arr)
            example_array = this_example.GetRasterBand(self.bands[0]).ReadAsArray()
            for i in self.bands[1:]:
                example_array = np.dstack((example_array,
                                           this_example.GetRasterBand(i).ReadAsArray())).astype(np.int16)
            '''
                This is a very bad approach because we have no idea how 
                many images are going to training, testing and validation!
            '''
            # if size if not 64*64, randomly crop 64*64 patch from it
            crop_size = 64
            # if example_array.shape[0] < crop_size or example_array.shape[1] < crop_size:
            #     return self.__getitem__(np.random.randint(self.__len__()))
            x = random.randint(0, example_array.shape[0] - crop_size)
            y = random.randint(0, example_array.shape[1] - crop_size)
            example_array = example_array[x:x + crop_size, y:y + crop_size, :]

            # transforms for augmentation
            if self.mode == 'train':
                example_array = np.squeeze(seq.augment_images(
                    (np.expand_dims(example_array, axis=0))), axis=0)
                pass

            # range of vals = [0,1]
            example_array = np.clip((example_array.astype(np.float)/4096), a_min=0, a_max=1)
            # range of vals = [-1,1]
            example_array = 2*example_array-1

            # max value in test set is 28000
            # this_max = example_array.max()
            # if this_max > self.max:
            #     self.max = this_max
            # print(example_array.max(), example_array.min(), example_array.mean())

            example_array = toTensor(image=example_array)
            if one_hot:
                return {'input': example_array, 'label': torch.LongTensor(label_arr)}
            return {'input': example_array, 'label': this_label}


        def __len__(self):
            return len(self.example_dictionary)


    """
        Okay so here is how we do it. We save the train, test and validation dictionaries if they don't exist, 
        and once they do, we load the preexisting ones to help us!
    """
    # check if we already have the data saved with us...
    cached_datapath = 'cached_dataset'
    if not os.path.exists(cached_datapath):
        os.mkdir(cached_datapath)
    cached_train_datapath = os.path.join(cached_datapath, 'train_loader.pkl')
    cached_val_datapath = os.path.join(cached_datapath, 'val_loader.pkl')
    cached_test_datapath = os.path.join(cached_datapath, 'test_loader.pkl')
    count_data = 0 # count tells us what to do
    if os.path.exists(cached_train_datapath):
        count_data += 1
        with open(cached_train_datapath, 'rb') as train_l:
            train_dictionary = p.load(train_l)
            print('INFO: Loaded pre-saved train data...')
    if os.path.exists(cached_val_datapath):
        count_data += 1
        with open(cached_val_datapath, 'rb') as val_l:
            val_dictionary = p.load(val_l)
            print('INFO: Loaded pre-saved eval data...')
    if os.path.exists(cached_test_datapath):
        count_data += 1
        with open(cached_test_datapath, 'rb') as test_l:
            test_dictionary = p.load(test_l)
            print('INFO: Loaded pre-saved test data...')

    # create training set examples dictionary
    if count_data != 3:
        all_examples = {}
        total, selected = 0, 0
        for folder in sorted(os.listdir(base_folder)):
            # each folder name is a label itself
            # new folder, new dictionary!
            class_examples = []
            inner_path = os.path.join(base_folder, folder)
            #####################################3 this was a problem for a long time now.. because of not sorting it
            all_images_of_current_class = [x for x in os.listdir(inner_path) if x.endswith('.tif')]
            all_images_of_current_class.sort(key=lambda f: int(filter(str.isdigit, f)))
            # if folder == 'Forest':
            #     print(all_images_of_current_class)
            for image in all_images_of_current_class:
                total += 1
                # dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))
                # print(image)
                image_path = os.path.join(inner_path, image)
                # at this point, we shall reject the images smaller than 64*64
                min_dimension = 64
                test_image = gdal.Open(image_path)
                shape = test_image.GetRasterBand(1).ReadAsArray().shape
                if shape[0] >= min_dimension and shape[1] >= min_dimension:
                    # for each index as key, we want to have its path and label as its items
                    class_examples.append(image_path)
                    selected += 1
            all_examples[folder] = class_examples

        print(selected, total)
        # split them into train and test
        train_dictionary, val_dictionary, test_dictionary = {}, {}, {}
        for class_name in all_examples.keys():
            class_examples = all_examples[class_name]
            # print(class_examples)
            ########################## this doesn't work
            # random.shuffle(class_examples)
            ########################### but this does
            random.Random(4).shuffle(class_examples)

            total = len(class_examples)
            train_count = int(total * 0.8); train_ = class_examples[:train_count]
            test = class_examples[train_count:]

            total = len(train_)
            train_count = int(total * 0.9); train = train_[:train_count]
            validation = train_[train_count:]

            for example in train:
                train_dictionary[len(train_dictionary)] = (example, class_name)
            for example in test:
                test_dictionary[len(test_dictionary)] = (example, class_name)
            for example in validation:
                val_dictionary[len(val_dictionary)] = (example, class_name)

        # # test dataset
        with open(os.path.join(cached_datapath, 'train1.txt'), 'wb') as train_check:
            for k in range(len(train_dictionary)):
                train_check.write('{}\n'.format(train_dictionary[k][0]))


    print('[training, evaluation, testing] -> ', map(len, [train_dictionary, val_dictionary, test_dictionary]))
    # create dataset class instances
    bands = [4, 3, 2, 5, 8] # these are [Red, Green, Blue, NIR, Vegetation Red Edge] bands
    # bands = [4, 3, 2] # these are [Red, Green, Blue] bands only
    train_data = dataset(data_dictionary=train_dictionary, bands=bands, mode='train')
    val_data = dataset(data_dictionary=val_dictionary, bands=bands, mode='eval')
    test_data = dataset(data_dictionary=test_dictionary, bands=bands, mode='test')
    print('train examples =', len(train_dictionary), 'val examples =', len(val_dictionary),
          'test examples =', len(test_dictionary))

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=True, num_workers=4)

    # save the created datasets
    if count_data != 3:
        with open(cached_train_datapath, 'wb') as train_l:
            p.dump(train_dictionary, train_l, protocol=p.HIGHEST_PROTOCOL)
        with open(cached_test_datapath, 'wb') as test_l:
            p.dump(test_dictionary, test_l, protocol=p.HIGHEST_PROTOCOL)
        with open(cached_val_datapath, 'wb') as val_l:
            p.dump(val_dictionary, val_l, protocol=p.HIGHEST_PROTOCOL)
        print('INFO: saved data pickle files for later use')
    return train_dataloader, val_dataloader, test_dataloader #, test_dictionary


def get_images_from_large_file(image_path, bands, label_path, destination, region, stride):
    image_path = image_path + str(region) + '.tif'
    label_path = label_path + str(region) + '.tif'
    print(image_path, label_path)
    # we will use this to divide those fnf images
    covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
    channel = covermap.GetRasterBand(1)
    x_size, y_size = covermap.RasterXSize, covermap.RasterYSize
    # min_x, min_y = 0, 0
    # max_x, max_y = x_size, y_size
    # read the corresponding label at 360m per pixel resolution
    label = channel.ReadAsArray()
    # let's get the actual image now
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
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
            labels = np.unique(label_subset)
            if 0 in labels:
                # skip the noise class
                continue

            # save this example/label pair of numpy arrays as a pickle file with an index
            this_example_save_path = os.path.join(destination, '{}_{}.pkl'.format(region, count))
            with open(this_example_save_path, 'wb') as this_pickle:
                p.dump((example_subset, label_subset), file=this_pickle, protocol=p.HIGHEST_PROTOCOL)
                print('log: Saved {} '.format(this_example_save_path), end='')
                print(i*stride, (i+1)*stride, j*stride, (j+1)*stride)
            pass
    pass


def fix(target_image):
    target_image[target_image > 2] = 2
    return target_image


def get_dataloaders_generated_data(generated_data_path, save_data_path, model_input_size=64, num_classes=16,
                                   one_hot=False, batch_size=16, num_workers=4):
    '''
        We are using this function to convert our semantic labels into patches
        :param generated_data_path:
        :param save_data_path:
        :param model_input_size:
        :param num_classes:
        :param train_split:
        :param one_hot:
        :param batch_size:
        :param num_workers:
        :param max_label:
        :return:
    '''

    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, data_list, data_map_path, stride, mode='train'):
            super(dataset, self).__init__()
            self.model_input_size = model_input_size
            self.data_list = data_list
            self.all_images = []
            self.total_images = 0
            self.stride = stride
            self.one_hot = one_hot
            self.num_classes = num_classes
            self.mode = mode
            if os.path.exists(data_map_path):
                print('LOG: Saved data map found! Loading now...')
                with open(data_map_path, 'rb') as data_map:
                    self.data_list, self.all_images = p.load(data_map)
                    self.total_images = len(self.all_images)
            else:
                print('LOG: No data map found! Generating now...')
                for example_path in self.data_list:
                    print('\t->creating map for {}'.format(example_path))
                    with open(example_path, 'rb') as this_data:
                        _, label = p.load(this_data)
                        row_limit, col_limit = label.shape[0]-model_input_size, label.shape[1]-model_input_size
                        for i in range(0, row_limit, self.stride):
                            for j in range(0, col_limit, self.stride):
                                # convert water to non-forest class 3->2
                                this_label_subset = fix(np.nan_to_num(label)).astype(np.uint8)
                                these_labels, their_frequency = np.unique(this_label_subset, return_counts=True)
                                this_patch_label = these_labels[np.argmax(their_frequency)]
                                # reject the examples having noise
                                if this_patch_label != 0:
                                    self.all_images.append((example_path, i, j))
                                    self.total_images += 1

                with open(data_map_path, 'wb') as data_map:
                    p.dump((self.data_list, self.all_images), file=data_map, protocol=p.HIGHEST_PROTOCOL)
                    print('LOG: {} saved!'.format(data_map_path))
            pass

        def __getitem__(self, k):
            (example_path, this_row, this_col) = self.all_images[k]
            with open(example_path, 'rb') as this_pickle:
                (example_subset, label_subset) = p.load(this_pickle)
                example_subset = np.nan_to_num(example_subset)
                label_subset = np.nan_to_num(label_subset)
            this_example_subset = example_subset[
                                  this_row:this_row + self.model_input_size,
                                  this_col:this_col + self.model_input_size, :]
            # get more indices to add to the example
            ndvi_band = (this_example_subset[:,:,4]-
                         this_example_subset[:,:,3])/(this_example_subset[:,:,4]+
                                                      this_example_subset[:,:,3]+1e-7)
            evi_band = 2.5*(this_example_subset[:,:,4]-
                            this_example_subset[:,:,3])/(this_example_subset[:,:,4]+
                                                         6*this_example_subset[:,:,3]-
                                                         7.5*this_example_subset[:,:,1]+1)
            savi_band = 1.5*(this_example_subset[:,:,4]-
                             this_example_subset[:,:,3])/(this_example_subset[:,:,4]+
                                                          this_example_subset[:,:,3]+0.5)
            msavi_band = 0.5*(2*this_example_subset[:,:,4]+1-
                              np.sqrt((2*this_example_subset[:,:,4]+1)**2-
                                      8*(this_example_subset[:,:,4]-
                                         this_example_subset[:,:,3])))
            ndmi_band = (this_example_subset[:,:,4]-
                         this_example_subset[:,:,5])/(this_example_subset[:,:,4]+
                                                      this_example_subset[:,:,5]+1e-7)
            nbr_band = (this_example_subset[:,:,4]-
                        this_example_subset[:,:,6])/(this_example_subset[:,:,4]+
                                                     this_example_subset[:,:,6]+1e-7)
            nbr2_band = (this_example_subset[:,:,5]-
                         this_example_subset[:,:,6])/(this_example_subset[:,:,5]+
                                                      this_example_subset[:,:,6]+1e-7)

            ndvi_band = np.nan_to_num(ndvi_band)
            evi_band = np.nan_to_num(evi_band)
            savi_band = np.nan_to_num(savi_band)
            msavi_band = np.nan_to_num(msavi_band)
            ndmi_band = np.nan_to_num(ndmi_band)
            nbr_band = np.nan_to_num(nbr_band)
            nbr2_band = np.nan_to_num(nbr2_band)

            this_example_subset = np.dstack((this_example_subset, ndvi_band))
            this_example_subset = np.dstack((this_example_subset, evi_band))
            this_example_subset = np.dstack((this_example_subset, savi_band))
            this_example_subset = np.dstack((this_example_subset, msavi_band))
            this_example_subset = np.dstack((this_example_subset, ndmi_band))
            this_example_subset = np.dstack((this_example_subset, nbr_band))
            this_example_subset = np.dstack((this_example_subset, nbr2_band))

            this_label_subset = label_subset[this_row:this_row + self.model_input_size,
                                             this_col:this_col + self.model_input_size,]
            # fix to convert the water label to non-forest label
            this_label_subset = fix(this_label_subset).astype(np.uint8)
            these_labels, their_frequency = np.unique(this_label_subset, return_counts=True)
            # the classes we have are 1 and 2, convert them to 1 and 0
            this_patch_label = these_labels[np.argmax(their_frequency)]-1
            if this_patch_label < 0:
                return self.__getitem__(np.random.randint(self.__len__()))

            if self.one_hot:
                this_patch_label = np.eye(self.num_classes)[this_patch_label]

            if self.mode == 'train':
                # augmentation
                if np.random.randint(0, 2) == 0:
                    this_example_subset = np.fliplr(this_example_subset).copy()
                if np.random.randint(0, 2) == 1:
                    this_example_subset = np.flipud(this_example_subset).copy()
                if np.random.randint(0, 2) == 1:
                    this_example_subset = np.fliplr(this_example_subset).copy()
                if np.random.randint(0, 2) == 0:
                    this_example_subset = np.flipud(this_example_subset).copy()
                pass

            this_example_subset = toTensor(image=this_example_subset)
            return {'input': this_example_subset, 'label': this_patch_label}

        def __len__(self):
            return 1*self.total_images if self.mode == 'train' else self.total_images
    ######################################################################################

    train_list, eval_list, test_list = [], [], []
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
        print('LOG: No saved data found. Making new data directory {}'.format(save_data_path))
        full_list = [os.path.join(generated_data_path, x) for x in os.listdir(generated_data_path)]
        random.shuffle(full_list)
        train_list = full_list[:int(len(full_list)*0.8)]
        eval_list = full_list[int(len(full_list)*0.8):]

    print('LOG: [train_list, eval_list, test_list] ->', map(len, (train_list, eval_list, test_list)))
    print('LOG: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))

    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    train_data = dataset(data_list=train_list, data_map_path=os.path.join(save_data_path, 'train_datamap.pkl'),
                         mode='train', stride=model_input_size) # more images for training
    eval_data = dataset(data_list=eval_list, data_map_path=os.path.join(save_data_path, 'eval_datamap.pkl'),
                        mode='test', stride=model_input_size)
    test_data = dataset(data_list=test_list, data_map_path=os.path.join(save_data_path, 'test_datamap.pkl'),
                        mode='test', stride=model_input_size)
    print('LOG: [train_data, eval_data, test_data] ->', len(train_data), len(eval_data), len(test_data))


    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


# We shall use this at inference time on our custom downloaded images...
def get_inference_loader(image_path, batch_size):
    # print('inside dataloading code...')

    class dataset(Dataset):
        def __init__(self, image_arr, index_dictionary):
            super(dataset, self).__init__()
            self.image_arr = image_arr
            self.index_dictionary = index_dictionary
            pass

        def __getitem__(self, k):
            x, x_, y, y_ = self.index_dictionary[k]
            example_array = self.image_arr[x:x_, y:y_, :]
            # this division is non-sense, but let's do it anyway...
            # print(example_array.max())
            # pl.imshow(example_array)
            # pl.show()

            # range of vals = [0,1]
            example_array = np.clip((example_array.astype(np.float)/4096), a_min=0, a_max=1)
            # range of vals = [-1,1]
            example_array = 2*example_array-1

            # example_array = (example_array.astype(np.float)/4096)
            # ex_array = []
            # for t in range(3, -1, -1):
            #     temp = np.expand_dims(example_array[:,:,t], 2)
            #     ex_array.append(temp)
            # example_array = np.dstack(ex_array)
            # print(example_array.shape)
            # print(example_array.shape)
            # example_array = np.dstack((example_array[:,:,2],example_array[:,:,1],example_array[:,:,0]))
            example_array = toTensor(image=example_array)
            return {'input': example_array, 'indices': torch.Tensor([x, x_, y, y_]).long()}

        def __len__(self):
            return len(self.index_dictionary)

    # create training set examples dictionary
    patch = 64 # this is fixed and default
    image_file = np.load(image_path, mmap_mode='r') # we don't want to load it into memory because it's huge
    image_read = image_file['pixels']
    # print(image_read.max())
    H, W = image_read.shape[0], image_read.shape[1]
    x_num = W // patch
    y_num = H //patch
    image_read = image_read[:y_num*patch, :x_num*patch, :]

    # get a dictionary of all possible indices to crop out of the actual tile image
    index_dict = {}
    for i in range(y_num):
        for j in range(x_num):
            index_dict[len(index_dict)] = (patch*i, patch*i+patch, j*patch, j*patch+patch)

    data = dataset(image_arr=image_read, index_dictionary=index_dict)
    print('number of test examples =', len(index_dict))

    train_dataloader = DataLoader(dataset=data, batch_size=batch_size,
                                  shuffle=False, num_workers=4)
    print(image_read.shape)
    return train_dataloader, image_read.shape


def histogram_equalization(in_image):
    for i in range(in_image.shape[2]): # each channel
        image = in_image[:,:,i]
        prev_shape = image.shape
        # Flatten the image into 1 dimension: pixels
        pixels = image.flatten()

        # Generate a cumulative histogram
        cdf, bins, patches = pl.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
        new_pixels = np.interp(pixels, bins[:-1], cdf*255)
        in_image[:,:,i] = new_pixels.reshape(prev_shape)
    return in_image


def check_dataloaders():
    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='/home/annus/Desktop/folders/'
    #                                                                                 'summer@TUKL_Kaiserslautern/'
    #                                                                                 'projects/forest_cover_change/'
    #                                                                                 'eurosat/images/tif',
    #                                                                     batch_size=16)

    # # Pakistani data
    train_dataloader, val_dataloader, \
    test_dataloader = get_dataloaders(base_folder='/home/annus/PycharmProjects/'
                                                  'ForestCoverChange_inputs_and_numerical_results/'
                                                  'patch_wise/Pakistani_data/'
                                                  'full_pakistan_data_unpacked/',
                                      batch_size=16)

    # train_dataloader, val_dataloader, \
    # test_dataloader = get_dataloaders(base_folder='Pakistani_Forest_Data/full_pakistan_data/',
    #                                   batch_size=16)

    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='Eurosat/tif/',
    #                                                                     batch_size=16)

    count = 0
    reversed_labels = {v:k for k, v in all_labels.iteritems()}
    while True:
        count += 1
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            print('{} -> on batch {}/{}, {}'.format(count, idx+1, len(train_dataloader), examples.size()))
            if True:
                # this = np.max(examples[0].numpy())
                # print(this)
                this = ((examples[0].numpy()+1)/2*255).transpose(1,2,0)[:,:,:3].astype(np.uint8)
                print(examples.shape, labels.shape)
                # this = histogram_equalization(this)
                pl.imshow(this)
                pl.title('{}'.format(reversed_labels[int(labels[0].numpy())]))
                pl.show()



def check_inference_loader():
    this_path = '/home/annus/Desktop/forest_images/test_images/muzaffarabad_pickle.pkl'
    inference_loader, _ = get_inference_loader(image_path=this_path, batch_size=4)
    count = 0
    while True:
        count += 1
        for idx, data in enumerate(inference_loader):
            examples, indices = data['input'], data['indices']
            print('{} -> on batch {}/{}, {}'.format(count, idx + 1, len(inference_loader), examples.size()))
            if True:
                this = np.max(examples[0].numpy())
                indices = indices.numpy()
                print(indices[:,0], indices[:,1], indices[:,2], indices[:,3])
                this = (examples[0].numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                # this = histogram_equalization(this)
                pl.imshow(this)
                pl.show()
    pass


def check_data_sanity():
    train, val, _, test1 = get_dataloaders(base_folder='/home/annus/Desktop/'
                                                      'projects/forest_cover_change/'
                                                      'eurosat/images/tif/',
                                          batch_size=16)

    train, val, _, test2 = get_dataloaders(base_folder='/home/annus/Desktop/'
                                                      'projects/forest_cover_change/'
                                                      'eurosat/images/tif/',
                                          batch_size=16)


    train, val, _, test3 = get_dataloaders(base_folder='/home/annus/Desktop/'
                                                      'projects/forest_cover_change/'
                                                      'eurosat/images/tif/',
                                          batch_size=16)

    def get_dict_diff(d1, d2):
        return len(set(d1.values()) - set(d2.values()))

    # compare on the same run
    print(get_dict_diff(test1, test2))
    print(get_dict_diff(test2, test3))
    print(get_dict_diff(test3, test1))

    # compare across runs
    if os.path.exists('test1.pkl'):
        with open('test1.pkl', 'rb') as ts1:
            test1_old = p.load(ts1)
        print('during cross runs, diff =', get_dict_diff(test2, test1_old))
    else:
        with open('test1.pkl', 'wb') as ts1:
            p.dump(test1, ts1, protocol=p.HIGHEST_PROTOCOL)


def check_downloaded_images():
    example_path = sys.argv[1]
    this_example = gdal.Open(example_path)
    total = this_example.RasterCount
    bands = [4, 3, 2]
    # bands = [4, 3, 2]
    print(this_example.RasterCount)
    example_array = this_example.GetRasterBand(bands[0]).ReadAsArray()
    for i in bands[1:]:
        example_array = np.dstack((example_array,
                                   this_example.GetRasterBand(i).ReadAsArray())).astype(np.int16)
    print(example_array.max())
    show_image = (example_array/4096*255).astype(np.uint8)
    pl.imshow(show_image)
    pl.show()
    pass


def check_semantic_to_patch_loaders():
    loaders = get_dataloaders_generated_data(generated_data_path='/home/annuszulfiqar/fnf_dataset/',
                                             save_data_path='pickled_32_size_saves',
                                             model_input_size=8, batch_size=128, one_hot=True,
                                             num_classes=2, num_workers=4)
    train_dataloader, val_dataloader, test_dataloader = loaders
    while True:
        for idx, data in enumerate(train_dataloader):
            examples, labels = data['input'], data['label']
            print('-> on batch {}/{}, {}'.format(idx + 1, len(train_dataloader), examples.size()))
            print(examples.shape, labels.shape)
    pass


if __name__ == '__main__':
    # check_dataloaders()
    # check_downloaded_images()

    # get_images_from_large_file(image_path='/home/annuszulfiqar/regions_25m/region_landsat_2017_',
    #                            bands=range(1,12),
    #                            label_path='/home/annuszulfiqar/fnf/fnf_2017_region_',
    #                            destination='/home/annuszulfiqar/fnf_smaller_patches_dataset/',
    #                            region=int(sys.argv[1]),
    #                            stride=32)

    check_semantic_to_patch_loaders()










