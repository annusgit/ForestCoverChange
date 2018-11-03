


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

all_labels = {
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

def toTensor(image):
    "converts a single input image to tensor"
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    # print(image.shape)
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

def get_dataloaders(base_folder, batch_size, one_hot=False):
    print('inside dataloading code...')

    class dataset(Dataset):
        def __init__(self, data_dictionary, bands, mode='train'):
            super(dataset, self).__init__()
            self.example_dictionary = data_dictionary
            # with open(mode+'.txt', 'wb') as this:
            #     this.write(json.dumps(self.example_dictionary))
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
    count_data = 0 # count tells us what to do
    if os.path.exists('train_loader.pkl'):
        count_data += 1
        with open('train_loader.pkl', 'rb') as train_l:
            train_dictionary = p.load(train_l)
            print('INFO: Loaded pre-saved train data...')
    if os.path.exists('val_loader.pkl'):
        count_data += 1
        with open('val_loader.pkl', 'rb') as val_l:
            val_dictionary = p.load(val_l)
            print('INFO: Loaded pre-saved eval data...')
    if os.path.exists('test_loader.pkl'):
        count_data += 1
        with open('test_loader.pkl', 'rb') as test_l:
            test_dictionary = p.load(test_l)
            print('INFO: Loaded pre-saved test data...')

    # create training set examples dictionary
    if count_data != 3:
        all_examples = {}
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
                # dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))
                # print(image)
                image_path = os.path.join(inner_path, image)
                # for each index as key, we want to have its path and label as its items
                class_examples.append(image_path)
            all_examples[folder] = class_examples

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
        with open('train1.txt', 'wb') as train_check:
            for k in range(len(train_dictionary)):
                train_check.write('{}\n'.format(train_dictionary[k][0]))


    print(map(len, [train_dictionary, val_dictionary, test_dictionary]))
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

    # with gzip.open('help.zip', mode='wb') as helpme:
    #     helpme.write(testman)
    # import gzip
    # file = gzip.GzipFile('dum.zip', 'wb')
    # testman = {'data': test_dataloader}
    # file.write(p.dumps(testman, protocol=p.HIGHEST_PROTOCOL))
    # file.close()
    # with open('help.pickle', 'wb') as helpme:
    #     p.dump(testman, file=helpme, protocol=p.HIGHEST_PROTOCOL)


    # save the created datasets
    if count_data != 3:
        with open('train_loader.pkl', 'wb') as train_l:
            p.dump(train_dictionary, train_l, protocol=p.HIGHEST_PROTOCOL)
        with open('test_loader.pkl', 'wb') as test_l:
            p.dump(test_dictionary, test_l, protocol=p.HIGHEST_PROTOCOL)
        with open('val_loader.pkl', 'wb') as val_l:
            p.dump(val_dictionary, val_l, protocol=p.HIGHEST_PROTOCOL)
        print('INFO: saved data pickle files for later use')
    return train_dataloader, val_dataloader, test_dataloader #, test_dictionary


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
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='/home/annus/Desktop/folders/'
                                                                                    'summer@TUKL_Kaiserslautern/'
                                                                                    'projects/forest_cover_change/'
                                                                                    'eurosat/images/tif',
                                                                        batch_size=16)
    # #
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


if __name__ == '__main__':
    check_dataloaders()
    # check_downloaded_images()













