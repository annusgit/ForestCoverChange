


from __future__ import print_function
from __future__ import division
import os
# import cv2
import gdal
# import json
import torch
import random
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
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-180, 175),
            mode=ia.ALL
        )),
    ],
    # do all of the above augmentations in random order
    random_order=True
)
######################################################################################################

def get_dataloaders(base_folder, batch_size):
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
            example_array = this_example.GetRasterBand(self.bands[0]).ReadAsArray()
            for i in self.bands[1:]:
                example_array = np.dstack((example_array,
                                           this_example.GetRasterBand(i).ReadAsArray())).astype(np.int16)

            # transforms
            if self.mode == 'train':
                example_array = np.squeeze(seq.augment_images(
                    (np.expand_dims(example_array, axis=0))), axis=0)
                pass

            # range of vals = [0,1]
            example_array = (example_array.astype(np.float)/4096)

            # max value in test set is 28000
            # this_max = example_array.max()
            # if this_max > self.max:
            #     self.max = this_max
            # print(example_array.max(), example_array.min(), example_array.mean())

            example_array = toTensor(image=example_array)
            return {'input': example_array, 'label': this_label}

        def __len__(self):
            return len(self.example_dictionary)

    # create training set examples dictionary
    all_examples = {}
    for folder in sorted(os.listdir(base_folder)):
        # each folder name is a label itself
        # new folder, new dictionary!
        class_examples = []
        inner_path = os.path.join(base_folder, folder)
        for image in [x for x in os.listdir(inner_path) if x.endswith('.tif')]:
            image_path = os.path.join(inner_path, image)
            # for each index as key, we want to have its path and label as its items
            class_examples.append(image_path)
        all_examples[folder] = class_examples

    # split them into train and test
    train_dictionary, val_dictionary, test_dictionary = {}, {}, {}
    for class_name in all_examples.keys():
        class_examples = all_examples[class_name]
        # print(class_examples)
        random.shuffle(class_examples)

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


    # create dataset class instances
    bands = [4, 3, 2]
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

    return train_dataloader, val_dataloader, test_dataloader


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


def main():
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(base_folder='/home/annus/Desktop/'
                                                                                    'forest_cover_change/'
                                                                                    'eurosat/images/tif',
                                                                        batch_size=1)
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
                this = np.max(examples[0].numpy())
                print(this)
                this = (examples[0].numpy()*255).transpose(1,2,0).astype(np.uint8)
                # this = histogram_equalization(this)
                pl.imshow(this)
                pl.title('{}'.format(reversed_labels[int(labels.numpy())]))
                pl.show()


if __name__ == '__main__':
    main()














