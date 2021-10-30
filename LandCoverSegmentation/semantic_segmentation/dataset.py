from __future__ import print_function
from __future__ import division
import os
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
# from torchvision import transforms


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


def mask_landsat8_image_using_rasterized_shapefile(rasterized_shapefiles_path, district, this_landsat8_bands_list):
    this_shapefile_path = os.path.join(rasterized_shapefiles_path, "{}_shapefile.tif".format(district))
    ds = gdal.Open(this_shapefile_path)
    assert ds.RasterCount == 1
    shapefile_mask = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
    clipped_full_spectrum = list()
    for idx, this_band in enumerate(this_landsat8_bands_list):
        print("{}: Band-{} Size: {}".format(district, idx, this_band.shape))
        clipped_full_spectrum.append(np.multiply(this_band, shapefile_mask))
    x_prev, y_prev = clipped_full_spectrum[0].shape
    x_fixed, y_fixed = int(128 * np.ceil(x_prev / 128)), int(128 * np.ceil(y_prev / 128))
    diff_x, diff_y = x_fixed - x_prev, y_fixed - y_prev
    diff_x_before, diff_y_before = diff_x // 2, diff_y // 2
    clipped_full_spectrum_resized = [np.pad(x, [(diff_x_before, diff_x - diff_x_before), (diff_y_before, diff_y - diff_y_before)], mode='constant')
                                     for x in clipped_full_spectrum]
    print("{}: Generated Image Size: {}".format(district, clipped_full_spectrum_resized[0].shape, len(clipped_full_spectrum_resized)))
    return clipped_full_spectrum_resized


def get_images_from_large_file(bands, year, region, stride):
    # local machine
    data_directory_path = '/mnt/e/Forest Cover - Redo 2020/Google Cloud - Training/Training Data/Clipped dataset/Images_and_GroundTruth'
    destination = '/mnt/e/Forest Cover - Redo 2020/Google Cloud - Training/Training Data/Clipped dataset/Pickled_data/'

    # # cloud machine
    # data_directory_path = '/home/azulfiqar_bee15seecs/training_data/clipped_training_data/'
    # destination = '/home/azulfiqar_bee15seecs/training_data/training_2015_pickled_data/'

    # # # tukl cluster
    # data_directory_path = '/work/mohsin/BTT_districts_maps/training_2015/'
    # destination = '/work/mohsin/BTT_districts_maps/training_2015_pickled_data/'

    image_path = os.path.join(data_directory_path, '{}_image.tif'.format(region))
    label_path = os.path.join(data_directory_path, '{}_{}.tif'.format(region, year))
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
    count = 1
    for i in range(y_size//stride):
        for j in range(x_size//stride):
            # read the label and drop this sample if it has all null pixels
            label_subset = label[i*stride:(i+1)*stride, j*stride:(j+1)*stride]
            if np.count_nonzero(label_subset) < 600:  # 0.01*256*256 ~ 650 pixels i.e at least 1% pixels should be valid
                print("(LOG): Dropping NULL Pixel Sample")
                continue
            # read the raster band by band for this subset
            example_subset = np.nan_to_num(all_raster_bands[0].ReadAsArray(j*stride, i*stride, stride, stride))
            for band in all_raster_bands[1:]:
                example_subset = np.dstack((example_subset, np.nan_to_num(band.ReadAsArray(j*stride, i*stride, stride, stride))))
            # save this example/label pair of numpy arrays as a pickle file with an index
            this_example_save_path = os.path.join(destination, '{}_{}_{}.pkl'.format(region, year, count))
            with open(this_example_save_path, 'wb') as this_pickle:
                pickle.dump((example_subset, label_subset), file=this_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                print('log: Saved {} '.format(this_example_save_path))
                print(i*stride, (i+1)*stride, j*stride, (j+1)*stride)
            count += 1
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


def fix(target_image):
    # we fix the label by
    # 1. Converting all NULL (0) pixels to Non-forest pixels (1)
    target_image[target_image == 0] = 1  # this will convert all null pixels to non-forest pixels
    # 2. Subtracting 1 from all labels => Non-forest = 0, Forest = 1
    target_image -= 1
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


def get_dataloaders_generated_data(generated_data_path, data_split_lists_path, model_input_size, bands, num_classes, train_split, one_hot, batch_size,
                                   num_workers):
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
            # self.bands = [x-1 for x in bands]
            self.num_classes = num_classes
            self.transformation = transformation
            self.mode = mode
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
                        if np.count_nonzero(label) < 160:
                            # we skip a label with too few valid pixels (<0.01*128*128)
                            continue
                        # label = np.array(label)
                        # print(label.shape)
                        row_limit, col_limit = label.shape[0]-model_input_size, label.shape[1]-model_input_size
                        label = None  # clear memory
                        _ = None  # clear memory
                        for i in range(0, row_limit, self.stride):
                            for j in range(0, col_limit, self.stride):
                                self.all_images.append((example_path, i, j))
                                self.total_images += 1
                with open(data_map_path, 'wb') as data_map:
                    pickle.dump((self.data_list, self.all_images), file=data_map)  # , protocol=pickle.HIGHEST_PROTOCOL)
                    print('LOG: {} saved!'.format(data_map_path))
            pass

        def __getitem__(self, k):
            k = k % self.total_images
            (example_path, this_row, this_col) = self.all_images[k]
            # fix example path here
            # print("Fixing datapath")
            # example_path = os.path.join("/mnt/e/Forest Cover - Redo 2020/Trainings and Results/Training Data/Clipped dataset/Pickled_data",
            #                             os.path.basename(os.path.normpath(example_path)))
            with open(example_path, 'rb') as this_pickle:
                (example_subset, label_subset) = pickle.load(this_pickle)
                example_subset = np.nan_to_num(example_subset)
                label_subset = np.nan_to_num(label_subset)
            this_example_subset = example_subset[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size, :]
            # get more indices to add to the example, landsat-8
            ndvi_band = (this_example_subset[:,:,4]-this_example_subset[:,:,3])/(this_example_subset[:,:,4]+this_example_subset[:,:,3]+1e-7)
            evi_band = 2.5*(this_example_subset[:,:,4]-this_example_subset[:,:,3])/(this_example_subset[:,:,4]+6*this_example_subset[:,:,3]-7.5*this_example_subset[:,:,1]+1)
            savi_band = 1.5*(this_example_subset[:,:,4]-this_example_subset[:,:,3])/(this_example_subset[:,:,4]+this_example_subset[:,:,3]+0.5)
            msavi_band = 0.5*(2*this_example_subset[:,:,4]+1-np.sqrt((2*this_example_subset[:,:,4]+1)**2-8*(this_example_subset[:,:,4]-this_example_subset[:,:,3])))
            ndmi_band = (this_example_subset[:,:,4]-this_example_subset[:,:,5])/(this_example_subset[:,:,4]+this_example_subset[:,:,5]+1e-7)
            nbr_band = (this_example_subset[:,:,4]-this_example_subset[:,:,6])/(this_example_subset[:,:,4]+this_example_subset[:,:,6]+1e-7)
            nbr2_band = (this_example_subset[:,:,5]-this_example_subset[:,:,6])/(this_example_subset[:,:,5]+this_example_subset[:,:,6]+1e-7)
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(ndvi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(evi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(savi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(msavi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(ndmi_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(nbr_band)))
            this_example_subset = np.dstack((this_example_subset, np.nan_to_num(nbr2_band)))
            # at this point, we pick which bands to forward based on command-line argument; (we are doing this in training_functions now)
            # this_example_subset = this_example_subset[:, :, self.bands]
            this_label_subset = label_subset[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size]
            if self.mode == 'train':
                # Convert NULL-pixels to Non-Forest Class only during training
                this_label_subset = fix(this_label_subset).astype(np.uint8)
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
            if self.one_hot:
                this_label_subset = np.eye(self.num_classes)[this_label_subset]
            # print(this_label_subset.shape, this_example_subset.shape)
            this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset, one_hot=self.one_hot)
            # if self.transformation:
            #     this_example_subset = self.transformation(this_example_subset)
            return {'input': this_example_subset, 'label': this_label_subset, 'sample_identifier': (example_path, this_row, this_col)}

        def __len__(self):
            return 1*self.total_images if self.mode == 'train' else self.total_images
    ######################################################################################
    transformation = None
    train_list, eval_list, test_list, temp_list = [], [], [], []
    if not os.path.exists(data_split_lists_path):
        os.mkdir(data_split_lists_path)
        print('LOG: No saved data found. Making new data directory {}'.format(data_split_lists_path))
        extended_data_path = generated_data_path
        full_examples_list = [os.path.join(extended_data_path, x) for x in os.listdir(extended_data_path)]
        random.shuffle(full_examples_list)
        train_split = int(train_split*len(full_examples_list))
        train_list = full_examples_list[:train_split]
        temp_list = full_examples_list[train_split:]
        eval_list = temp_list[0:len(temp_list)//2]
        test_list = temp_list[len(temp_list)//2:]
    ######################################################################################
    print('LOG: [train_list, eval_list, test_list] ->', len(train_list), len(eval_list), len(test_list))
    print('LOG: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))
    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    train_data = dataset(data_list=train_list, data_map_path=os.path.join(data_split_lists_path, 'train_datamap.pkl'), mode='train', stride=8,
                         transformation=transformation)  # more images for training
    eval_data = dataset(data_list=eval_list, data_map_path=os.path.join(data_split_lists_path, 'eval_datamap.pkl'), mode='test', stride=model_input_size,
                        transformation=transformation)
    test_data = dataset(data_list=test_list, data_map_path=os.path.join(data_split_lists_path, 'test_datamap.pkl'), mode='test', stride=model_input_size,
                        transformation=transformation)
    print('LOG: [train_data, eval_data, test_data] ->', len(train_data), len(eval_data), len(test_data))
    print('LOG: Data Split Integrity: set(train_list).isdisjoint(set(eval_list)) ->', set(train_list).isdisjoint(set(eval_list)))
    print('LOG: Data Split Integrity: set(train_list).isdisjoint(set(test_list)) ->', set(train_list).isdisjoint(set(test_list)))
    print('LOG: Data Split Integrity: set(test_list).isdisjoint(set(eval_list)) ->', set(test_list).isdisjoint(set(eval_list)))
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader


def check_generated_fnf_datapickle(example_path):
    with open(example_path, 'rb') as this_pickle:
        (example_subset, label_subset) = pickle.load(this_pickle)
        example_subset = np.nan_to_num(example_subset)
        label_subset = fix(np.nan_to_num(label_subset))
    # print(example_subset)
    this = np.asarray(255*(example_subset[:,:,[3,2,1]]), dtype=np.uint8)
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

    # loaders = get_dataloaders_generated_data(generated_data_path='/home/azulfiqar_bee15seecs/training_data/pickled_clipped_training_data/',
    # save_data_path = '/home/azulfiqar_bee15seecs/training_data/training_lists'
    loaders = get_dataloaders_generated_data(generated_data_path='/mnt/e/Forest Cover - Redo 2020/Google Cloud - Training/Training Data/Clipped dataset/'
                                                                 'Pickled_data/',
                                             save_data_path="/mnt/e/Forest Cover - Redo 2020/Google Cloud - Training/training_lists",
                                             model_input_size=128, num_classes=2, train_split=0.8, one_hot=True, batch_size=16, num_workers=4, max_label=2)

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
            this_example_subset = (examples[0].numpy()).transpose(1, 2, 0)
            this = np.asarray(255*(this_example_subset[:, :, [3, 2, 1]]), dtype=np.uint8)
            that = labels[0].numpy().astype(np.uint8)
            # ndvi = this_example_subset[:,:,11]
            print(this.shape, that.shape, np.unique(that))
            # that = np.argmax(that, axis=0)
            # print()
            for j in range(7):
                pl.subplot(4,3,j+1)
                pl.imshow(this_example_subset[:,:,11+j])
            pl.show()
            pass
        pass
    pass


if __name__ == '__main__':
    main()

    # # generate pickle files to train from
    # all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
    #                  "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
    # for district in all_districts:
    #     print("=======================================================================================================")
    #     get_images_from_large_file(bands=range(1, 12), year=2015, region=district, stride=256)

    # # check some generated pickle files
    # for i in range(1, 65):
    #     check_generated_fnf_datapickle(f'E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\Clipped dataset\\Pickled_data\\'
    #                                    f'abbottabad_2015_{i}.pkl')

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







