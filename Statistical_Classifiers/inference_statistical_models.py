"""
    Given the path to a single test image, this function generates its corresponding segmentation map
"""
from __future__ import print_function
from __future__ import division
import os
import gdal
import time
import torch
import shutil
import random
import argparse
import numpy as np
import pickle as cPickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as matimg
np.random.seed(int(time.time()))
random.seed(int(time.time()))

rasterized_shapefiles_path = "/home/azulfiqar_bee15seecs/District_Shapefiles_as_Clipping_bands/"
FOREST_LABEL, NON_FOREST_LABEL, NULL_LABEL = 2, 1, 0


def mask_landsat8_image_using_rasterized_shapefile(district, this_landsat8_bands_list):
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
    diff_x_before, diff_y_before = diff_x//2, diff_y//2
    clipped_full_spectrum_resized = [np.pad(x, [(diff_x_before, diff_x-diff_x_before), (diff_y_before, diff_y-diff_y_before)], mode='constant')
                                     for x in clipped_full_spectrum]
    clipped_full_spectrum_stacked_image = np.dstack(clipped_full_spectrum_resized)
    print("{}: Generated Image Size: {}".format(district, clipped_full_spectrum_stacked_image.shape))
    return clipped_full_spectrum_stacked_image


def toTensor(**kwargs):
    image = kwargs['image']
    'will convert image and label from numpy to torch tensor'
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    # image = image.transpose((2, 0, 1))
    return torch.from_numpy(np.nan_to_num(image)).float()


def get_inference_loader(district, image_path, model_input_size=128, num_classes=3, one_hot=False, batch_size=64, num_workers=4):

    # This function is faster because we have already saved our data as subset pickle files
    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, image_path, bands, stride=model_input_size, transformation=None):
            super(dataset, self).__init__()
            self.model_input_size = model_input_size
            self.image_path = image_path
            self.all_images = []
            self.total_images = 0
            self.stride = stride
            self.one_hot = one_hot
            self.num_classes = num_classes
            self.transformation = transformation
            self.temp_dir = 'temp_numpy_saves'
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.mkdir(self.temp_dir)
            print('LOG: Generating data map now...')
            image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
            all_raster_bands = [image_ds.GetRasterBand(x).ReadAsArray() for x in bands]
            # mask the image and adjust its size at this point
            test_image = mask_landsat8_image_using_rasterized_shapefile(district=district, this_landsat8_bands_list=all_raster_bands)
            temp_image_path = os.path.join(self.temp_dir, 'temp_image.npy')
            np.save(temp_image_path, test_image)
            self.temp_test_image = np.load(temp_image_path, mmap_mode='r')
            row_limit = self.temp_test_image.shape[0] - model_input_size
            col_limit = self.temp_test_image.shape[1] - model_input_size
            test_image, image_ds, all_raster_bands = [None] * 3  # release memory
            for i in range(0, row_limit+1, self.stride):
                for j in range(0, col_limit+1, self.stride):
                    self.all_images.append((i, j))
                    self.total_images += 1
            self.shape = [i+self.stride, j+self.stride]
            pass

        def __getitem__(self, k):
            (this_row, this_col) = self.all_images[k]
            this_example_subset = self.temp_test_image[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size, :]
            this_example_subset = toTensor(image=this_example_subset)
            return {'coordinates': np.asarray([this_row, this_row + self.model_input_size, this_col, this_col + self.model_input_size]),
                    'input': this_example_subset}

        def __len__(self):
            return self.total_images

        def get_image_size(self):
            return self.shape

        def clear_mem(self):
            shutil.rmtree(self.temp_dir)
            print('Log: Temporary memory cleared')

    ######################################################################################
    transformation = None
    ######################################################################################
    # create dataset class instances
    inference_data = dataset(image_path=image_path, bands=[x for x in range(1, 12)], transformation=transformation)
    print('LOG: inference_data ->', len(inference_data))
    inference_loader = DataLoader(dataset=inference_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return inference_loader


@torch.no_grad()
def run_inference(args):
    trained_classifier = None
    with open(args.model_path, 'rb') as model_file:
        trained_classifier = cPickle.load(model_file)
    print('Log: Loaded pretrained from {}'.format(args.model_path))
    all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                     "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
    years = [2014, 2016, 2017, 2018, 2019, 2020]
    # change this to do this for all the images in that directory
    for district in all_districts:
        for year in years:
            print("(LOG): On District: {} @ Year: {}".format(district, year))
            test_image_path = os.path.join(args.dir_path, 'landsat8_4326_30_{}_region_{}.tif'.format(year, district))
            inference_loader = get_inference_loader(district=district, image_path=test_image_path, model_input_size=128, num_classes=3, one_hot=True,
                                                    batch_size=args.bs, num_workers=4)
            # we need to fill our new generated test image
            generated_map = np.empty(shape=inference_loader.dataset.get_image_size())
            for idx, data in enumerate(inference_loader):
                coordinates, test_x = data['coordinates'].tolist(), data['input']
                # print(test_x.shape)
                pred_numpy = trained_classifier.predict(test_x.reshape(-1, 11))
                # print(pred_numpy.shape)
                pred_numpy = pred_numpy.reshape((-1, 128, 128))
                # print(pred_numpy.shape)
                if idx % 5 == 0:
                    print('LOG: on {} of {}'.format(idx, len(inference_loader)))
                for k in range(test_x.shape[0]):
                    x, x_, y, y_ = coordinates[k]
                    generated_map[x:x_, y:y_] = pred_numpy[k,:,:]
            # save generated map as png image, not numpy array
            forest_map_rband = np.zeros_like(generated_map)
            forest_map_gband = np.zeros_like(generated_map)
            forest_map_bband = np.zeros_like(generated_map)
            forest_map_rband[generated_map == NON_FOREST_LABEL] = 255
            forest_map_gband[generated_map == FOREST_LABEL] = 255
            forest_map_for_visualization = np.dstack([forest_map_rband, forest_map_gband, forest_map_bband]).astype(np.uint8)
            save_this_map_path = os.path.join(args.dest, '{}_{}.png'.format(district, year))
            matimg.imsave(save_this_map_path, forest_map_for_visualization)
            print('Saved: {} @ {}'.format(save_this_map_path, forest_map_for_visualization.shape))
            # save_path = os.path.join(args.dest, 'generated_map_{}_{}.npy'.format(district, year))
            # np.save(save_path, generated_map)
            #########################################################################################3
            inference_loader.dataset.clear_mem()
            pass
        pass
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', '--model', dest='model_path', type=str)
    parser.add_argument('--d', dest='dir_path', type=str)
    parser.add_argument('--s', dest='dest', type=str)
    parser.add_argument('--b', dest='bs', type=int)
    parser.add_argument('--cuda', dest='cuda', type=int)
    parser.add_argument('--device', dest='device', type=int)
    args = parser.parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
