

from __future__ import print_function
from __future__ import division
import os
import gdal
import shutil
import numpy as np


def WriteRaster(InputArray, file_name, dimension):
    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(file_name, dimension, dimension, 13, gdal.GDT_Int16)
    for i in range(13):
        dst_ds.GetRasterBand(i+1).WriteArray(InputArray[:,:,i])  # write r-band to the raster
    dst_ds.FlushCache()  # write to disk


def get_all_images(image_path, dest, name, dimension=64, stride=5, count_seed=0):
    '''
        This function gets all images from one large images with a stride given in the argument
    :param image_path: path to large image
    :param dest: folder to save the resultant images
    :param stride: step size in number of pixels
    :return: None
    '''
    # os.mkdir(dest)
    # print('log: Created {}'.format(dest))
    this_example = gdal.Open(image_path)
    bands = range(1, this_example.RasterCount+1)
    example_array = this_example.GetRasterBand(bands[0])
    example_array = example_array.ReadAsArray()
    for i in bands[1:]:
        example_array = np.dstack((example_array,
                                   this_example.GetRasterBand(i).ReadAsArray())).astype(np.int16)
    # at this point we have read in the full image
    # now let's begin the stride
    # print(example_array.shape)
    count = count_seed
    for i in range(0, example_array.shape[0]-dimension, stride):
        for j in range(0, example_array.shape[1]-dimension, stride):
            count += 1
            new_image = example_array[i:i+dimension, j:j+dimension, :]
            # print(new_image.shape)
            WriteRaster(InputArray=new_image,
                        file_name=os.path.join(dest, '{}-{}.tif').format(name, count),
                        dimension=dimension)
    pass


def main(src, dst):
    '''
        This function will just go through all of the images in all of the folders and create a new dataset folder
    :return: None
    '''
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)
    for class_folder in os.listdir(src):
        count = 0
        dst_class_folder = os.path.join(dst, class_folder)
        os.mkdir(dst_class_folder)
        class_folder_path = os.path.join(src,class_folder)
        for image in os.listdir(class_folder_path):
            count += 1
            print('log: On class {}, image {}'.format(class_folder, image))
            image_name, ext = os.path.splitext(image)
            src_image_path = os.path.join(class_folder_path,image)
            # dst_image_path = os.path.join(dst_class_folder, os.path.join(image_name+'-{}.tif'.format(count)))
            get_all_images(image_path=src_image_path, dest=dst_class_folder,
                           name=image_name, dimension=10, stride=5, count_seed=0)
    pass


def single_test():
    dest = '/home/annus/Desktop/test/'
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    get_all_images(image_path='/home/annus/Desktop/label_plainland_10.tif',
                   dest=dest,
                   name='this',
                   dimension=10,
                   stride=5)


if __name__ == '__main__':
    main(src='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
             'patch_wise/pakistan_better_data/data_unpacked',
         dst='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
             'patch_wise/pakistan_better_data/new_data_with_dim_10')
    # single_test()






