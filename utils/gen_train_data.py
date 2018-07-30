

from __future__ import print_function
from __future__ import division
from skimage.measure import block_reduce
import matplotlib.pyplot as pl
import scipy as sp
from tqdm import trange
import numpy as np
import argparse
import cv2
import os


# argv[1] -> source directory containing cuts
# argv[2] -> destination directory to store augmented data
# argv[3] -> number of random crops to generate from each image

def generate():
    all_image_files = os.listdir(source)
    crop_size = 128
    for k in trange(len(all_image_files)):
        name, _ = os.path.splitext(all_image_files[k])
        x = os.path.join(source, all_image_files[k])
        y = os.path.join(annotations, all_image_files[k])
        actual_image = pl.imread(x)
        label = pl.imread(y)
        # print(all_image_files[k])
        w, h, c = actual_image.shape
        if actual_image is not None:
            coords = []
            # this is for random crops
            for i in range(crops):
                # crop randomly
                x = np.random.randint(low=0, high=w-crop_size)
                y =	np.random.randint(low=0, high=h-crop_size)
                while (x,y) in coords:
                    x = np.random.randint(low=0, high=w-crop_size)
                    y =	np.random.randint(low=0, high=h-crop_size)
                coords.append((x,y))
                croped_image = actual_image[x:x+crop_size,y:y+crop_size,:]
                croped_label = label[x:x+crop_size,y:y+crop_size,:]
                # # downsample now
                # for _ in range(4):
                #     croped_image = block_reduce(image=croped_image, block_size=(2,2,1), func=np.max)
                #     croped_label = block_reduce(image=croped_label, block_size=(2,2,1), func=np.max)

                cv2.imwrite(os.path.join(im_dest_path, name+'_{}.png'.format(i)), croped_image)
                cv2.imwrite(os.path.join(an_dest_path, name+'_{}.png').format(i), croped_label)
    pass


def convert_labels(label_im):
    conversions = {76:0, 150:1, 179:2, 226:3, 255:4}
    gray = cv2.cvtColor(label_im, cv2.COLOR_RGB2GRAY)
    for k in conversions.keys():
        gray[gray == k] = conversions[k]
    # print(np.unique(gray))
    return gray


def get_subimages():
    sub_folders = os.listdir(source)
    for folder in sub_folders:
        source_image_folder = os.path.join(source, folder)
        source_label_folder = os.path.join(annotations, folder)
        dest_image_folder = os.path.join(im_dest_path, folder)
        dest_label_folder = os.path.join(an_dest_path, folder)
        os.mkdir(dest_image_folder)
        os.mkdir(dest_label_folder)
        all_image_files = os.listdir(source_image_folder)
        for k in trange(len(all_image_files)):
            name, _ = os.path.splitext(all_image_files[k])
            x = os.path.join(source_image_folder, all_image_files[k])
            y = os.path.join(source_label_folder, all_image_files[k])
            # image -> (H, W, C)
            actual_image = cv2.imread(x)
            label = cv2.imread(y)
            # print(all_image_files[k])
            if actual_image is not None:
                w, h, c = actual_image.shape
                coords = []
                # this is for random crops
                for i in range(crops):
                    # crop randomly
                    x = np.random.randint(low=0, high=w - crop_size)
                    y = np.random.randint(low=0, high=h - crop_size)
                    while (x, y) in coords:
                        x = np.random.randint(low=0, high=w - crop_size)
                        y = np.random.randint(low=0, high=h - crop_size)
                    coords.append((x, y))
                    croped_image = actual_image[x:x + crop_size, y:y + crop_size, :]
                    croped_label = label[x:x + crop_size, y:y + crop_size, :]
                    # print(croped_image.shape, croped_label.shape)
                    sp.misc.imsave(os.path.join(dest_image_folder, name + '_{}.png'.format(i)), croped_image)
                    sp.misc.imsave(os.path.join(dest_label_folder, name + '_{}.png').format(i), croped_label)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', dest='images_path')
    parser.add_argument('--annotations', dest='labels_path')
    parser.add_argument('--crops', dest='crops')
    parser.add_argument('--cropsize', dest='crop_size')
    parser.add_argument('--im_dest', dest='im_dest_path')
    parser.add_argument('--an_dest', dest='an_dest_path')
    args = parser.parse_args()

    global source, annotations, im_dest_path, an_dest_path, crops, crop_size
    source = args.images_path
    annotations = args.labels_path
    crops = int(args.crops)
    crop_size = int(args.crop_size)
    im_dest_path = args.im_dest_path
    an_dest_path = args.an_dest_path
    # print('paths => ', im_dest_path, an_dest_path)

    import shutil
    if os.path.exists(im_dest_path):
        shutil.rmtree(im_dest_path)
        print('removed {}'.format(im_dest_path))
    if os.path.exists(an_dest_path):
        shutil.rmtree(an_dest_path)
        print('removed {}'.format(an_dest_path))

    os.mkdir(im_dest_path)
    os.mkdir(an_dest_path)
    # important for reproducibility
    np.random.seed(17)
    get_subimages()








