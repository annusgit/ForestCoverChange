

"""
    A few helper functions
"""

from __future__ import print_function
from __future__ import division
import os
import numpy as np
import PIL.Image as Image
import scipy.io as sio


def load_weights_from_matfiles(dir_path):
    """
        Uses scipy.io to read .mat files and loads weights into torch model
    :param path_to_file: path to mat file to read
    :return: None, but saves the model dictionary!
    """
    import pickle
    model_file = 'Unet_pretrained_model.pkl'
    if os.path.exists(os.path.join(dir_path, model_file)):
        print('loading saved model dictionary...')
        with open(os.path.join(dir_path, model_file), 'rb') as handle:
            model_dict = pickle.load(handle)
        for i, layer in enumerate(model_dict.keys(), 1):
            print('{}.'.format(i), layer, model_dict[layer].shape)
    else:
        model_dict = {}
        for file in [x for x in os.listdir(dir_path) if x.endswith('.mat')]:
            layer, _ = os.path.splitext(file)
            try:
                read = sio.loadmat(os.path.join(dir_path, file))
            except:
                print(layer)
            print(layer, read[layer].shape)
            model_dict[layer] = read[layer]
        pass
        os.chdir('/home/annus/Desktop/trainedUnet/weightsforpython/')
        with open(model_file, 'wb') as handle:
            pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved model!!!')


def show_image():

    def histeq(im):
        """  Histogram equalization of a grayscale image. """
        nbr_bins = 256
        # get image histogram
        imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
        cdf = imhist.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        im2 = np.interp(im.flatten(), bins[:-1], cdf)

        return im2.reshape(im.shape)

    os.chdir('/home/annus/Desktop/rit18_data/')
    train_data = np.load('train_data.npy', mmap_mode='r').transpose((2, 1, 0))
    print(train_data.shape)
    w, h, patch = 2000, 2000, 1000
    image = train_data[w:w + patch, h:h + patch, 4:]
    # image = (255 / 65536 * image).astype(np.int8)
    r, g, b = map(histeq, [image[:, :, 0], image[:, :, 1], image[:, :, 2]])
    image = Image.fromarray(np.dstack((r, g, b)), 'RGB')
    # image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
    #                            dtype=cv2.CV_32F).astype(np.int8)
    # print(image.shape, image.dtype, np.max(np.max(image)), np.min(np.min(image)), np.mean(np.mean(image)))
    # pl.imshow(image)
    # pl.axis('off')
    # pl.show()
    os.chdir('/home/annus/Desktop/')
    image.save('image.png')