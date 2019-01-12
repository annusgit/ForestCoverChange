

from __future__ import print_function
from __future__ import division
import os
import sys
import gdal
import random
import pickle
import numpy as np
bands = range(1, 14)  # we need 13 bands


def get_signatures(image_path, save_path, save_number, seed_count):
    example = gdal.Open(image_path)
    example_array = example.GetRasterBand(bands[0]).ReadAsArray()
    for i in bands[1:]:
        example_array = np.dstack((example_array,
                                   example.GetRasterBand(i).ReadAsArray())).astype(np.int16)
    signatures = example_array.reshape((-1,13))
    non_zero_indices = np.sum(signatures, axis=1)!=0
    non_zero_full_array = signatures[non_zero_indices]
    non_zero_rows = np.count_nonzero((signatures != 0).sum(1))
    for k in range(non_zero_full_array.shape[0]):
        np.save(os.path.join(save_path, '{}.npy'.format(k+1+seed_count)), non_zero_full_array[k])
    return non_zero_rows


def signatures_in_one_folder(folder_path, save_number, save_path):
    total_signatures = 0
    # all_signatures_array = np.empty((1,13))
    for image in os.listdir(folder_path):
        valid_signature_count = get_signatures(image_path=os.path.join(folder_path, image),
                                               save_path=save_path,
                                               save_number=save_number,
                                               seed_count=total_signatures)
        total_signatures += valid_signature_count
        # all_signatures_array = np.vstack((all_signatures_array, valid_signatures))
    # return total_signatures, all_signatures_array
    return total_signatures


def main(folder_path, dest_path):
    # we will pick only 3,00,000 entries for each class
    signatures_dictionary = {}
    all_signatures = None
    if os.path.exists(dest_path):
        import shutil
        shutil.rmtree(dest_path)
    os.mkdir(dest_path)
    # randomly pick 3,00,000 elements
    required_signature_count = 300000

    for signature_folder in os.listdir(folder_path):
        print('log: on signature: {}'.format(signature_folder))
        inner_save_path = os.path.join(dest_path, signature_folder)
        os.mkdir(inner_save_path)
        total_signatures = signatures_in_one_folder(folder_path=os.path.join(folder_path,
                                                                             signature_folder),
                                                    save_number=required_signature_count,
                                                    save_path=inner_save_path)
        # signatures_array = signatures_array[random.sample(range(total_signatures),
        #                                                   required_signature_count)]
        # print('{}: {}, {}'.format(signature_folder, total_signatures, signatures_array.shape))
        # np.save('signatures/{}.npy'.format(signature_folder), signatures_array)
        # reshaped_signatures = np.reshape(signatures_array,(1,required_signature_count,13))
        # if all_signatures is not None:
        #     all_signatures = np.concatenate((all_signatures, reshaped_signatures), axis=0)
        # else:
        #     all_signatures = reshaped_signatures
    # for v, k in signatures_dictionary.items():
    #     print(v, k.shape)
    # np.save('signatures/all_signatures.npy', all_signatures)
    pass


def check_saved_data(pickle_path):
    with open(pickle_path, 'rb') as pickle_file:
        signatures_dictionary = pickle.load(pickle_file)
    print('\nchecking saved data now...')
    for v, k in signatures_dictionary.items():
        print(v, k.shape)
    pass


def check_saved_nparrs(folder_path):
    print('\nlog: Checking now...')
    for file in [x for x in os.listdir(folder_path) if x.endswith('.npy')]:
        arr = np.load(os.path.join(folder_path, file))
        print(file, arr.shape)
    pass


if __name__ == '__main__':
    main(folder_path=sys.argv[1], dest_path=sys.argv[2])
    # check_saved_nparrs(folder_path=sys.argv[2])








