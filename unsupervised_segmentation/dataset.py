

from __future__ import print_function
from __future__ import division
import os
import sys
import torch
import random
import pickle as p
import numpy as np
random.seed(74)
import matplotlib.pyplot as pl
from torch.utils.data import Dataset, DataLoader


all_labels = {
    'new_signature_lowvegetation': 0,
    'new_signature_forest': 1,
    'new_signature_urban': 2,
    'new_signature_cropland': 3,
    'new_signature_waterbody': 4
}
labels_arr_shape = (5, 300000, 13)


def toTensor(image):
    "converts a single input image to tensor"
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    # print(image.shape)a
    return torch.from_numpy(image).float()


def get_dataloaders(path_to_nparray, batch_size, normalize=False):
    print('inside dataloading code...')

    class dataset(Dataset):
        def __init__(self, num_classes, num_examples_per_class, data_list):
            self.classes = num_classes
            self.class_examples_count = num_examples_per_class
            # we will map the array to readable one within this function
            self.dataset = np.load(path_to_nparray, mmap_mode='r')
            self.data_list = data_list # this will differentiate between training, testing and validation lists
            self.normalize = normalize
            pass

        def __getitem__(self, k):
            # dataset size is 5*300000, where we have 300000 examples in each class and five classes
            # k is the index of the example (total examples = 5*300000 = 1500000 (0-1499999))
            label, example_number = k // self.class_examples_count, k % self.class_examples_count
            # this is very very important, because we are loading from three different sets
            example_number = self.data_list[example_number]
            example = np.expand_dims(self.dataset[label, example_number, :], axis=2)
            if normalize:
                example = 2*(example/4096.).clip(0,1)-1
            example = torch.Tensor(example)
            # print(label_arr)
            # print(example)
            return example, label

        def __len__(self):
            return self.class_examples_count*self.classes

    if not os.path.exists('training.pkl'):
        print("log: No cached data available, saving now...")
        total_examples_per_class = 300000
        # we will create three datasets for training, evaluation and testing
        total_examples = range(total_examples_per_class)
        random.shuffle(total_examples) # this happens inplace
        training_list = total_examples[:210000]
        evaluation_list = total_examples[210000:230000]
        testing_list = total_examples[230000:]
        with open('training.pkl', 'wb') as training_pickle:
            p.dump(training_list, training_pickle, protocol=p.HIGHEST_PROTOCOL)
        with open('evaluation.pkl', 'wb') as eval_pickle:
            p.dump(evaluation_list, eval_pickle, protocol=p.HIGHEST_PROTOCOL)
        with open('testing.pkl', 'wb') as testing_pickle:
            p.dump(testing_list, testing_pickle, protocol=p.HIGHEST_PROTOCOL)
    else:
        print("log: Found cached data, loading now...")
        with open('training.pkl', 'rb') as training_pickle:
            training_list = p.load(training_pickle)
        with open('evaluation.pkl', 'rb') as eval_pickle:
            evaluation_list = p.load(eval_pickle)
        with open('testing.pkl', 'rb') as testing_pickle:
            testing_list = p.load(testing_pickle)

    train_dataset = dataset(num_classes=5, num_examples_per_class=210000, data_list=training_list)
    val_dataset = dataset(num_classes=5, num_examples_per_class=20000, data_list=evaluation_list)
    test_dataset = dataset(num_classes=5, num_examples_per_class=70000, data_list=testing_list)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return (train_loader, val_loader, test_loader)


def check_dataloaders():
    load1, load2, load3 = get_dataloaders(path_to_nparray='/home/annus/Desktop/signatures/all_signatures.npy',
                                          batch_size=16, normalize=True)
    for idx, data in enumerate(load2):
        examples, labels = data
        print('on batch {}/{}, {}'.format(idx + 1, len(load2), examples.size()))
    pass



if __name__ == '__main__':
    check_dataloaders()













