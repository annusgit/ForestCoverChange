

from __future__ import print_function
from __future__ import division
import torch
import random
random.seed(74)
import numpy as np
import matplotlib.pyplot as pl
from torch.utils.data import Dataset, DataLoader
from json_parser import get_values


def get_dataloaders(file_path, batch_size):
    print('inside dataloading code...')

    class dataset(Dataset):
        def __init__(self, data, max_seq_len=20, mode='train'):
            super(dataset, self).__init__()
            self.max_seq_len = max_seq_len
            self.data = data
            self.mode = mode
            pass

        def __getitem__(self, k):
            if self.mode != 'train':
                this_example = self.data[:]
                this_label = self.data[-1]
                this_example = torch.Tensor(this_example)
                this_label = torch.Tensor([this_label])
                return {'input': this_example, 'label': this_label}
            length = 5 # random.randint(2, self.max_seq_len)
            this_example = self.data[k:k+length]
            this_label = self.data[k+length]
            this_example = torch.Tensor(this_example)
            this_label = torch.Tensor([this_label])
            return {'input': this_example, 'label': this_label}

        def __len__(self):
            if self.mode != 'train':
                return 1
            return len(self.data)-self.max_seq_len

    dataset_list = get_values(this_file=file_path, window_size=8)['value_mean']
    # split them into train and test
    train, val, test = dataset_list[:350], dataset_list[350:400], dataset_list[400:]
    train_data = dataset(data=train, max_seq_len=20, mode='train')
    val_data = dataset(data=val, mode='no_train')
    test_data = dataset(data=test, mode='no_train')
    print('train examples =', len(train), 'val examples =', len(val), 'test examples =', len(test))
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=False, num_workers=4)
    return train_dataloader, val_dataloader, test_dataloader


def main():
    this_file = '/home/annus/Desktop/statistics_250m_16_days_NDVI.json'
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(file_path=this_file, batch_size=1)
    count = 0
    count += 1
    for idx, data in enumerate(test_dataloader):
        examples, labels = data['input'], data['label']
        print('{} -> on batch {}/{}, {}'.format(count, idx +1, len(train_dataloader), examples.size()))
        if True:
            print(examples, labels)


if __name__ == '__main__':
    main()














