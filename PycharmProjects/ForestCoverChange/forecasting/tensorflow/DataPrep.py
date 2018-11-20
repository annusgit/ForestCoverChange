

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import math
from Smooth import smooth
import pandas

def pre_data(file_path, batch_size, seq_len):
    '''
    Prepare data for training and testing

    :batch_size: batch size for training/testing, type: int
    :seq_len: sequence length, type: int
    :return: (encoder input, expected decoder output), type: tuple, shape: [seq_len, batch_size, out_dim]
    '''
    # converting mean values from the CSV file into a smoothened numpy array
    data_dir = file_path
    df = pandas.read_csv(data_dir, converters={"value_mean": float})["value_mean"].values
    df = smooth(df, window_len=5)
    # print(df.shape)
    # pl.plot(df)
    # pl.show()
    X_batch = []
    Y_batch = []
    for _ in range(batch_size):
        # offset = np.random.random_sample() * math.pi
        # t = np.linspace(start=offset, stop=offset+4*math.pi, num=2*seq_len)
        # print(t)
        # seq_1 = np.sin(t)
        seq_1 = df
        # seq_2 = np.cos(t)
        x1 = seq_1[:seq_len]
        y1 = seq_1[seq_len:]
        # x2 = seq_2[:seq_len]
        # y2 = seq_2[seq_len:]
        X = np.array([x1])  # size: [out_dim, seq_len]
        Y = np.array([y1])
        X = X.T  # size: [seq_len, out_dim]
        Y = Y.T
        X_batch.append(X)
        Y_batch.append(Y)

    X_batch = np.array(X_batch)  # size: [batch_size, seq_len, out_dim]
    Y_batch = np.array(Y_batch)
    print ("X batch", X_batch.shape)
    print("Y batch", Y_batch.shape)
    X_batch = np.transpose(X_batch, (1, 0, 2))  # size: [seq_len, batch_size, out_dim]
    Y_batch = np.transpose(Y_batch, (1, 0, 2))
    return X_batch, Y_batch


def train_test_data(file_path, batch_size, input_length, test_percent=0.2, scale_factor=1):
    '''
        Prepare data for training and testing, split it into training and testing examples
        :batch_size: batch size for training/testing, type: int
        :input_length: input length to feed into the recurrent network, the remaining will be the labels, type: int
        :return: (encoder input, expected decoder output), type: tuple, shape: [seq_len, batch_size, out_dim] for training set
                 and test_set values to see how the model behaves on unseen data
        '''
    # converting mean values from the CSV file into a smoothened numpy array
    data_dir = file_path
    df = pandas.read_csv(data_dir, converters={"value_mean": float})["value_mean"].values
    df = scale_factor*smooth(df, window_len=5)
    # divide into train and test sets
    train_count = int((1-test_percent)*len(df))
    df_train = df[:train_count]
    df_test = df[train_count:]
    X_batch = []
    Y_batch = []
    for _ in range(batch_size):
        seq_1 = df_train
        X = np.array([seq_1[:input_length]])  # size: [out_dim, seq_len]
        Y = np.array([seq_1[input_length:]])
        X = X.T  # size: [seq_len, out_dim]
        Y = Y.T
        X_batch.append(X)
        Y_batch.append(Y)

    TEST = np.array([df_test])  # size: [out_dim, seq_len]
    TEST = TEST.T  # size: [seq_len, out_dim]
    TEST = TEST.reshape(1, TEST.shape[0], 1)  # size: [batch_size=1, seq_len=(test-set-length), out_dim=1]

    X_batch = np.array(X_batch)  # size: [batch_size, seq_len, out_dim]
    Y_batch = np.array(Y_batch)
    print("X batch", X_batch.shape)
    print("Y batch", Y_batch.shape)
    print("TEST data", TEST.shape)
    X_batch = np.transpose(X_batch, (1, 0, 2))  # size: [seq_len, batch_size, out_dim]
    Y_batch = np.transpose(Y_batch, (1, 0, 2))
    TEST = np.transpose(TEST, (1, 0, 2))
    return (X_batch, Y_batch), TEST


if __name__ == '__main__':
    (X_train, Y_train), test_set = train_test_data(file_path='convertcsv.csv',
                                                   batch_size=16, input_length=200,
                                                   test_percent=0.1)
    print(X_train.shape, Y_train.shape, test_set.shape)













