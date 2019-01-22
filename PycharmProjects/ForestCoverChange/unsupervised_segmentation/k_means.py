


from __future__ import print_function
from __future__ import division
import time
import gdal
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt


def call_kmeans(samples_vec, n_clusters=10, n_iterations=30, pickle_file='kmeans.pkl'):
    np.random.seed(int(time.time()))
    kmeans = KMeans(n_clusters=n_clusters,
                    init='random',
                    n_init=10,
                    max_iter=n_iterations,
                    tol=1e-4,
                    precompute_distances='auto',
                    verbose=True,
                    random_state=int(time.time()),
                    algorithm='auto').fit(samples_vec)
    if pickle_file is not None:
        with open(pickle_file, 'wb') as pickle:
            print('log: writing k-means output in {}'.format(pickle_file))
            pkl.dump(kmeans, pickle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        return kmeans
    pass


def call_mini_kmeans(n_clusters, max_iter, batch_size, pickle_file=None):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             max_iter=max_iter,
                             batch_size=batch_size,
                             verbose=True)
    if pickle_file is not None:
        with open(pickle_file, 'wb') as pickle:
            print('log: writing k-means output in {}'.format(pickle_file))
            pkl.dump(kmeans, pickle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        return kmeans
    pass













