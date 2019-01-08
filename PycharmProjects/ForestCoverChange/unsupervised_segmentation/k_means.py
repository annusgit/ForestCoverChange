


from __future__ import print_function
from __future__ import division
import time
import gdal
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
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
    with open(pickle_file, 'wb') as pickle:
        print('log: writing k-means output in {}'.format(pickle_file))
        pkl.dump(kmeans, pickle, protocol=pkl.HIGHEST_PROTOCOL)
    pass








