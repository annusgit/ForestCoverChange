

"""
    TODO: Add support for k-means for one whole image and then combine for multi-temporal k-means
"""


from __future__ import print_function, division
import os
import random
from scipy import ndimage
from k_means import *
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from model import MODEL


color_mapping = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255), 3: (255,255,0), 4:(0,255,255)}

def combine_bands(mapped_bands_list, coordinates):
    r, r_, c, c_ = coordinates
    full_array = mapped_bands_list[0][r:r_, c:c_]
    for k in range(1,13):
        full_array = np.dstack((full_array, mapped_bands_list[k][r:r_, c:c_]))
    return full_array


def run_clustering(example_path, clusters_save_path, look_at_images=False, normalize=True):
    # first map all bands in memory for reading later on...
    all_bands = []
    for i in range(1,14):
        all_bands.append(np.load(os.path.join(example_path, '{}.npy'.format(i)), mmap_mode='r'))

    full_test_site_shape = (3663, 5077)
    # the following variables are for a single slice of the entire image
    n_clusters = 10
    n_iterations = 30
    stride = 400
    n_data_samples = stride**2
    # samples_vec = np.random.random_sample(size=(n_data_samples, 2))

    # row and column wise stride on the entire image
    total = full_test_site_shape[0] // stride * full_test_site_shape[1] // stride
    count = 1
    for row in range(0, full_test_site_shape[0]//stride):
        for col in range(0, full_test_site_shape[1]//stride):
            this_cluster_save_path = os.path.join(clusters_save_path, '{}_{}_{}_{}.pkl'.format(row*stride,
                                                                                               (row+1)*stride,
                                                                                               col*stride,
                                                                                               (col+1)*stride))
            # in here read the 13 bands separately and combine them
            full_array = combine_bands(mapped_bands_list=all_bands, coordinates=(row*stride, (row+1)*stride,
                                                                                 col*stride, (col+1)*stride))
            if look_at_images:
                print(full_array.shape)
                show_image = np.asarray(np.clip(full_array[:,:,[4,3,2]]/4096, 0, 1)*255, dtype=np.uint8)
                pl.imshow(show_image)
                pl.title("rows {}-{}, columns {}-{}".format(row*stride, (row+1)*stride, col*stride, (col+1)*stride))
                pl.show()

            samples_vec = full_array.reshape((-1, 13))
            if normalize:
                samples_vec = 2*(samples_vec/4096.).clip(0,1)-1
                # print(samples_vec)
            samples_vec = np.nan_to_num(samples_vec) # important nan save
            print('Calling kmeans on {}/{} images'.format(count, total))
            call_kmeans(samples_vec=samples_vec, n_clusters=n_clusters, n_iterations=n_iterations,
                        pickle_file=this_cluster_save_path)
            count += 1



##########################################################################################
############3 these are for analyzing the results

def recombine_clusters_and_view(clusters_save_path):
    full_test_site_shape = (3663, 5077)
    # the following variables are for a single slice of the entire image
    stride = 400

    # row and column wise stride on the entire image
    full_clustered_image = np.zeros(shape=(stride*int(full_test_site_shape[0]/stride),
                                           stride*int(full_test_site_shape[1]/stride)))
    for row in range(0, full_test_site_shape[0]//stride):
        for col in range(0, full_test_site_shape[1]//stride):
            this_cluster_save_path = os.path.join(clusters_save_path, '{}_{}_{}_{}.pkl'.format(row*stride,
                                                                                               (row+1)*stride,
                                                                                               col*stride,
                                                                                               (col+1)*stride))

            print("rows {}-{}, columns {}-{}".format(row * stride, (row + 1) * stride,
                                                     col * stride, (col + 1) * stride))
            with open(this_cluster_save_path, 'rb') as read_this_one:
                kmeans_out_subset = pkl.load(read_this_one)
                array = kmeans_out_subset.labels_.reshape(stride, stride)
                # print(array.shape)
                full_clustered_image[row*stride:(row+1)*stride, col*stride:(col+1)*stride] = array
    pl.imshow(full_clustered_image)
    pl.show()
    pass


@torch.no_grad()
def classify_cluster_centers(clusters_save_path, model_path, save_image=None):
    full_test_site_shape = (3663, 5077)
    stride = 400
    # row and column wise stride on the entire image
    full_classified_image = np.zeros(shape=(stride*int(full_test_site_shape[0]/stride),
                                            stride*int(full_test_site_shape[1]/stride)))

    # load model for inference
    model = MODEL(in_channels=13)
    model.load_state_dict(torch.load(model_path))
    print('log: loaded saved model {}'.format(model_path))
    model.eval()

    for row in range(0, full_test_site_shape[0]//stride):
        for col in range(0, full_test_site_shape[1]//stride):
            this_cluster_save_path = os.path.join(clusters_save_path, '{}_{}_{}_{}.pkl'.format(row*stride,
                                                                                               (row+1)*stride,
                                                                                               col*stride,
                                                                                               (col+1)*stride))
            print("rows {}-{}, columns {}-{}".format(row * stride, (row + 1) * stride,
                                                     col * stride, (col + 1) * stride))
            with open(this_cluster_save_path, 'rb') as read_this_one:
                kmeans_subset = pkl.load(read_this_one)
                kmeans_subset_labels = kmeans_subset.labels_
                kmeans_subset_cluster_centers = np.expand_dims(kmeans_subset.cluster_centers_, 2)
                kmeans_subset_cluster_centers_tensor = torch.Tensor(kmeans_subset_cluster_centers)
                out_x, pred = model(kmeans_subset_cluster_centers_tensor)
                pred_arr = pred.numpy()

                # now we assign classes to clusters on the bases of their classification results
                classified_mini_array = kmeans_subset_labels.copy()
                for u in range(10): # because we have ten clusters in each subset
                    # u is the cluster number
                    classified_mini_array[kmeans_subset_labels == u] = pred_arr[u]

                # convert classified array to image
                mini_image_array = classified_mini_array.reshape(stride, stride)
                full_classified_image[row*stride:(row+1)*stride, col*stride:(col+1)*stride] = mini_image_array
    # apply median filter for clearer results
    full_classified_image = ndimage.median_filter(full_classified_image, size=5)
    print(full_classified_image.shape)
    pl.imshow(full_classified_image, cmap='tab10', alpha=1.0)
    pl.show()
    if save_image:
        # new array for saving rgb image
        image = np.zeros(shape=(full_classified_image.shape[0], full_classified_image.shape[1], 3))
        for j in range(5):
            image[full_classified_image == j, :] = color_mapping[j]
        pl.imsave(save_image, image)
        print('log: Saved {}'.format(save_image))
    pass


@torch.no_grad()
def classify_cluster_arrays(example_path, clusters_save_path, model_path, is_normalized=True):
    # we'll need the data for it too
    all_bands = []
    for i in range(1, 14):
        all_bands.append(np.load(os.path.join(example_path, '{}.npy'.format(i)), mmap_mode='r'))

    full_test_site_shape = (3663, 5077)
    stride = 400
    # row and column wise stride on the entire image
    full_classified_image = np.zeros(shape=(stride*int(full_test_site_shape[0]/stride),
                                            stride*int(full_test_site_shape[1]/stride)))

    # load model for inference
    model = MODEL(in_channels=13)
    model.load_state_dict(torch.load(model_path))
    print('log: loaded saved model {}'.format(model_path))
    model.eval()

    for row in range(0, full_test_site_shape[0]//stride):
        for col in range(0, full_test_site_shape[1]//stride):
            this_cluster_save_path = os.path.join(clusters_save_path, '{}_{}_{}_{}.pkl'.format(row*stride,
                                                                                               (row+1)*stride,
                                                                                               col*stride,
                                                                                               (col+1)*stride))
            print("rows {}-{}, columns {}-{}".format(row * stride, (row + 1) * stride,
                                                     col * stride, (col + 1) * stride))
            with open(this_cluster_save_path, 'rb') as read_this_one:
                kmeans_subset = pkl.load(read_this_one)
                kmeans_subset_labels = kmeans_subset.labels_

                # predict classes after receiving the data
                full_array = combine_bands(mapped_bands_list=all_bands, coordinates=(row * stride, (row + 1) * stride,
                                                                                     col * stride, (col + 1) * stride))
                samples_vec = full_array.reshape((-1, 13))
                samples_vec = np.nan_to_num(samples_vec)  # important nan save

                # pick 100 vectors for each cluster, classify and assign to clusters
                classified_mini_array = kmeans_subset_labels.copy()
                for u in range(10): # because we have ten clusters in each subset
                    test_these = samples_vec[kmeans_subset_labels==u]
                    if is_normalized:
                        test_these = 2*(test_these/4096.).clip(0,1)-1 # if normalization is used
                    test_vectors = np.expand_dims(test_these[random.sample(range(len(test_these)), 100)], axis=2)
                    test_tensor = torch.Tensor(test_vectors)
                    # print(test_tensor.shape)
                    out_x, pred = model(test_tensor)
                    pred_numpy = pred.numpy()
                    vals, counts = np.unique(pred_numpy, return_counts=True)
                    this_index = np.argmax(counts)
                    classified_mini_array[classified_mini_array == u] = vals[this_index]

                # convert classified array to image
                mini_image_array = classified_mini_array.reshape(stride, stride)
                full_classified_image[row*stride:(row+1)*stride, col*stride:(col+1)*stride] = mini_image_array

    pl.imshow(full_classified_image, cmap='tab10', alpha=1.0)
    pl.show()
    pass


def cross_clustering_single_image(clusters_save_path, model_path):
    '''
        Will combine clusters from a single image using cluster centers for smaller subsets
    :return:
    '''
    full_test_site_shape = (3663, 5077)
    stride = 400
    full_test_site_used_shape = (stride * (full_test_site_shape[0] // stride),
                                 stride * (full_test_site_shape[1] // stride))
    # print(full_test_site_used_shape)
    n_clusters = 400
    n_iterations = 30
    all_cluster_centers = None
    all_cluster_labels = np.zeros(full_test_site_used_shape[0]*full_test_site_used_shape[1])

    full_clustered_image = np.zeros(shape=(stride * int(full_test_site_shape[0] / stride),
                                           stride * int(full_test_site_shape[1] / stride)))

    # row and column wise stride on the entire image
    # count = -1
    for row in range(0, full_test_site_shape[0] // stride):
        for col in range(0, full_test_site_shape[1] // stride):
            # count += 1
            this_cluster_save_path = os.path.join(clusters_save_path, '{}_{}_{}_{}.pkl'.format(row * stride,
                                                                                               (row + 1) * stride,
                                                                                               col * stride,
                                                                                               (col + 1) * stride))
            print("log: Reading rows {}-{}, columns {}-{}".format(row * stride, (row + 1) * stride,
                                                                  col * stride, (col + 1) * stride))
            with open(this_cluster_save_path, 'rb') as read_this_one:
                kmeans_subset = pkl.load(read_this_one)
                if all_cluster_centers is not None:
                    all_cluster_centers = np.dstack((all_cluster_centers, kmeans_subset.cluster_centers_))
                else:
                    all_cluster_centers = kmeans_subset.cluster_centers_
                array = kmeans_subset.labels_.reshape(stride, stride)
                full_clustered_image[row * stride:(row + 1) * stride, col * stride:(col + 1) * stride] = array
                # bad bad bad
                # all_cluster_labels[count*stride**2:count*stride**2+stride**2] = np.asarray(array.reshape(-1) +
                #                                                                            10*count, dtype=np.uint16)

    pl.imshow(full_clustered_image)
    pl.title('Input image')
    pl.show()

    # verified that this doesn't work
    # reshaping loop (because I feel insecure with this thing)
    all_cluster_labels = full_clustered_image.reshape((-1, 108))
    # print(another == all_cluster_labels)
    # print(np.unique(all_cluster_labels), all_cluster_labels.dtype)

    all_cluster_centers = all_cluster_centers.transpose((2, 1, 0))
    all_cluster_centers = all_cluster_centers.reshape((-1, 13))

    # update labels to new values to keep them separated
    for i in range(all_cluster_labels.shape[1]):
        all_cluster_labels[:, i] += 10*i

    # cluster cluster-centers now
    print('log: Clustering now...')
    kmeans = call_kmeans(samples_vec=all_cluster_centers, n_clusters=n_clusters, n_iterations=n_iterations,
                         pickle_file=None)
    print('log: done clustering')
    # print(kmeans.labels_.shape)
    # we should have n_clusters unique labels now
    # now we assign classes to clusters on the bases of their classification results
    # because we have 108 subsets, each having their own 0-9 cluster labels
    print('log: assigning new labels...')
    # for i in range(108):
    for i in range(all_cluster_labels.shape[1]):
        for j in range(10):
            all_cluster_labels[all_cluster_labels == i*10+j] = kmeans.labels_[i*10+j]

    # check
    # show_image = all_cluster_labels.reshape(full_test_site_used_shape)
    # pl.imshow(show_image)
    # pl.title('Classified image')
    # pl.show()
    # return

    ########### classification starts now ...
    # load model for inference
    model = MODEL(in_channels=13)
    model.load_state_dict(torch.load(model_path))
    print('log: loaded saved model {}'.format(model_path))
    model.eval()

    outer_kmeans_centers = np.expand_dims(kmeans.cluster_centers_, 2)
    outer_kmeans_centers_tensor = torch.Tensor(outer_kmeans_centers)
    out_x, pred = model(outer_kmeans_centers_tensor)
    pred_arr = pred.numpy()

    # now we assign classes to clusters on the bases of their classification results
    all_cluster_labels = all_cluster_labels.reshape(-1)
    for u in range(n_clusters):  # because we n_clusters now
        # u is the cluster number
        all_cluster_labels[all_cluster_labels == u] = pred_arr[u]
        pass

    show_image = all_cluster_labels.reshape(full_test_site_used_shape)
    pl.imshow(show_image)
    pl.title('Classified image')
    pl.show()

    all_labels = {
        'new_signature_lowvegetation': 0,
        'new_signature_forest': 1,
        'new_signature_urban': 2,
        'new_signature_cropland': 3,
        'new_signature_waterbody': 4
    }
    reversed_labels = {v:k for k,v in all_labels.items()}
    classes_found = np.unique(show_image)
    for p in classes_found:
        pl.imshow(show_image == p)
        pl.title('Class {}'.format(reversed_labels[p]))
        pl.show()
    pass


if __name__ == '__main__':
    # run_clustering(example_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                   'full-test-site-pakistan/numpy_sums/2018',
    #      clusters_save_path='/home/annus/Desktop/all_clustering_results/normalized/2018',
    #                normalize=True)

    # recombine_clusters_and_view('/home/annus/Desktop/all_clustering_results/normalized/2018/')

    # classify_cluster_centers(clusters_save_path='/home/annus/Desktop/all_clustering_results/normalized/2018',
    #                          model_path='/home/annus/Desktop/trained_signature_classifier_normalized_input/model-7.pt',
    #                          save_image='/home/annus/Desktop/normalized_results/2018.png')

    # classify_cluster_arrays(example_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                                      'full-test-site-pakistan/numpy_sums/2017',
    #                         clusters_save_path='/home/annus/Desktop/all_clustering_results/normalized/2017',
    #                         model_path='/home/annus/Desktop/trained_signature_classifier_normalized_input/model-7.pt')

    cross_clustering_single_image(clusters_save_path='/home/annus/Desktop/all_clustering_results/normalized/2018',
                                  model_path='/home/annus/Desktop/trained_signature_classifier_normalized_input/'
                                             'model-7.pt')


############################################################################################################

# main(example_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                   'patch_wise/Pakistani_data/pakistan_test/pakistan_test_2.tif')
    # main(example_path='/home/annus/Desktop/image-2014.tif')


# bands = range(1,14) # 13 bands to read
# this_example = gdal.Open(example_path)
# example_array = this_example.GetRasterBand(bands[0]).ReadAsArray()
# for i in bands[1:]:
#     example_array = np.dstack((example_array,
#                                this_example.GetRasterBand(i).ReadAsArray())).astype(np.int16)
# print('image shape:', example_array.shape)
# example_array = example_array.reshape((-1,13)) # 13 bands
# # do some conditioning
# example_array = 2*(example_array/4096.).clip(0,1)-1
# print('array shape:', example_array.shape)
# samples_vec = np.nan_to_num(example_array.copy()) # important nan save
# print(samples_vec.max())
# print(samples_vec)

# divide the 18 million array into 18 1 million arrays
# for j in range(0, ):
# sub_sample_vec =


    # with open(pickle_file, 'rb') as pickle:
    #     print('log: reading k-means output...')
    #     kmeans_out = pkl.load(pickle)
    # print(dir(kmeans_out))
    # print(kmeans_out.cluster_centers_.shape)
    # print(len(kmeans_out.labels_))
    # print(samples_vec.shape)

    # cluster_count = {}
    # for i in range(n_clusters):
    #     cluster_count[i] = np.count_nonzero([kmeans_out.labels_ == i])
    # reversed_cluster_count = {v:k for k,v in cluster_count.items()}
    #
    # clustered_image = kmeans_out.labels_.reshape((example_array.shape[0], example_array.shape[1]))
    # for k in range(n_clusters):
    #     plt.figure()
    #     plt.title('Cluster: {}'.format(k))
    #     plt.imshow(clustered_image==k)
    #     plt.show()
    # forest_percentage = np.count_nonzero(kmeans_out.labels_==9)/(kmeans_out.labels_.shape[0])
    # plt.imshow(clustered_image, cmap='tab10', alpha=1.0)
    # plt.title('forest percentage = {:.2f}%'.format(forest_percentage*100))
    # plt.show()

    # plt.figure()
    # plt.title('original data')
    # plt.scatter(samples_vec.transpose()[0], samples_vec.transpose()[1],
    #             cmap='hsv', alpha=0.75)
    # clusters = {}
    # plt.figure()
    # plt.title('clustered data')
    # for i in range(n_clusters):
    #     clusters[i] = samples_vec[kmeans_out.labels_ == i]
    #     plt.scatter(clusters[i].transpose()[0],
    #                 clusters[i].transpose()[1], cmap='hsv', alpha=0.75)
    #     plt.scatter(kmeans_out.cluster_centers_[i].transpose()[0],
    #                 kmeans_out.cluster_centers_[i].transpose()[1],
    #                 c=[0, 0, 0], marker='x')
    # plt.show()
    # pass