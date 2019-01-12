

from __future__ import print_function, division
import os
from k_means import *
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
from model import MODEL


def combine_bands(mapped_bands_list, coordinates):
    r, r_, c, c_ = coordinates
    full_array = mapped_bands_list[0][r:r_, c:c_]
    for k in range(1,13):
        full_array = np.dstack((full_array, mapped_bands_list[k][r:r_, c:c_]))
    return full_array


def run_clustering(example_path, clusters_save_path, look_at_images=False):
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
def classify_cluster_centers(clusters_save_path, model_path):
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
    pl.imshow(full_classified_image)
    pl.show()
    pass


@torch.no_grad()
def classify_cluster_arrays(example_path, clusters_save_path, model_path):
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
                    test_vectors = np.expand_dims(samples_vec[kmeans_subset_labels==u][:100], axis=2)
                    test_tensor = torch.Tensor(test_vectors)
                    # print(test_tensor.shape)
                    out_x, pred = model(test_tensor)
                    pred_numpy = pred.numpy()
                    vals, counts = np.unique(pred_numpy, return_counts=True)
                    this_index = np.argmax(counts)
                    classified_mini_array[classified_mini_array == u] = vals[this_index]
                    # print('{}%'.format(counts[this_index]))
                    # print()


                # # now we assign classes to clusters on the bases of their classification results
                # classified_mini_array = kmeans_subset_labels.copy()
                # for u in range(10): # because we have ten clusters in each subset
                #     # u is the cluster number
                #     classified_mini_array[kmeans_subset_labels == u] = pred_arr[u]

                # convert classified array to image
                mini_image_array = classified_mini_array.reshape(stride, stride)
                full_classified_image[row*stride:(row+1)*stride, col*stride:(col+1)*stride] = mini_image_array
    pl.imshow(full_classified_image)
    pl.show()
    pass


if __name__ == '__main__':
    # run_clustering(example_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
    #                   'full-test-site-pakistan/numpy_sums/2015',
    #      clusters_save_path='/home/annus/Desktop/all_clustering_results/2015')

    # recombine_clusters_and_view('/home/annus/Desktop/all_clustering_results/2015')

    # classify_cluster_centers(clusters_save_path='/home/annus/Desktop/all_clustering_results/2015',
    #                               model_path='/home/annus/Desktop/trained_models/model-25.pt')

    classify_cluster_arrays(example_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                                         'full-test-site-pakistan/numpy_sums/2015',
                            clusters_save_path='/home/annus/Desktop/all_clustering_results/2015',
                            model_path='/home/annus/Desktop/trained_models/model-25.pt')


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
