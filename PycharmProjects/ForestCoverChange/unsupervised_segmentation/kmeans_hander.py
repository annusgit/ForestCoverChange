

from __future__ import print_function, division
import os
from k_means import *
import matplotlib.pyplot as pl


def combine_bands(mapped_bands_list, coordinates):
    r, r_, c, c_ = coordinates
    full_array = mapped_bands_list[0][r:r_, c:c_]
    for k in range(1,13):
        full_array = np.dstack((full_array, mapped_bands_list[k][r:r_, c:c_]))
    return full_array


def main(example_path, clusters_save_path, look_at_images=False):
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


if __name__ == '__main__':
    main(example_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                      'full-test-site-pakistan/numpy_sums/2015',
         clusters_save_path='/home/annus/Desktop/all_clustering_results/2015')




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
