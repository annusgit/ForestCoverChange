

from __future__ import print_function, division
from k_means import *


def main(example_path):
    n_clusters = 10
    n_iterations = 30
    n_data_samples = 5000
    pickle_file = '../unsupervised_results/10clusters_2016.pkl'
    # samples_vec = np.random.random_sample(size=(n_data_samples, 2))

    bands = range(1,14) # 13 bands to read
    this_example = gdal.Open(example_path)
    example_array = this_example.GetRasterBand(bands[0]).ReadAsArray()
    for i in bands[1:]:
        example_array = np.dstack((example_array,
                                   this_example.GetRasterBand(i).ReadAsArray())).astype(np.int16)
    print('image shape:', example_array.shape)
    # example_array = example_array.reshape((-1,13))
    # do some conditioning
    # example_array = 2*(example_array / 4096.).clip(0,1)-1
    # print('array shape:', example_array.shape)
    # samples_vec = example_array.copy()
    # print(samples_vec)

    # call_kmeans(samples_vec=samples_vec, n_clusters=n_clusters, n_iterations=n_iterations, pickle_file=pickle_file)

    with open(pickle_file, 'rb') as pickle:
        print('log: reading k-means output...')
        kmeans_out = pkl.load(pickle)
    # print(dir(kmeans_out))
    # print(kmeans_out.cluster_centers_.shape)
    # print(len(kmeans_out.labels_))
    # print(samples_vec.shape)

    cluster_count = {}
    for i in range(n_clusters):
        cluster_count[i] = np.count_nonzero([kmeans_out.labels_ == i])
    reversed_cluster_count = {v:k for k,v in cluster_count.items()}

    clustered_image = kmeans_out.labels_.reshape((example_array.shape[0], example_array.shape[1]))
    # for k in range(n_clusters):
    #     plt.figure()
    #     plt.title('Cluster: {}'.format(k))
    #     plt.imshow(clustered_image==k)
    #     plt.show()
    forest_percentage = np.count_nonzero(kmeans_out.labels_==9)/(kmeans_out.labels_.shape[0])
    plt.imshow(clustered_image==0, cmap='tab10', alpha=1.0)
    plt.title('forest percentage = {:.2f}%'.format(forest_percentage*100))
    plt.show()

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


if __name__ == '__main__':
    # main(example_path='pakistan_test_4.tif')
    main(example_path='/home/annus/PycharmProjects/ForestCoverChange_inputs_and_numerical_results/'
                      'patch_wise/Pakistani_data/pakistan_test/pakistan_test_2.tif')




