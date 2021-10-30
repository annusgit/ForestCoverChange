

from __future__ import print_function
from __future__ import division
import sys
import pickle
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def confusion(file_path):
    # all_labels = ['{}'.format(x) for x in range(3)]
    all_labels = ['noise', 'forest', 'not-forest']
    with open(file_path, 'rb') as this:
        matrix = pickle.load(this)
    print(matrix)

    # matrix = [[9.99424875e-01, 0.00000000e+00, 5.75103273e-04],
    #           [0.00000000e+00, 8.61850142e-01, 1.38149858e-01],
    #           [8.33658269e-05, 2.07877174e-01, 7.92039454e-01]]

    # df_cm = pd.DataFrame(matrix, index=[i for i in all_labels],
    #                      columns=[i for i in all_labels])
    # plt.figure(figsize=(10,7))
    # sn.heatmap(df_cm, cmap='BuPu', annot=True)
    # plt.show()


if __name__ == '__main__':
    confusion(file_path=sys.argv[1])
