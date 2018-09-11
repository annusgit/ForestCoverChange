

from __future__ import print_function
from __future__ import division
import sys
import pickle
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def confusion(file_path):
    all_labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

    with open(file_path, 'rb') as this:
        matrix = pickle.load(this)

    df_cm = pd.DataFrame(matrix, index=[i for i in all_labels],
                         columns=[i for i in all_labels])
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, cmap='BuPu', annot=True)
    plt.show()


if __name__ == '__main__':
    confusion(file_path=sys.argv[1])
