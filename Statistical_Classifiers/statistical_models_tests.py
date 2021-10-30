from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as cPickle
import numpy as np
import random
import time
import sys
import os

# get all the classifiers we want to experiment with
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def print_to_file(this_str, file_name):
    print(this_str)
    print(this_str, file=open(os.path.join("E:/Forest Cover - Redo 2020/Trainings and Results/Training Data/Clipped dataset/"
                                           "statistical_models_dataset/15-Districts/", file_name), "a"))


def train_and_test_statistical_model(name, classifier, x_train, y_train, x_test, y_test, process_name):
    # fit the model on your dataset
    trained_classifier = classifier.fit(x_train, y_train)
    # get predictions on unseen data
    y_pred = trained_classifier.predict(x_test)
    # get an accuracy score please
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    # get confusion matrix
    confusion_matrix_to_print = confusion_matrix(y_test, y_pred)
    # show confusion matrix and classification report for precision, recall, F1-score
    print_to_file("################################ {} ################################".format(name), file_name=f"{process_name}.txt")
    print_to_file('Model Accuracy: {:.2f}%'.format(100*accuracy), file_name=f"{process_name}.txt")
    print_to_file('Confusion Matrix', file_name=f"{process_name}.txt")
    print_to_file(confusion_matrix_to_print, file_name=f"{process_name}.txt")
    print_to_file('Classification Report', file_name=f"{process_name}.txt")
    print_to_file(classification_report(y_test, y_pred, target_names=['Non-Forest', 'Forest']), file_name=f"{process_name}.txt")
    return trained_classifier


def load_or_create_dataset(this_dataset):
    assert this_dataset == "100K" or this_dataset == "1M"
    raw_dataset_path = "E:/Forest Cover - Redo 2020/Trainings and Results/Training Data/Clipped dataset/Pickled_data/"
    processed_dataset_path = f"E:/Forest Cover - Redo 2020/Trainings and Results/Training Data/Clipped dataset/statistical_models_dataset/15-Districts/" \
                             f"{this_dataset}_dataset.pkl"
    # prepare data if it doesn't already exist
    if os.path.exists(processed_dataset_path):
        print("(LOG): Found Precompiled Serialized Dataset...")
        with open(processed_dataset_path, 'rb') as processed_dataset:
            (datapoints_as_array, labels_as_array) = cPickle.load(processed_dataset)
        print("(LOG): Dataset Size: Datapoints = {}; Ground Truth Labels {}".format(datapoints_as_array.shape, labels_as_array.shape))
        print("(LOG): Loaded Precompiled Serialized Dataset Successfully!")
    else:
        print("(LOG): No Precompiled Dataset Found! Creating New Dataset Now...")
        all_pickle_files_in_pickled_dataset = os.listdir(raw_dataset_path)
        datapoints_as_array, labels_as_array = np.empty(shape=[1, 18]), np.empty(shape=[1, ])
        # random.seed(datetime.now())
        np.random.seed(232)
        num_samples = 1500
        if this_dataset == "100K":
            num_samples /= 10
        for idx, this_pickled_file in enumerate(all_pickle_files_in_pickled_dataset):
            district_name_first_part = this_pickled_file.split('_')[0]
            if district_name_first_part == 'upper' or district_name_first_part == 'chitral':
                print("[LOG] Skipping {}".format(this_pickled_file))
                continue
            full_data_sample_path = os.path.join(raw_dataset_path, this_pickled_file)
            if idx % 100 == 0:
                print("(LOG): Processing ({}/{}) => {}".format(idx, len(all_pickle_files_in_pickled_dataset), full_data_sample_path))
            with open(full_data_sample_path, 'rb') as this_small_data_sample:
                small_image_sample, small_label_sample = cPickle.load(this_small_data_sample, encoding='bytes')
                this_shape = small_image_sample.shape
                random_rows, random_cols = np.random.randint(0, this_shape[0], size=num_samples), np.random.randint(0, this_shape[0], size=num_samples)
                sample_datapoints = np.nan_to_num(small_image_sample[random_rows, random_cols, :])
                sample_labels = np.nan_to_num(small_label_sample[random_rows, random_cols])
                # pick only valid (not NULL) pixels
                valid_samples = (sample_labels != 0)
                sample_datapoints = sample_datapoints[valid_samples]
                sample_labels = sample_labels[valid_samples]
                # apply the following code if you want 18 bands in your sample points
                # get more indices to add to the example, landsat-8
                ndvi_band = (sample_datapoints[:, 4] - sample_datapoints[:, 3]) / (sample_datapoints[:, 4] + sample_datapoints[:, 3] + 1e-3)
                evi_band = 2.5 * (sample_datapoints[:, 4] - sample_datapoints[:, 3]) / (
                            sample_datapoints[:, 4] + 6 * sample_datapoints[:, 3] - 7.5 * sample_datapoints[:, 1] + 1)
                savi_band = 1.5 * (sample_datapoints[:, 4] - sample_datapoints[:, 3]) / (sample_datapoints[:, 4] + sample_datapoints[:, 3] + 0.5)
                msavi_band = 0.5 * (2 * sample_datapoints[:, 4] + 1 - np.sqrt(
                    (2 * sample_datapoints[:, 4] + 1) ** 2 - 8 * (sample_datapoints[:, 4] - sample_datapoints[:, 3])))
                ndmi_band = (sample_datapoints[:, 4] - sample_datapoints[:, 5]) / (sample_datapoints[:, 4] + sample_datapoints[:, 5] + 1e-3)
                nbr_band = (sample_datapoints[:, 4] - sample_datapoints[:, 6]) / (sample_datapoints[:, 4] + sample_datapoints[:, 6] + 1e-3)
                nbr2_band = (sample_datapoints[:, 5] - sample_datapoints[:, 6]) / (sample_datapoints[:, 5] + sample_datapoints[:, 6] + 1e-3)
                sample_datapoints = np.concatenate((sample_datapoints, np.expand_dims(ndvi_band, axis=1)), axis=1)
                sample_datapoints = np.concatenate((sample_datapoints, np.expand_dims(evi_band, axis=1)), axis=1)
                sample_datapoints = np.concatenate((sample_datapoints, np.expand_dims(savi_band, axis=1)), axis=1)
                sample_datapoints = np.concatenate((sample_datapoints, np.expand_dims(msavi_band, axis=1)), axis=1)
                sample_datapoints = np.concatenate((sample_datapoints, np.expand_dims(ndmi_band, axis=1)), axis=1)
                sample_datapoints = np.concatenate((sample_datapoints, np.expand_dims(nbr_band, axis=1)), axis=1)
                sample_datapoints = np.concatenate((sample_datapoints, np.expand_dims(nbr2_band, axis=1)), axis=1)
            datapoints_as_array = np.concatenate((datapoints_as_array, sample_datapoints), axis=0)
            labels_as_array = np.concatenate((labels_as_array, sample_labels), axis=0)
        # at this point, we just serialize the arrays and save them
        with open(processed_dataset_path, 'wb') as processed_dataset:
            cPickle.dump((datapoints_as_array, labels_as_array), processed_dataset)
        print("(LOG): Dataset Size: Datapoints = {}; Ground Truth Labels {}".format(datapoints_as_array.shape, labels_as_array.shape))
        print("(LOG): Compiled and Serialized New Dataset Successfully!")
    # fix before return
    labels_as_array[labels_as_array == 0] = 1
    labels_as_array -= 1  # all labels should be 0 or 1 (non-Forest, Forest)
    return datapoints_as_array, labels_as_array


def train_stat_model(this_dataset, this_model, these_bands, class_1_weight, class_2_weight, datapoints_as_array, labels_as_array, process_name):
    model_path = f"E:/Forest Cover - Redo 2020/Trainings and Results/Training Data/Clipped dataset/statistical_models_dataset/15-Districts/" \
                 f"{process_name}-{these_bands}.pkl"
    assert this_model == "SVC" or this_model == "Perceptron" or this_model == "GaussianNB" or this_model == "LogisticRegression" or \
           this_model == "DecisionTreeClassifier" or this_model == "RandomForestClassifier"
    assert these_bands == "rgb" or these_bands == "full-spectrum" or these_bands == "augmented" or these_bands == "extended"
    print_to_file(f"(LOG): Parameters: {this_dataset} dataset with {this_model} and {these_bands} bands using C1W: {class_1_weight} and C2W: "
                  f"{class_2_weight}", file_name=f"{process_name}.txt")
    print_to_file(f"(LOG): Will save trained model @ {model_path}", file_name=f"{process_name}.txt")
    # raw_dataset_path = "/home/azulfiqar_bee15seecs/training_data/pickled_clipped_training_data"
    # processed_dataset_path = "/home/azulfiqar_bee15seecs/training_data/statistical_models_dataset/1M_dataset.pkl"
    # model_path = "/home/azulfiqar_bee15seecs/training_data/statistical_models_dataset/logistic_regressor.pkl"
    bands_lists = {"rgb": [3, 2, 1], "full-spectrum": [*range(11)], "augmented": [*range(18)], "extended": [*range(11, 18)]}
    # get your model (RandomForestClassifier, DecisionTreeClassifier, SVC, GaussianNB, LogisticRegression, Perceptron)
    classifiers = {
        "SVC": SVC(verbose=1, class_weight={0: class_1_weight, 1: class_2_weight}),
        "Perceptron": Perceptron(verbose=1, n_jobs=4, class_weight={0: class_1_weight, 1: class_2_weight}),
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(verbose=1, n_jobs=4, max_iter=1000, solver='lbfgs', class_weight={0: class_1_weight, 1: class_2_weight}),
        "DecisionTreeClassifier": DecisionTreeClassifier(class_weight={0: class_1_weight, 1: class_2_weight}),
        "RandomForestClassifier": RandomForestClassifier(verbose=1, n_jobs=4, class_weight={0: class_1_weight, 1: class_2_weight}),
    }
    # create training and testing arrays from loaded data
    total_datapoints = len(datapoints_as_array)
    split = int(0.8*total_datapoints)
    # 1:4 implies RGB Model
    x_train, y_train = datapoints_as_array[:split, bands_lists[these_bands]], labels_as_array[:split].astype(np.uint8)
    x_test, y_test = datapoints_as_array[split:, bands_lists[these_bands]], labels_as_array[split:].astype(np.uint8)
    print_to_file("(LOG): Dataset for Training and Testing Prepared", file_name=f"{process_name}.txt")
    print_to_file("(LOG): Training Data: {}; Testing Data: {}".format(x_train.shape, x_test.shape), file_name=f"{process_name}.txt")
    # call model for training
    trained_classifier = train_and_test_statistical_model(name=f"{this_model}-{these_bands}", classifier=classifiers[this_model],
                                                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, process_name=process_name)
    with open(model_path, 'wb') as model_file:
        print_to_file(cPickle.dump(trained_classifier, model_file), file_name=f"{process_name}.txt")
    print_to_file("(LOG): Saved Trained Classifier as {}".format(model_path), file_name=f"{process_name}.txt")
    pass


if __name__ == "__main__":
    # this_dataset = sys.argv[1]  # "100K"
    # this_model = sys.argv[2]  # "RandomForestClassifier"
    # these_bands = sys.argv[3]
    # class_1_weight, class_2_weight = float(sys.argv[4]), float(sys.argv[5])
    classifiers = ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"]
    band_combinations = ["rgb", "full-spectrum", "augmented", "extended"]
    this_dataset, class_1_weight, class_2_weight = "1M", 1, 1
    datapoints, labels = load_or_create_dataset(this_dataset)
    destination_folder = "E:\\Forest Cover - Redo 2020\\Trainings and Results\\Training Data\\Clipped dataset\\statistical_models_dataset"
    for this_model in classifiers:
        for these_bands in band_combinations:
            process_name = f"{this_model}_{this_dataset}_C1W_{class_1_weight}_C2W_{class_2_weight}"
            train_stat_model(this_dataset, this_model, these_bands, class_1_weight, class_2_weight, datapoints_as_array=datapoints, labels_as_array=labels,
                             process_name=process_name)
            pass
        pass
    pass