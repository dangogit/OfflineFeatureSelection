# from skfeature.function.information_theoretical_based import FCBF
# from skfeature.function.information_theoretical_based import MRMR
# from skfeature.function.information_theoretical_based import MIM
# from skfeature.function.similarity_based import lap_score
# from skfeature.utility import construct_W
# from skfeature.utility import unsupervised_evaluation
import pandas as pd
import time
import glob
import os

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.svm import LinearSVC

classifiers = {"LinearSVM": LinearSVC(random_state=0, tol=1e-5,max_iter=10000),
                "KNN": KNeighborsClassifier(n_neighbors=3),
                "NB": GaussianNB(),
                "ANN": MLPClassifier(random_state=1, max_iter=300)}

def read_csv_data_from_folder(folder_path):
    dataframes_list = []
    csv_files = list(filter(lambda f: f.endswith('.csv'), os.listdir(folder_path)))
    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file)
        dataframes_list.append(temp_df)
    return dataframes_list


def normalize_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def run_classifier(classifier, X, y):
    clf = classifiers[classifier]
    clf_cv = cross_val_score(clf, X, y, cv=10)
    print("=== Mean accuracy  ===")
    print("Mean accuracy Score - {}: ".format(classifier), clf_cv.mean())
    return clf_cv.mean()


def run_fs_method(fs, k, classifier):
    fs_methods = {"ls": "",
                  "fcbf": "",
                  "mrmr": "",
                  "mim": "",
                  "skewness": "",
                  "variance": ""}
    # for i in range(k)
    # calculate time and acc for fs method
    # return avc_acc, avg_time


def run_simulations(k, classifier, folder_path):
    df = read_csv_data_from_folder(folder_path)
    normalize_data(df)
    fs_mthods = []

    results = pd.DataFrame()
    for fs in fs_mthods:
        avc_acc, avg_time = run_fs_method(fs, k, classifier)


def main():
    run_simulations(5, "LinearSVM", "")

if __name__ == "__main__":
    pass