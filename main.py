from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
import pandas as pd

def read_csv_data_from_folder(folder_path) -> pd.DataFrame:
    pass


def normalize_data(data):
    pass


def run_classifier(classifier):
    classifiers = {"LinearSVM": "",
                   "KNN": "",
                   "NB": "",
                   "ANN": ""}

    # return acc


def run_fs_method(fs, k, classifier):
    fs_methods = {"ls": "",
                  "fcbf": "",
                  "mrmr": "",
                  "mim": "",
                  "skewness": "",
                  "variance": ""}
    # for i in range(k)
    # calculate time and acc for fs method
    return avc_acc, avg_time


def run_simulations(k, classifier, folder_path):
    df = read_csv_data_from_folder(folder_path)
    normalize_data(df)
    fs_mthods = []

    results = pd.DataFrame()
    for fs in fs_mthods:
        avc_acc, avg_time = run_fs_method(fs, k, classifier)


def main():
    run_simulations(5, "LinearSVM", "")