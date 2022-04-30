from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation
import pandas as pd
import time


def read_csv_data_from_folder(folder_path):
    pass


def normalize_data(data):
    pass


def run_classifier(X, Y, classifier):
    classifiers = {"LinearSVM": "",
                   "KNN": "",
                   "NB": "",
                   "ANN": ""}

    # return acc


def run_fs_method(data, fs, k, classifier):
    fs_methods = {"ls": lap_score.lap_score,
                  "fcbf": FCBF.fcbf,
                  "mrmr": MRMR.mrmr,
                  "mim": MIM.mim,
                  "skewness": "",
                  "variance": ""}
    # for i in range(k)
    # calculate time and acc for fs method
    total_acc = 0
    total_time = 0
    y = data.iloc[:, -1:]
    data_values = data.values.astype(float)
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": k, 't': 1}
    W = construct_W.construct_W(data, **kwargs_W)

    for i in range(1, k + 1):
        start = time.time()
        kwargs = {"X": data_values, "y": y, "W": W, "n_selected_features": i}
        score = fs_methods[fs](**kwargs)
        ranking = list(score)[:i]
        end = time.time()
        data_cut = data.iloc[:, ranking]
        data_cut.columns = [''] * len(data_cut.columns)
        accuracy = run_classifier(data_cut, y, classifier)
        total_acc += accuracy
        total_time += end - start

    return total_acc / k, total_time / k


def run_simulations(k, classifier, folder_path):
    fs_methods = ["ls", "fcbf", "mrmr", "mim", "skewness", "variance"]
    acc_results = pd.DataFrame(columns="Dataset" + fs_methods)
    time_results = pd.DataFrame(columns="Dataset" + fs_methods)

    for data_name, data in read_csv_data_from_folder(folder_path).items():
        normalize_data(data)
        data_acc_res = {"Dataset": data_name}
        data_time_res = {"Dataset": data_name}

        for fs in fs_methods:
            avc_acc, avg_time = run_fs_method(data, fs, k, classifier)
            data_acc_res[fs] = avc_acc
            data_time_res[fs] = avg_time

        acc_results.append(data_acc_res)
        time_results.append(data_time_res)


def main():
    run_simulations(5, "LinearSVM", "")
