import scipy
from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from scipy.stats import kurtosis, skew
import operator
import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

classifiers = {"LinearSVM": LinearSVC(random_state=0, tol=1e-5, max_iter=10000),
               "KNN": KNeighborsClassifier(n_neighbors=3),
               "NB": GaussianNB(),
               "ANN": MLPClassifier(random_state=1, max_iter=300)}


def find_csv_files_in_folder(folder_path):
    """
    Read data from folder that contains csv file
    """
    dataframes = {}
    print(f"Looking for csv files in {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_name = file.split('.csv')[0]
                csv_file = os.path.join(root, file)
                print(f"Found [{file_name}] dataset")
                dataframes[file_name] = csv_file
    return dataframes


def normalize_data(data):
    """
    Normalize data using min_max scalar
    """
    features = list(data.columns)
    scaler = MinMaxScaler()
    scaler.fit(data[features[0:len(features)-1]])
    return scaler.transform(data[features[0:len(features)-1]])


def run_classifier(classifier, X, y):
    """
    Run the classifier
    """
    clf = classifiers[classifier]
    clf_cv = cross_val_score(clf, X, y, cv=10)
    return clf_cv.mean()


def skewness(X, **kwargs):
    """
    Calculate the skewness
    """
    skew = {}
    i = 0
    skewness = pd.DataFrame()
    skewness['Skew'] = scipy.stats.skew(X, axis=0)
    skewi = skewness['Skew']
    for item in skewi:
        skew[i] = item
        i = i + 1
    sorted_i1 = sorted(skew.items(), key=operator.itemgetter(1), reverse=True)
    sorted_keys = [item[0] for item in sorted_i1]
    return sorted_keys


def variance(X, **kwargs):
    """
    Calculate the variance
    """
    var = {}
    i = 0
    variance = pd.DataFrame()
    variance['Var'] = X.var(axis=0)
    vari = variance['Var']
    for item in vari:
        var[i] = item
        i = i + 1
    sorted_i1 = sorted(var.items(), key=operator.itemgetter(1), reverse=True)
    sorted_keys = [item[0] for item in sorted_i1]
    return sorted_keys


def run_fs_method(data, fs, k, classifier):
    """
    Calculate the average run time and accuracy for feature selection method
    """
    fs_methods = {"ls": lap_score.lap_score,
                  "fcbf": FCBF.fcbf,
                  "mrmr": MRMR.mrmr,
                  "mim": MIM.mim,
                  "skewness": skewness,
                  "variance": variance}

    total_acc = 0
    total_time = 0
    y = np.ravel(data.iloc[:, -1:])
    X = data.iloc[:, :-1]
    X_values = X.values.astype(float)
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": k, 't': 1}
    try:
        W = construct_W.construct_W(X, **kwargs_W)
    except MemoryError:
        print("Memory error skipping fs method.")
        return 0, 0

    # Running evaluation:
    for i in range(1, k + 1):
        start = time.time()
        kwargs = {"X": X_values, "y": y, "n_selected_features": i, "W": W}
        try:
            score = fs_methods[fs](**kwargs)
        except MemoryError:
            print("Memory error skipping fs method.")
            return 0, 0
        ranking = list(score)[:i]
        time_passed = time.time() - start
        data_cut = X.iloc[:, ranking]
        data_cut.columns = [''] * len(data_cut.columns)
        print("Running classifier")
        accuracy = run_classifier(classifier, data_cut, y)
        print(f"Accuracy {accuracy}, Time {time_passed}")
        total_acc += accuracy
        total_time += time_passed

    return total_acc / k, total_time / k


def run_simulations(k, classifier, folder_path):
    """
    Main function for running the program
    """
    fs_methods = ["LS", "MRMR", "FCBF", "MIM", "Skewness", "Variance"]
    acc_results = pd.DataFrame(columns=["Dataset"] + fs_methods)
    time_results = pd.DataFrame(columns=["Dataset"] + fs_methods)

    for data_name, data_path in find_csv_files_in_folder(folder_path).items():
        print(f"Loading [{data_name}]")
        try:
            data = pd.read_csv(data_path)
        except MemoryError:
            print("Memory error, skipping dataset.")
            continue
        print("Normalizing Data")
        normalize_data(data)
        data_acc_res = {"Dataset": data_name}
        data_time_res = {"Dataset": data_name}

        print("Begin feature selection evaluations")
        for fs in fs_methods:
            print(f"Evaluating {fs}")
            avc_acc, avg_time = run_fs_method(data, fs.lower(), k, classifier)
            print(f"Finished {fs} evaluation.")
            data_acc_res[fs] = avc_acc
            data_time_res[fs] = avg_time

        acc_results = acc_results.append(data_acc_res, ignore_index=True)
        time_results = time_results.append(data_time_res, ignore_index=True)

    print("Final accuracy results table:")
    print(acc_results)
    acc_results.to_csv(f'{classifier}_acc_results.csv')

    print("Final cpu time results table:")
    print(time_results)
    time_results.to_csv(f'{classifier}_time_results.csv')


def main():
    for c in classifiers:
        print(f"Running {c} model")
        run_simulations(5, c, r"C:\Users\Daniel\Desktop\Fires Dataset-20220425T153127Z-001\Fires Dataset\datasets\a")


if __name__ == "__main__":
    main()
