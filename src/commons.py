from typing import List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os


# data handling utils

def normalize(data: np.array, params=None):
    """Method to make a z-score normalization for data vectors.
    ARGS:
        - data (np.array): data on input batches to normalize. normalizes each batch.
        - params (list): list of [mean, std] to use for calculation"""
    if params is None:
        m = np.mean(data, axis=0)
        s = np.std(data, axis=0)
    else:
        m = params[0]
        s = params[1]
    return (data - m) / s


def unnormalize(data: np.ndarray, ref_data: np.ndarray):
    """Method to undo a z-score normalization, given reference population data"""
    m = np.mean(ref_data, axis=0)
    s = np.std(ref_data, axis=0)
    return data * s + m


def read_raw_columns(columns: List[str]):
    """Method to get raw data from specific columns"""
    df = pd.read_csv("../data/all_data.csv", index_col=0)
    df.columns = [c.lower().strip() for c in df.columns]
    return df.loc[:, columns]


def make_data(property: str, representation):
    """Method to read the data into x, y vectors.
    ARGS:
        - property (str): name of calculated property
        - representation (Representation): representation to use for the data prep
    RETURNS:
        (tuple) X, y for the fit"""
    df = read_raw_columns([property])
    if not "etot" in property.lower():
        return np.array(representation.represent(df.index)), df.values
    else:
        # doing size normalization for total energy properties
        return np.array(representation.represent(df.index)), df.values / (30 + read_raw_columns(["n_rings"]).values * 18)


def split_train_test(X, y, test_size: int, random_seed=1):
    org_state = np.random.get_state()
    # setting seed for uniform results
    np.random.seed(random_seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # returning to original random state (to keep consistancy with other subroutines)
    np.random.set_state(org_state)
    return x_train, x_test, y_train, y_test
    
def bootstrap_idxs(n, n_bootstraps, n_test):
    idxs = range(n)
    res = []
    for _ in range(n_bootstraps):
      train = resample(idxs, replace=True, n_samples=(len(idxs)-n_test))
      test = resample([x for x in idxs if not x in train], replace=True, n_samples=n_test)
      res.append((train, test))
    return res 
    
def bootstrap_data(X, y, n_bootstraps, test_size, random_seed=1):
    org_state = np.random.get_state()
    # setting seed for uniform results
    np.random.seed(random_seed)
    res = []
    bs = bootstrap_idxs(len(y), n_bootstraps, test_size)
    for train_idxs, test_idxs in bs:
        x_train = np.array([X[i] for i in train_idxs])
        x_test = np.array([X[i] for i in test_idxs])
        y_train = np.array([y[i] for i in train_idxs])
        y_test = np.array([y[i] for i in test_idxs])
        res.append((x_train, x_test, y_train, y_test))
    # returning to original random state (to keep consistancy with other subroutines)
    np.random.set_state(org_state)
    return res
    

# hyperparameter optimization

def run_bayes_cv(model, search_space: dict, X, y, cv: int, n_iter: int, n_jobs: int):
    """Method to run a bayesian hyeperparameter optimization (using n-fold cross-validation) on a model type, given the search space.
    RETURNS:
        (Estimator) trained model (on X and y) with best hyperparameters"""
    # setting up optimizer
    searchcv = BayesSearchCV(
        model,
        search_space,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        verbose=4,
        random_state=0
    )
    # running bayesian optimization (with normalized y)
    searchcv.fit(X, normalize(y))
    # returning optimized model, retrained on ALL train data (no cross-validation)
    return searchcv.best_estimator_


def analyze_model(model, x_train, y_train, x_test, y_test, results_directory: str, prefix: str="", **plot_kwargs):
    """Method to analyze fit of an estimator and save it in a target results directory. saves the following data
        - fit scores on both train and test sets
        - fit plots for train and test sets
        - saves best model (parameters and other information, using save method of each model)"""
    # makes results directory
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)
    # saving model
    model_dir = os.path.join(results_directory, prefix + "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # using model.save method
    model.save(model_dir)
    # estimating fit
    res_dict = pd.DataFrame({"train": estimate_fit(model, x_train, y_train),
                             "test": estimate_fit(model, x_test, y_test)})
    # saving data
    res_dict.to_csv(os.path.join(results_directory, prefix + "error_metrics.csv"))
    # plotting fit
    plot_fit(model, x_train, y_train, title="Train set", **plot_kwargs)
    plt.savefig(os.path.join(results_directory, "Train.png"))
    plt.close()
    plot_fit(model, x_test, y_test, title="Test set", **plot_kwargs)
    plt.savefig(os.path.join(results_directory, "Test.png"))
    plt.close()


# estimate fit utils.

def _safe_calc_sum_of_binary_func(pred, true, func) -> float:
    """Method to calculate sum of binary function values on two vectors in a memory-safe way"""
    s = 0
    for p, t in zip(pred, true):
        val = func(p, t)
        if not val == [np.inf] and not val == np.inf:
            s = s + val
    return s


def calc_rmse(pred, true) -> float:
    f = lambda p, t: np.square(p - t)
    return np.sqrt(_safe_calc_sum_of_binary_func(pred, true, f) / len(pred))


def calc_mae(pred, true) -> float:
    f = lambda p, t: np.abs(p - t)
    return _safe_calc_sum_of_binary_func(pred, true, f) / len(pred)


def calc_mare(pred, true) -> float:
    f = lambda p, t: np.abs((p - t) / t) if not t == 0 else 0
    return _safe_calc_sum_of_binary_func(pred, true, f)/ len(pred)


def calc_r_squared(pred, true) -> float:
    avg_t = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: t) / len(true)
    avg_p = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: p) / len(pred)
    var_t = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: np.square(t - avg_t)) / len(true)
    var_p = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: np.square(p - avg_p)) / len(pred)
    cov = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: (t - avg_t) * (p - avg_p)) / len(true)
    return cov**2 / (var_p * var_t)


def estimate_fit(model, X, y, prefix="", unnormalize_y=True) -> dict:
    pred = model.predict(X)
    if unnormalize_y:
        pred = unnormalize(pred, y)
    return {
        prefix + "rmse": calc_rmse(pred, y)[0],
        prefix + "mae": calc_mae(pred, y)[0],
        prefix + "mare": calc_mare(pred, y)[0],
        prefix + "r_squared": calc_r_squared(pred, y)[0]
    }

def plot_fit(model, X, y, unnormalize_y=True, title="", **plot_kwargs):
    pred = model.predict(X)
    if unnormalize_y:
        pred = unnormalize(pred, y)
    plt.figure()
    plt.title(title)
    plt.scatter(pred, y, **plot_kwargs)
    plt.xlabel("Predicted")
    plt.ylabel("True")