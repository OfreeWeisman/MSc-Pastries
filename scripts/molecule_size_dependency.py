"""Script to checkt the error dependence on the molecule size"""
import os
import matplotlib.pyplot as plt
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.models import BaseEstimator
from src.core.representations import SmilesString
from src.commons import make_data, split_train_test, unnormalize
import settings


def apply_func_on_smiles(func):
    smiles, _ = make_data("gap_ev", SmilesString())
    res = []
    for smile in smiles:
        res.append(func(smile))
    return res
    
def num_atoms(string):
    return string.count("c")

def calc_str_lens(test_size):
    x_train, x_test, y_train, y_test = split_train_test(apply_func_on_smiles(num_atoms), apply_func_on_smiles(num_atoms), test_size)
    return y_test
    
def get_error_measure(true, pred, func):
    res = []
    for t, p in zip(true, pred):
        res.append(func(t, p))
    return res

def abs_diff(t, p):
    return abs(t - p)[0]


def run(model, rep, property, test_size):
    if property in settings.properties:
        X, y = make_data(property.lower(), rep)
    # error if not valid property
    else:
        raise ValueError("Unrecognized property {}".format(property))
    # splits to train / test
    x_train, x_test, y_train, y_test = split_train_test(X, y, test_size)
    y_pred = unnormalize(model.predict(x_test), y_test)
    return get_error_measure(y_pred, y_test, abs_diff)
    

def parse_results_dir(target_dir):
    property = os.path.split(os.path.dirname(os.path.dirname(target_dir)))[-1]
    test_size = int(os.path.split(os.path.dirname(os.path.dirname(os.path.dirname(target_dir))))[-1])
    for model_name in settings.models_dict:
        if model_name in target_dir:
            model = settings.models_dict[model_name]
            model = model.load(os.path.join(target_dir, "model"))
    for rep_name in settings.reps_dict:
        if rep_name in target_dir:
            rep = settings.reps_dict[rep_name]
    return model, rep, property, test_size
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check molecule size dependency for given model")
    parser.add_argument("model_path", type=str, help="results directory of the model")
    parser.add_argument("image_path", type=str, help="path for image")
    args = parser.parse_args()
    model, rep, property, test_size = parse_results_dir(args.model_path)
    errs = run(model, rep, property, test_size)
    sizes = calc_str_lens(test_size)
    print(len(errs), len(sizes))
    plt.scatter(sizes, errs, alpha=0.3)
    plt.savefig(args.image_path)
    

    