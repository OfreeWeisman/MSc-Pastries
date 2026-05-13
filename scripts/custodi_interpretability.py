import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.commons import read_raw_columns
import settings

def ngram_counter(strings, degree):
    """Method to count average number of occurances for ngram each in a set of strings (total number / len(strings)).
    RETURNS (dict) dictionary with strings (characters) and average counts"""
    char_dict = {}
    for string in strings:
        for idx in range(len(string)):
            for i in range(degree):
                try:
                    a = string[idx:(idx + i + 1)]
                    if len(a) == i + 1:
                        if a in char_dict:
                            char_dict[a] += 1
                        else:
                            char_dict[a] = 1
                except IndexError:
                    pass
    return {k: v / len(strings) for k, v in char_dict.items()}
    
def calc_weights(property, custodi_dict, custodi_intercept, ngram_dict):
    normalized_weights = np.array([custodi_dict[k] * ngram_dict[k] for k in custodi_dict.keys()] + [custodi_intercept])
    normalized_weights = normalized_weights / (np.sum(normalized_weights) + custodi_intercept)
    normalized_weights_d = {k: x for k, x in zip(list(custodi_dict.keys()) + ["intercept"], normalized_weights)}
    return normalized_weights_d
    
def plot_ngram_weights(property, normalized_weights_d, plot_th, color_d):
    """Method to plot the weight of each ngram for a given custodi model dictionary"""
    norm_d = {k: v for k, v in normalized_weights_d.items() if v > plot_th}
    s = 0
    for char, weight in norm_d.items():
        plt.bar(property, weight, bottom=s, label=char, color=color_d[char])
        s += weight
        
def read_custodi_data(property, test_size, base_str, rep):
    """Method to read CUSTODI model information (dictionary + intercept) for given property and base string (LALA, SMILES)"""
    path = "../results/models/{}/{}/custodi_model_{}_str_rep/{}_model/tok_parameters.json".format(test_size, property.lower(), base_str.lower(), str(rep))
    if not os.path.isfile(path):
        return None, None
    with open(path, "r") as f:
        d = json.load(f)
        return d["dictionary"], d["intercept"]
        
def get_ngram_dict(base_str, degree):
    if base_str.lower() == "smiles":
        return ngram_counter(read_raw_columns(["smiles"])["smiles"].values, degree)
    elif base_str.lower() == "lala":
        return ngram_counter(read_raw_columns(["lalas"])["lalas"].values, degree)
    elif base_str.lower() == "augmented_lala":
        return ngram_counter(read_raw_columns(["augmented_lalas"])["augmented_lalas"].values, degree)
        
def read_property_data(property, test_size, base_str, plot_th):
    df = pd.DataFrame()
    for rep in range(1, 6):
        custodi_d, custodi_intercept = read_custodi_data(property, test_size, base_str, rep)
        if custodi_d:
            degree = max([len(s) for s in custodi_d])
            ngram_d = get_ngram_dict(base_str, degree)
            color_d = make_color_palet(ngram_d)
            norm_d = calc_weights(property, custodi_d, custodi_intercept, ngram_d)
            df = df.append(norm_d, ignore_index=True)
    # calculating aggregations
    n = len(df)
    avg = df.mean()
    std = 1 / (n - 1) * ((df - avg)**2).sum()
    # formatting df
    return pd.DataFrame(
          {
              "property": [property for _ in range(len(df.columns))],
              "str": df.columns,
              "n": [n for _ in range(len(df.columns))],
              "avg": avg.values,
              "var": std.values
        }
    )
    
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def make_color_palet(ngram_dict):
    """Make colors for plotting"""
    np.random.seed(0)
    color_d = {k: np.random.rand(3) for k in ngram_dict.keys()}
    color_d["intercept"] = np.random.rand(3)
    return color_d 


if __name__ == "__main__":
    for base_str in ["lala", "augmented_lala", "smiles"]:
        plt.figure(figsize=(12, 8))
        res = pd.DataFrame()
        for property in settings.properties:
            print("running", base_str, "with", property)
            df = read_property_data(property, 1000, base_str, 0.01)
            res = pd.concat([res, df])
        res.to_csv("../results/feature_importance/custodi_{}.csv".format(base_str))
        #plt.title(base_str)
        #legend_without_duplicate_labels(plt.gca())
        #plt.savefig("../results/feature_importance/custodi_{}.png".format(base_str))