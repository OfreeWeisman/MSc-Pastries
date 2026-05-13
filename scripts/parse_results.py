import pandas as pd
import numpy as np
import json
import os

model_rep_pairs = [
          ("rf_model", "lala_features_rep"),
          ("rf_model", "augmented_lala_features_rep"),
          ("rf_model", "lala_features_rep_no_ratio"),
          ("rf_model", "augmented_lala_features_rep_no_ratio"),
          ("rf_model_custodi_rep", "augmented_lala_str_rep"),
          ("rf_model_custodi_rep", "lala_str_rep"),
          ("rf_model_custodi_rep", "smiles_str_rep"),
          ("krr_model","lala_features_rep"),
          ("krr_model","augmented_lala_features_rep"),
          ("krr_model_custodi_rep", "lala_str_rep"),
          ("krr_model_custodi_rep", "augmented_lala_str_rep"),
          ("krr_model_custodi_rep", "smiles_str_rep"),
          ("custodi_model", "lala_str_rep"),
          ("custodi_model", "augmented_lala_str_rep"),
          ("custodi_model", "smiles_str_rep"),
          ("rnn", "lala_one_hot_rep"),
          ("rnn", "augmented_lala_one_hot_rep"), 
          ("rnn", "smiles_one_hot_rep"),
          ("graph_conv", "mol_conv_rep")
        ]

def parse_results_dir_str(path):
    """Method to parse the results directory path into property, model and representation.
    RETURNS: dictionary with property, model name and represntation name"""
    property = os.path.split(os.path.dirname(path))[-1]
    for model, rep in model_rep_pairs:
        if "{}_{}".format(model, rep) == os.path.split(path)[-1]:
            return {"property": property, "model": model, "representation": rep}
            
def calc_aggreagations(vals, prefix):
    vals = np.array(vals)
    avg = np.mean(vals)
    n = len(vals)
    var = 1 / (n - 1) * np.sum(np.square(vals - avg)) if n > 1 else 0
    return {
        prefix + "avg": avg,
        prefix + "var": var,
        prefix + "n": n,
    }
            
def unite_all_results_to_csv(parent_res_dir, error_measure):
    main_df = pd.DataFrame()
    if error_measure in ["mae", "rmse"]:
        norm_factors = {
                          "homo_ev": 0.179,
                        	"lumo_ev": 0.208,
                          "gap_ev": 0.384,
                          "rel_etot_ev": 0.413,
                          "aea_ev": 0.210,
                          "aip_ev": 0.182,
                      }
    else:
        norm_factors = {
                          "homo_ev": 1,
                        	"lumo_ev": 1,
                          "gap_ev": 1,
                          "rel_etot_ev": 1,
                          "aea_ev": 1,
                          "aip_ev": 1,
                      }
    all_train_vals = {}
    all_test_vals = {}
    for prop_dir in os.listdir(parent_res_dir):
        if not "test" in prop_dir and not ".csv" in prop_dir:
            if os.path.isdir(os.path.join(parent_res_dir, prop_dir)):
                for res_dir in os.listdir(os.path.join(parent_res_dir, prop_dir)):
                    full_res_dir = os.path.join(parent_res_dir, prop_dir, res_dir)
                    print(full_res_dir)
                    res_d = parse_results_dir_str(full_res_dir)
                    train_vals = []
                    test_vals = []
                    if os.path.isdir(full_res_dir):
                        for csv in os.listdir(full_res_dir):
                            if csv.endswith(".csv"):
                                try:
                                    res_df = pd.read_csv(os.path.join(full_res_dir, csv), index_col="Unnamed: 0")
                                except pd.errors.EmptyDataError:
                                    print("Empty data here, skipping")
                                test_val = res_df.loc[error_measure, "test"]
                                train_val = res_df.loc[error_measure, "train"]
                                test_vals.append(test_val)
                                train_vals.append(train_val)
                        if not res_dir in all_train_vals: 
                            all_train_vals[res_dir] = [v / norm_factors[res_d["property"]] for v in train_vals]
                        else:
                            all_train_vals[res_dir] += [v / norm_factors[res_d["property"]] for v in train_vals]
                        if not res_dir in all_test_vals: 
                            all_test_vals[res_dir] = [v / norm_factors[res_d["property"]] for v in test_vals]
                        else:
                            all_test_vals[res_dir] += [v / norm_factors[res_d["property"]] for v in test_vals]
                        res_d.update(calc_aggreagations(train_vals, prefix="train_"))
                        res_d.update(calc_aggreagations(test_vals, prefix="test_"))
                        main_df = main_df.append(res_d, ignore_index=True)
    # appending avg model results
    for res_dir in all_train_vals.keys(): 
        prop_d = parse_results_dir_str(res_dir)
        prop_d.update(calc_aggreagations(all_train_vals[res_dir], prefix="train_"))
        prop_d.update(calc_aggreagations(all_test_vals[res_dir], prefix="test_"))
        prop_d["property"] = "average"
        main_df = main_df.append(prop_d, ignore_index=True)
    # saving
    test_size = os.path.split(parent_res_dir)[-1]
    main_df.to_csv(os.path.join("../results/models", "parsed_results", "{}_test_{}_results.csv".format(test_size, error_measure)))


def parse_time_data(parent_res_dir):
    main_df = pd.DataFrame()
    for prop_dir in os.listdir(parent_res_dir):
        if not "test" in prop_dir and not ".csv" in prop_dir:
            for res_dir in os.listdir(os.path.join(parent_res_dir, prop_dir)):
                print(os.path.join(parent_res_dir, prop_dir, res_dir))
                if not os.path.isfile(os.path.join(parent_res_dir, prop_dir, res_dir, "run_info.json")):
                    print("NO RUN INFO FILE")
                    continue
                with open(os.path.join(parent_res_dir, prop_dir, res_dir, "run_info.json")) as f:
                    d = json.load(f)
                    time = d["runtime"]
                    train_size = d["train_size"]
                res_d = parse_results_dir_str(os.path.join(parent_res_dir, prop_dir, res_dir))
                res_d["runtime_per_mol"] = time / train_size
                main_df = main_df.append(res_d, ignore_index=True)
    test_size = os.path.split(parent_res_dir)[-1]
    main_df.to_csv(os.path.join("../results/models", "parsed_results", "{}_test_time_results.csv".format(test_size)))

    
if __name__ == "__main__":
    # parsing results
    parent_res = "../results/models/"
    parse_time_data(parent_res + "5000")
    #for parent in os.listdir(parent_res):
    #    unite_all_results_to_csv(os.path.join(parent_res, parent), "mare")
    #    unite_all_results_to_csv(os.path.join(parent_res, parent), "mae")
    
    # calculating metrics for data
    data = pd.read_csv("../data/all_data.csv")
    data.describe().to_csv(os.path.join("../results/models", "parsed_results", "data_descrps.csv"))
    