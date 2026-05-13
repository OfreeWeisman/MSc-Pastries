"""Settings for all calculations to come.
Contains
    - hyperparameter search space for BayesCV
    - Model - representation pairs
    - properties - list of properties for calculations"""
import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import models
from src.core import representations as reps
from skopt.space import Real, Integer, Categorical


# list of (model, representation) pairs, to make data easily

args_dict = {
              "tensorflow": {
                              "pairs": [("rnn", "lala_one_hot_rep"),
                                        ("rnn", "augmented_lala_one_hot_rep"),
                                        ("rnn", "smiles_one_hot_rep")],
                              "exe": "/home/shaharpit/miniconda3/envs/tensorflow/bin/python"
                            },
              "deepchem":   {
                              "pairs": [("graph_conv", "mol_conv_rep")],
                              "exe": "/home/shaharpit/miniconda3/envs/tensorflow/bin/python"
                            },
              "sklearn":    {
                              "pairs": [
                                        ("rf_model", "lala_features_rep"),
                                        ("rf_model", "augmented_lala_features_rep"),
                                        ("rf_model", "lala_features_rep_no_ratio"),
                                        ("rf_model", "augmented_lala_features_rep_no_ratio"),
                                        ("rf_model_custodi_rep", "augmented_lala_str_rep_padd"),
                                        ("rf_model_custodi_rep", "lala_str_rep_padd"),
                                        ("rf_model_custodi_rep", "smiles_str_rep_padd"),
                                        ("krr_model","lala_features_rep"),
                                        ("krr_model","augmented_lala_features_rep"),
                                        ("krr_model_custodi_rep", "lala_str_rep_padd"),
                                        ("krr_model_custodi_rep", "augmented_lala_str_rep_padd"),
                                        ("krr_model_custodi_rep", "smiles_str_rep_padd"),
                                        ("custodi_model", "lala_str_rep"),
                                        ("custodi_model", "augmented_lala_str_rep"),
                                        ("custodi_model", "smiles_str_rep")
                                        ],
                              "exe": "/home/shaharpit/miniconda3/envs/tensorflow/bin/python"
                            },              
              }

models_dict = {
    "rnn": models.RNN(),
    "custodi_model": models.CustodiModel(),
    "graph_conv": models.GraphConv(),
    "krr_model": models.KrrModel(),
    "rf_model": models.RfModel(),
    "krr_model_custodi_rep": models.CustodiKrrModel(),
    "rf_model_custodi_rep": models.CustodiRfModel()
    }

reps_dict = {
        "lala_one_hot_rep": reps.LalaOneHot(),
        "augmented_lala_one_hot_rep": reps.AugmentedLalaOneHot(),
        "smiles_one_hot_rep": reps.SmilesOneHot(),
        "mol_conv_rep": reps.MolConvRepresentation(),
        "lala_features_rep": reps.LalaFeatures(True),
        "augmented_lala_features_rep": reps.AugmentedLalaFeatures(True),
        "lala_features_rep_no_ratio": reps.LalaFeatures(False),
        "augmented_lala_features_rep_no_ratio": reps.AugmentedLalaFeatures(False),
        "smiles_str_rep": reps.SmilesString(),
        "smiles_str_rep_padd": reps.SmilesString(padd=True),
        "lala_str_rep": reps.LalaString(),
        "lala_str_rep_padd": reps.LalaString(padd=True),
        "augmented_lala_str_rep": reps.AugmentedLalaString(),
        "augmented_lala_str_rep_padd": reps.AugmentedLalaString(padd=True)
}

# list of model hyperparameter search spaces
model_search_spaces = {
    "rnn": {
            "learning_rate": Real(1e-6, 1e-1),
            "activation": Categorical(["tanh", "relu"]),
            "dropout_rate": Real(0, 1),
            "batch_size": Integer(16, 256)
           },
    "custodi_model": {
            "degree": Integer(1, 6),
            "alpha": Real(1e-5, 1e3),
           },
    "graph_conv": {
            "n_filters": Integer(64, 256),
            "n_fully_connected_nodes": Integer(64, 256),
            #"n_pair_feat": Integer(5, 25),
            #"n_hidden": Integer(50, 300),
            "learning_rate": Real(1e-6, 1e-1),
            "batch_size": Integer(16, 256),
           },
    "krr_model": {
            "alpha": Real(1e-5, 1e2),
            "kernel": Categorical(["linear", "rbf"]),
           },
    "rf_model": {
            "n_estimators": Integer(100, 500)
            },
    "krr_model_custodi_rep": {
            "degree": Integer(1, 6),
            "custodi_alpha": Real(1e-5, 1e2),
            "alpha": Real(1e-5, 1e2),
            "kernel": Categorical(["rbf"]),
           },
    "rf_model_custodi_rep": {
            "n_estimators": Integer(100, 500),
            "degree": Integer(1, 6),
            "custodi_alpha": Real(1e-5, 1e2),
           }
    }

# list of properties to run for
properties = ["HOMO_eV",
              "LUMO_eV",
              "GAP_eV",
              # "Dipmom_Debye",
              "rel_Etot_eV",
              # "rel_Etot_pos_eV",
              # "rel_Etot_neg_eV",
              "aEA_eV",
              "aIP_eV"]
properties = [s.lower() for s in properties]

# general running parameters
cv_number = 3
cv_n_iter = 8
cv_njobs = 3
n_bootstraps = 5
fit_plot_kwargs = {"alpha": 0.2}
parent_res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models")
test_size = 1000 