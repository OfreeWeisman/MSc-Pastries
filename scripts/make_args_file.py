from settings import args_dict, properties
import os

def make_args_list(comp_type: str):
    """Makes a list of strings to print to the argument file"""
    if comp_type in args_dict:
        args = []
        for test_size in [1000, 2500, 5000, 7600]:
            for property in properties:
                for pair in args_dict[comp_type]["pairs"]:
                    model = pair[0]
                    rep = pair[1] if not "padd" in pair[1] else pair[1][:-5]
                    print("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep))
                    if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)): 
                        args.append(" ".join([args_dict[comp_type]["exe"], property, pair[0], pair[1], str(test_size)]))
        return args
    else:
        raise ValueError("Not valid comp_type={}. Allowed types {}".format(comp_type, ", ".join(args_dict.keys())))
        
def _make_args_list():
    """Makes a list of strings to print to the argument file"""
    comp_type = "tensorflow"
    if comp_type in args_dict:
        args = []
        for test_size in [1000, 2500, 5000, 7600]:
            for property in properties:
                # running on all non-lala representations
                for pair in [
                              #("rf_model", "lala_features_rep"),
                              #("rf_model", "augmented_lala_features_rep"),
                              #("krr_model","lala_features_rep"),
                              #("krr_model","augmented_lala_features_rep"),
                              ("rf_model_custodi_rep", "augmented_lala_str_rep_padd"),
                              ("rf_model_custodi_rep", "lala_str_rep_padd"),
                              ("rf_model_custodi_rep", "smiles_str_rep_padd"),
                              ("krr_model_custodi_rep", "lala_str_rep_padd"),
                              ("krr_model_custodi_rep", "augmented_lala_str_rep_padd"),
                              ("krr_model_custodi_rep", "smiles_str_rep_padd"),
                              ("custodi_model", "lala_str_rep"),
                              ("custodi_model", "augmented_lala_str_rep"),
                              ("custodi_model", "smiles_str_rep")
                    ]:
                    model = pair[0]
                    rep = pair[1] if not "padd" in pair[1] else pair[1][:-5]
                    #if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)):
                    #    print("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep))
                    #    #if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)): 
                    #    args.append(" ".join([args_dict[comp_type]["exe"], property, pair[0], pair[1], str(test_size)]))
                    #elif len([1 for fname in os.listdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)) if fname.endswith(".csv")]) < 5:
                    #    print("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep))
                    #    #if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)): 
                    #    args.append(" ".join([args_dict[comp_type]["exe"], property, pair[0], pair[1], str(test_size)]))
                    args.append(" ".join([args_dict[comp_type]["exe"], property, pair[0], pair[1], str(test_size)]))
        return args
    else:
        raise ValueError("Not valid comp_type={}. Allowed types {}".format(comp_type, ", ".join(args_dict.keys())))
        
def _redo_one():
    """Makes a list of strings to print to the argument file"""
    comp_type = "tensorflow"
    if comp_type in args_dict:
        args = []
        for test_size in [1000]:
            for property in ["aEA_eV"]:
                # running on all non-lala representations
                for pair in [("graph_conv", "mol_conv_rep")]:
                    model = pair[0]
                    rep = pair[1] if not "padd" in pair[1] else pair[1][:-5]
                    if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)):
                        print("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep))
                        #if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)): 
                        args.append(" ".join([args_dict[comp_type]["exe"], property, pair[0], pair[1], str(test_size)]))
                    elif sum([1 for fname in os.listdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)) if fname.endswith(".csv")]) < 5:
                        print("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep))
                        #if not os.path.isdir("../results/models/{}/{}/{}_{}".format(test_size, property, model, rep)): 
                        args.append(" ".join([args_dict[comp_type]["exe"], property, pair[0], pair[1], str(test_size)]))
        return args
    else:
        raise ValueError("Not valid comp_type={}. Allowed types {}".format(comp_type, ", ".join(args_dict.keys())))

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to arguments file")
    path = parser.parse_args().path
    with open(path, "w") as f:
        #for l in make_args_list("tensorflow") + make_args_list("sklearn") + make_args_list("deepchem"):
        for l in _make_args_list():
            f.write(l + "\n")
    
    