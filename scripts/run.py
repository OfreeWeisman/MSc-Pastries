import os
import json
from time import time
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.models import BaseEstimator
from src.core.representations import Representation
from src.commons import make_data, bootstrap_data, run_bayes_cv, analyze_model
import settings

def main_run(model: BaseEstimator, rep: Representation, property: str, test_size: int):
    """Main run function for model, representation and property. 
    Makes hyperparameter optimization, trains model on final training set and saves all required data."""
    # making data
    if property in settings.properties:
        X, y = make_data(property.lower(), rep)
    # error if not valid property
    else:
        raise ValueError("Unrecognized property {}".format(property))
    # making appropriate results directory
    prop_dir = os.path.join(settings.parent_res_dir, str(test_size), property)
    if not os.path.isdir(prop_dir):
        os.makedirs(prop_dir)
    res_dir = os.path.join(prop_dir, "{}_{}".format(model.name, rep.name))
    # Making Bootstraped data
    data = bootstrap_data(X, y, settings.n_bootstraps, test_size)
    counter = 1
    # running CV
    for x_train, x_test, y_train, y_test in data:
        print_msg("PREFORMING BOOTSTRAP EXPERIMENT {} OUT OF {}".format(counter, len(data)))
        tick = time()
        opt_model = run_bayes_cv(model, 
                                    settings.model_search_spaces[model.name],
                                    x_train, 
                                    y_train, 
                                    settings.cv_number, 
                                    settings.cv_n_iter,
                                    settings.cv_njobs)
        tock = time()
        # estimating performance & saving model
        analyze_model(opt_model, x_train, y_train, x_test, y_test, res_dir, prefix="{}_".format(counter), **settings.fit_plot_kwargs)
        counter += 1
    with open(os.path.join(res_dir, "run_info.json"), "w") as f:
        json.dump({"runtime": tock - tick, "njobs": settings.cv_njobs, "test_size": len(y_test), "train_size": len(y_train)}, f)


def print_msg(msg: str):
    l = max(int(round(len(msg) * 1.5)), 50)
    pre_space = int(round(l / 2 - len(msg) / 2))
    print("=" * l)
    print(" " * pre_space + msg)
    print("=" * l)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run single model-representation pair")
    parser.add_argument("model", type=str, help="Name of the model. Allowed values {}".format(", ".join(settings.models_dict.keys())))
    parser.add_argument("representation", type=str, help="Name of the representaion. Allowed values {}".format(", ".join(settings.reps_dict.keys())))
    parser.add_argument("property", type=str, help="Name of the property. Allowed values {}".format(", ".join(settings.properties)))
    parser.add_argument("test_size", type=int, help="Number of samples in test set")
    args = parser.parse_args()
    print_msg("Running for model {} with representation {} for {} property with {} test points".format(args.model, args.representation, args.property, args.test_size))
    model = settings.models_dict[args.model]
    rep = settings.reps_dict[args.representation]
    try:
        main_run(model, rep, args.property, args.test_size)
        print_msg("COMPUTATION ENDED SUCCESSFULLY")
    except Exception as err:
        print_msg("COMPUTATION ENDED WITH AN ERROR")
        raise err