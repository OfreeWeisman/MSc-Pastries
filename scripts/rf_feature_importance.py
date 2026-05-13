import pandas as pd
import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.models import RfModel
import settings


def load_model_for_prop(property, feat_group, test_size, rep):
    model_dir = "../results/models/{}/{}/rf_model_{}/{}_model".format(test_size, property, feat_group, rep)
    if not os.path.isdir(model_dir):
        return None
    model = RfModel.load(model_dir)
    return model

def read_data(property, test_size, nreps, feat_group, feature_names):
    # gathering results
    df = pd.DataFrame()
    for rep in range(1, nreps + 1):
        model = load_model_for_prop(property, feat_group, test_size, rep)
        if model:
          d = {name: im for name, im in zip(feature_names, model.feature_importances_)}
          df = df.append(d, ignore_index=True)
    # aggregating
    n = len(df)
    avg = df.mean()
    std = 1 / (n - 1) * ((df - avg)**2).sum()
    # formatting
    return pd.DataFrame(
          {
              "property": [property for _ in range(len(df.columns))],
              "str": df.columns,
              "n": [n for _ in range(len(df.columns))],
              "avg": avg.values,
              "var": std.values
        })

if __name__ == "__main__":
    feature_groups = {k: v for k, v in settings.reps_dict.items() if "features_rep" in k and not "no_ratio" in k}
    for feat_group, representation in feature_groups.items():
        res = pd.DataFrame()
        feature_names = representation.col_names
        for property in settings.properties:
            print("running", feat_group, "with", property)
            df = read_data(property, 1000, 5, feat_group, feature_names)
            res = pd.concat([res, df])
        res.to_csv("../results/feature_importance/rf_{}.csv".format(feat_group))
        