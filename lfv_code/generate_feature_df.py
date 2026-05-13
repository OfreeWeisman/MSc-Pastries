import pandas as pd
# import custom feature extracting functions
from feature_extractor import *

basename = "COMPAS-1D"
# read in the dataframe
df = pd.read_csv(basename + '.csv', index_col = 0)
# make new dataframe only with annulations
df_features = pd.DataFrame(index = df.index)
df_features['augmented_lalas'] = [l for l in df['augmented_lalas'].values]
df_features['lalas'] = [l for l in df['lalas'].values]
# drop nan values
df_features = df_features.dropna()

# Define features and the functions to get them
feature_fn_dict = {
    "n_rings": get_n_rings,
    "n_branching_points": get_n_branching_points,
    "n_LAL": get_n_LAL,
    "L_ratio": lambda x: get_ratio(x, "L", case_sensitive = True),
    "longest_L": get_longest_L,
    "longest_L_degeneracy": get_longest_L_degeneracy,
    "A_ratio": lambda x: get_ratio(x, "A", case_sensitive = True),
    "Aa_ratio": get_a_ratio,
    "longest_A": lambda x: get_longest_A(x, case_sensitive = True),
    "longest_A_insensitive": lambda x: get_longest_A(x, case_sensitive = False)
}

# Loop over the features and generate new columns of data in the dataframe
for feature, function in feature_fn_dict.items():
    df_features[feature] = [function(lala) for lala in df_features['augmented_lalas']]

print(df_features.head(5))
# save dataframe
df_features.to_csv(basename + '_features.csv', float_format='%.4f')
