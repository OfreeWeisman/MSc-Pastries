"""Script to convert the raw into single CSV file and remove unwanted entries"""
import pandas as pd
    
def parse_raw():
    prop_df = pd.read_csv("../data/raw/COMPAS-1D.csv")
    feat_df = pd.read_csv("../data/raw/COMPAS-1D_features.csv")
    # READING PROPERTIES
    # dropping irrelevant molecules
    prop_df = prop_df[~prop_df.molecule.str.contains('c46h26')].reset_index(drop=True)
    prop_df = prop_df[~prop_df.molecule.str.contains('c6h6')].reset_index(drop=True)
    prop_df = prop_df[~prop_df.molecule.str.contains('c10h8')].reset_index(drop=True)
    # taking relevant columns
    prop_df = prop_df[["molecule", "smiles", "augmented_lalas", "lalas", "HOMO_eV", "LUMO_eV", "GAP_eV", "aEA_eV", "aIP_eV", "Erel_eV"]]
    # setting index
    prop_df = prop_df.set_index("molecule")
    # READING LALA FEATURE VECTOR
    # dropping irrelevant molecules
    feat_df = feat_df[~feat_df.molecule.str.contains('c46h26')].reset_index(drop=True)
    feat_df = feat_df[~feat_df.molecule.str.contains('c6h6')].reset_index(drop=True)
    feat_df = feat_df[~feat_df.molecule.str.contains('c10h8')].reset_index(drop=True)
    # dropping irrelevant columns
    feat_df = feat_df.drop(columns=["augmented_lalas", "lalas"])
    # setting index
    feat_df = feat_df.set_index("molecule")
    # joining dataframes
    joined = pd.concat([prop_df, feat_df], axis=1, join="inner")
    print(joined)
    joined.to_csv("../data/all_data.csv")
    

if __name__ == "__main__":
    parse_raw()
