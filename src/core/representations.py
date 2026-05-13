"""File to contain representations"""

from abc import ABC, abstractclassmethod
from typing import List
import numpy as np

try:
    from rdkit import Chem
    import deepchem as dc
except ImportError:
    pass

from ..commons import read_raw_columns

class Representation (ABC):

    @abstractclassmethod
    def represent(self, molecule_ids: List[str]):
        """Method to make representation for molecules given their IDs in the dataset"""
        pass


class OneHot (Representation):
    """Generic one-hot tokenization (includes padding of strings of different lengths)"""

    def __init__(self, strings):
        self.tokenization_dict = self.set_tokenization_dict(strings)
        self.max_length = max([len(s) for s in strings])

    @staticmethod
    def set_tokenization_dict(strings: List[str]):
        """Method to generate standard word tokenization dictionary"""
        d = {}
        for string in strings:
            for char in string:
                if char not in d.keys():
                    d[char] = len(d.keys())
        return d


    def tokenize_string(self, string):
        """Method to tokenize a string"""
        tok = []
        for s in string:
            v = np.zeros(len(self.tokenization_dict) + 2)
            v[self.tokenization_dict[s]] = 1
            tok.append(v)
        padd_vec = np.zeros(len(self.tokenization_dict) + 2)
        padd_vec[-1] = 1
        while len(tok) < self.max_length:
            tok.append(padd_vec)
        return tok


    def represent(self, molecule_id: str):
        return super().represent(molecule_id)


class LalaOneHot (OneHot):
    """LALA based one-hot"""

    name = "lala_one_hot_rep"


    def __init__(self):
        strings = read_raw_columns(["lalas"])["lalas"].values
        super().__init__(strings)

    def represent(self, molecule_ids: List[str]):
        strings = read_raw_columns(["lalas"]).loc[molecule_ids, :]["lalas"].values
        return [self.tokenize_string(s) for s in strings]
        
class AugmentedLalaOneHot (OneHot):
    """LALA based one-hot"""

    name = "augmented_lala_one_hot_rep"


    def __init__(self):
        strings = read_raw_columns(["augmented_lalas"])["augmented_lalas"].values
        super().__init__(strings)

    def represent(self, molecule_ids: List[str]):
        strings = read_raw_columns(["augmented_lalas"]).loc[molecule_ids, :]["augmented_lalas"].values
        return [self.tokenize_string(s) for s in strings]


class SmilesOneHot (OneHot):
    """SMILES based one-hot"""

    name = "smiles_one_hot_rep"


    def __init__(self):
        strings = read_raw_columns(["smiles"])["smiles"].values
        super().__init__(strings)

    def represent(self, molecule_ids: List[str]):
        strings = read_raw_columns(["smiles"]).loc[molecule_ids, :]["smiles"].values
        return [self.tokenize_string(s) for s in strings]


class MolConvRepresentation (Representation):
    """Representation as a deepchem.ConvMol for graph convolutions data"""

    name = "mol_conv_rep"
    
    def rdmols_from_smiles(self, smiles_list: List[str]):
        return [Chem.MolFromSmiles(smiles, sanitize=True) for smiles in smiles_list]
        #featurizer = dc.feat.ConvMolFeaturizer()
        return featurizer.featurize(rdmol)

    def represent(self, molecule_ids: List[str]):
        strings = read_raw_columns(["smiles"])["smiles"].values
        featurizer = dc.feat.ConvMolFeaturizer()
        #featurizer = dc.feat.MolGraphConvFeaturizer()
        return featurizer.featurize(strings)


class AugmentedLalaFeatures (Representation):
    """Representation as the LALA feature vector"""

    name = "augmented_lala_features_rep"
    
    def __init__(self, add_ratio: bool):
        self.col_names = ["n_rings",
                          "n_branching_points",
                          #"n_secondary_branching_points",
                          "n_LAL",
                          "longest_L",
                          "longest_L_degeneracy",
                          #"second_longest_L",
                          #"A_ratio_insensitive",
                          "Aa_ratio",
                          "longest_A",
                          "longest_A_insensitive",
                          #"longest_A_degeneracy"
                          ]
                          #"second_longest_A",
                          #"second_longest_A_insensitive"]
        if add_ratio:
            self.col_names += ["A_ratio", "L_ratio"]
    
    def represent(self, molecule_ids: List[str]):
        return read_raw_columns([c.lower() for c in self.col_names]).loc[molecule_ids, :]

class LalaFeatures (Representation):
    """Representation as the LALA feature vector"""

    name = "lala_features_rep"
    
    def __init__(self, add_ratio: bool):
        self.col_names = ["n_rings",
                          "n_branching_points",
                          #"n_secondary_branching_points",
                          "n_LAL",
                          "longest_L",
                          "longest_L_degeneracy",
                          #"second_longest_L",
                          #"A_ratio_insensitive",
                          "longest_A_insensitive",
                          ]
                          #"second_longest_A_insensitive"]
        if add_ratio:
            self.col_names += ["L_ratio"]
    
    def represent(self, molecule_ids: List[str]):
        return read_raw_columns([c.lower() for c in self.col_names]).loc[molecule_ids, :]
                                  
                                  
        

def _padd_strings(strings: List[str]):
    """private method to padd list of strings. returns padded stings"""
    max_l = max([len(s) for s in strings])
    for i in range(len(strings)):
        while len(strings[i]) < max_l:
            strings[i] += " "
    return strings


class SmilesString (Representation):
    """Representation that returns the SMILES string of the molecules as representation"""

    name = "smiles_str_rep"

    def __init__(self, padd=False) -> None:
        self.padd = padd


    def represent(self, molecule_ids: List[str]):
        if self.padd:
            return _padd_strings(read_raw_columns(["smiles"]).loc[molecule_ids, :]["smiles"].values)
        else:
            return read_raw_columns(["smiles"]).loc[molecule_ids, :]["smiles"].values


class LalaString (Representation):
    """Representation that returns the LALA string of the molecules as representation"""

    name = "lala_str_rep"

    
    def __init__(self, padd=False) -> None:
        self.padd = padd


    def represent(self, molecule_ids: List[str]):
        if self.padd:
            return _padd_strings(read_raw_columns(["lalas"]).loc[molecule_ids, :]["lalas"].values)
        else:
            return read_raw_columns(["lalas"]).loc[molecule_ids, :]["lalas"].values
            
class AugmentedLalaString (Representation):
    """Representation that returns the LALA string of the molecules as representation"""

    name = "augmented_lala_str_rep"

    
    def __init__(self, padd=False) -> None:
        self.padd = padd


    def represent(self, molecule_ids: List[str]):
        if self.padd:
            return _padd_strings(read_raw_columns(["augmented_lalas"]).loc[molecule_ids, :]["augmented_lalas"].values)
        else:
            return read_raw_columns(["augmented_lalas"]).loc[molecule_ids, :]["augmented_lalas"].values

