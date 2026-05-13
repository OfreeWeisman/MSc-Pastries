"""File to contain regression models as a sklearn Estimators"""
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import Ridge
from typing import List
import os
import json
import joblib

from ..commons import calc_mae, calc_rmse
# trying to import required packages (not all required at the same time)
try: 
    import tensorflow as tf
except ImportError:
    pass

try:
    import deepchem as dc
except ImportError:
    pass


# importing standard usful models already implemented in sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge


class RNN (BaseEstimator):

    name = "rnn"
    
    def __init__(self, learning_rate: float=0.001, dropout_rate: float=0.1, activation: str="tanh", batch_size: int=32):
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def build_rnn(self, input_shape: tuple, learning_rate: float, dropout_rate: float, activation: str):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(input_shape))
        model.add(tf.keras.layers.LSTM(50, activation='tanh'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(50, activation=activation))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), tf.keras.losses.MSE)
        return model
    

    def fit(self, X, y):
        model = self.build_rnn(X.shape[1:], self.learning_rate, self.dropout_rate, self.activation)
        model.fit(X, y, epochs=100, batch_size=self.batch_size)
        self.model = model
        return self


    def predict(self, X):
        return self.model.predict(X)

    
    def score(self, X, y):
        pred = self.predict(X)
        return - calc_mae(pred, y)


    def save(self, model_dir: str):
        """Method to save the model to disk. saving to model_dir"""
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        self.model.save(os.path.join(model_dir, "tf_model"))
        with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
            json.dump(self.get_params(), f)
        

    @staticmethod
    def load(model_dir: str):
        """Method to load model from disk. model saved at model_dir. returns loaded model"""
        tf_model = tf.keras.models.load_model(os.path.join(model_dir, "tf_model"))
        with open(os.path.join(model_dir, "model_parameters.json"), "r") as f:
            d = json.load(f)
            model = RNN(**d)
        model.model = tf_model
        return model

    

class CustodiModel (BaseEstimator):

    """A CUSTODI model.
    ARGS:
        - degree (int): required degree of CUSTODI model
        - alpha (float): regularization constant for fitting CUSTODI's dictionary
        - max_iter (int): maximum iterations for convergence of dictionary"""

    name = "custodi_model"

    def __init__(self, degree: int=2, alpha: float=0.1):
        self.degree = degree
        self.alpha = alpha
        self.max_iter = 1e5
        self.fit_intercept = True


    @staticmethod
    def _gen_idx_dict_for_custodi(X, degree):
        char_sets = [set() for _ in range(degree)]
        for string in X:
            for idx in range(len(string)):
                for i, s in enumerate(char_sets):
                    try:
                        a = string[idx:(idx + i + 1)]
                        if len(a) == i + 1:
                            s.add(a)
                    except IndexError:
                        pass
        idx_dict = {}
        for i, s in enumerate(char_sets):
            for j, char in enumerate(list(s)):
                if i == 0:
                    idx_dict[char] = j
                else:
                    idx_dict[char] = j + len(char_sets[i - 1])
        return idx_dict


    def fit(self, X: List[str], y: np.ndarray):
        idx_dict = self._gen_idx_dict_for_custodi(X, self.degree)
        counts_vector = []
        strings = np.ravel(X)
        for string in strings:
            x = np.zeros(len(idx_dict))
            for idx in range(len(string)):
                for i in range(self.degree):
                    try:
                        x[idx_dict[string[idx:(idx + i + 1)]]] += 1
                    except IndexError:
                        pass
            counts_vector.append(x)
        reg = Ridge(fit_intercept=self.fit_intercept, alpha=self.alpha, max_iter=self.max_iter)
        counts_vector = np.array(counts_vector)
        reg.fit(counts_vector, y)
        d = {}
        for key, c in zip(idx_dict.keys(), reg.coef_[0]):
            d[key] = c
        self.dictionary = d
        self.intercept = reg.intercept_[0] if self.fit_intercept else 0
        return self

    
    @staticmethod
    def transform_string(string: str, dictionary: dict, degree: int):
        tokenized = []
        for idx in range(len(string)):
            t = 0
            for i in range(degree):
                try:
                    if not len(string[idx:(idx + i + 1)]) == i + 1:
                        break
                    t += dictionary[string[idx:(idx + i + 1)]]
                except KeyError:
                    pass
            tokenized.append(t)
        return tokenized


    def transform(self, X: List[str]):
        tokenized = []
        strings = np.ravel(X)
        for string in strings:
            tokenized.append(self.transform_string(string, self.dictionary, self.degree))
        return tokenized


    def predict(self, X: List[str]):
        encoded = self.transform(X)
        return np.array([sum(v) + self.intercept for v in encoded])


    def score(self, X, y):
        pred = self.predict(X)
        return - calc_mae(pred, y)


    def save(self, model_dir: str):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        save_dict = {}
        save_dict["dictionary"] = self.dictionary
        save_dict["intercept"] = self.intercept
        with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
            json.dump(self.get_params(), f)
        with open(os.path.join(model_dir, "tok_parameters.json"), "w") as f:
            json.dump(save_dict, f)

    @staticmethod
    def load(model_dir: str):
        """Method to load model from disk. model saved at model_dir. returns loaded model"""
        with open(os.path.join(model_dir, "model_parameters.json"), "r") as f:
            d = json.load(f)
            model = CustodiModel(**d)
        with open(os.path.join(model_dir, "tok_parameters.json"), "r") as f:
            d = json.load(f)
            model.dictionary = d["dictionary"]
            model.intercept = d["intercept"]
        return model


class GraphConv (BaseEstimator):

    name = "graph_conv"
    
    def __init__(self, learning_rate: float=0.001, n_filters: int=64, n_fully_connected_nodes: int=64, batch_size: int=128):
        self.n_filters = n_filters
        self.n_fully_connected_nodes = n_fully_connected_nodes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epochs = 500
    
    #def __init__(self, learning_rate: float=0.001, n_hidden: int=64, n_pair_feat: int=64, batch_size: int=128):
    #    self.n_hidden = n_hidden
    #    self.n_pair_feat = n_pair_feat
    #    self.learning_rate = learning_rate
    #    self.batch_size = batch_size

    def fit(self, X, y):
        # build data for training
        dataset = dc.data.NumpyDataset(X, y)
        # build model
        model = dc.models.GraphConvModel(n_tasks=1, 
                        graph_conv_layers=[self.n_filters, self.n_filters], 
                        dense_layer_size=self.n_fully_connected_nodes, 
                        batch_size=self.batch_size,
                        learning_rate=self.learning_rate,
                        mode='regression',
                        number_atom_features=100)
        #model = dc.models.MPNNModel(n_tasks=1,
        #                n_hidden=self.n_hidden,
        #                n_pair_feat=self.n_pair_feat,
        #                batch_size=self.batch_size,
        #                learning_rate=self.learning_rate,
        #                mode='regression')
        model.fit(dataset, nb_epoch=1000)
        self.model = model
        return self


    def predict(self, X):
        return self.model.predict_on_batch(X)


    def score(self, X, y):
        pred = self.predict(X)
        return - calc_mae(pred, y)

    def save(self, model_dir: str):
        # deepchem is based on keras models, so basically same syntax
        """Method to save the model to disk. saving to model_dir"""
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        #self.model.model.save(os.path.join(model_dir, "tf_model"))
        with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
            json.dump(self.get_params(), f)


    @staticmethod
    def load(model_dir: str):
        """Method to load model from disk. model saved at model_dir. returns loaded model"""
        tf_model = tf.keras.models.load_model(os.path.join(model_dir, "tf_model"))
        with open(os.path.join(model_dir, "model_parameters.json"), "r") as f:
            model = GraphConv(**json.load(f))
            dc_model = dc.models.GraphConvModel(n_tasks=1, 
                graph_conv_layers=[model.n_filters, model.n_filters], 
                dense_layer_size=model.n_fully_connected_nodes, 
                batch_size=model.batch_size,
                learning_rate=model.learning_rate,
                mode='regression')
            dc_model.model = tf_model
            model.model = dc_model
            return model


class CustodiRepModel (BaseEstimator):
    
    """Generic model that runs on CUSTODI representation.
    It is easier to implement it as a model as the CV will be on both CUSTODI and the model's arguments"""

    def fit_custodi(self, X, y):
        custodi_model = CustodiModel(self.degree, self.custodi_alpha)
        custodi_model.fit_intercept = True # allowing for fitting intercept in custodi representations
        custodi_model.fit(X, y)
        self.custodi_model = custodi_model
        self.padd_length = max([len(x) for x in X])


    def predict(self, X):
        # making data
        tok_X = self.custodi_model.transform(X)
        # predicting
        return self.model.predict(tok_X)


    def score(self, X, y):
        pred = self.predict(X)
        return - calc_mae(pred, y)


    def save(self, model_dir):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        # saving model using joblib
        joblib.dump(self.model, os.path.join(model_dir, "model.sav"))
        # saving model parameters to json
        with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
            json.dump(self.model.get_params(), f)
        # saving custodi model
        self.custodi_model.save(os.path.join(model_dir, "custodi"))
        

    @staticmethod
    def load(model_dir):
        # loading sk model
        sk_model = joblib.load(os.path.join(model_dir, "model.sav"))
        with open(os.path.join(model_dir, "model_parameters.json"), "r") as f:
            sk_model.set_params(**json.load(f))
        # loading custodi
        custodi_model = CustodiModel.load(os.path.join(model_dir, "custodi"))
        # putting it all together
        out_model = CustodiRepModel()
        out_model.custodi_model = custodi_model
        out_model.model = sk_model
        return out_model


class CustodiKrrModel (CustodiRepModel):

    name = "krr_model_custodi_rep"

    def __init__(self, alpha: float=0.1, kernel: str="rbf", degree: int=2, custodi_alpha: float=0.1):
        self.alpha = alpha
        self.kernel = kernel
        self.degree = degree
        self.custodi_alpha = custodi_alpha
        self.max_iter = 1e5

    def fit(self, X, y):
        # train custodi tokenizer
        self.fit_custodi(X, y)
        # tokenize using custodi
        tok_X = self.custodi_model.transform(X)
        # build model
        sk_model = KrrModel(alpha=self.alpha, kernel=self.kernel)
        # train
        sk_model.fit(tok_X, y)
        self.model = sk_model
        return self


class CustodiRfModel (CustodiRepModel):

    name = "rf_model_custodi_rep"

    
    def __init__(self, n_estimators: int=100, degree: int=2, custodi_alpha: float=0.1):
        self.degree = degree
        self.custodi_alpha = custodi_alpha
        self.n_estimators = n_estimators
        self.max_iter = 1e5

    def fit(self, X, y):
        # train custodi tokenizer
        self.fit_custodi(X, y)
        # tokenize using custodi
        tok_X = self.custodi_model.transform(X)
        # build model
        sk_model = RfModel(n_jobs=1)
        # train
        sk_model.fit(tok_X, y)
        self.model = sk_model
        return self


class KrrModel (KernelRidge):

    name = "krr_model"

    def score(self, X, y):
        pred = self.predict(X)
        return - calc_mae(pred, y)


    def save(self, model_dir):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        # saving model using joblib
        joblib.dump(self, os.path.join(model_dir, "model.sav"))
        # saving model parameters to json
        with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
            json.dump(self.get_params(), f)


    @staticmethod
    def load(model_dir):
        sk_model = joblib.load(os.path.join(model_dir, "model.sav"))
        with open(os.path.join(model_dir, "model_parameters.json"), "r") as f:
            sk_model.set_params(**json.load(f))
        return sk_model


class RfModel (RandomForestRegressor):

    name = "rf_model"

    def score(self, X, y):
        pred = self.predict(X)
        return - calc_mae(pred, y)


    def save(self, model_dir):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        # saving model using joblib
        joblib.dump(self, os.path.join(model_dir, "model.sav"))
        # saving model parameters to json
        with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
            json.dump(self.get_params(), f)


    @staticmethod
    def load(model_dir):
        sk_model = joblib.load(os.path.join(model_dir, "model.sav"))
        with open(os.path.join(model_dir, "model_parameters.json"), "r") as f:
            sk_model.set_params(**json.load(f))
        return sk_model
