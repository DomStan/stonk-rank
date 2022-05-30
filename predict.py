import pandas as pd
import numpy as np

import xgboost as xgb

import pickle
import os

import preprocessing


class StonkModelInterface():
    _model = None
    _scalers = None
    
    def predict(self, X):
        pass

class XGBStonkModel(StonkModelInterface):
    def __init__(self, model_dir='data', model_file='xgb_classifier.json', scalers_file='scalers.json'):
        self._model_dir = model_dir
        self._model_file = model_file
        self._scalers_file = scalers_file
        
        self._model = xgb.XGBClassifier()
        self._model.load_model(os.path.join(self._model_dir, self._model_file))
        with open(os.path.join(self._model_dir, self._scalers_file), 'rb') as fp:
            self._scalers = pickle.load(fp)

    def predict(self, X):
        X_transformed, _ = preprocessing.transform_features(X, scalers=self._scalers, add_noise=False)
        return self._model.predict_proba(X_transformed)[:, 1]

