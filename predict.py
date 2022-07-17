import pandas as pd
import numpy as np

from typing import Tuple

import xgboost as xgb

import pickle
import os

import preprocessing


class StonkModelInterface:
    _model = None
    _scalers = None

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        pass


class XGBStonkModel(StonkModelInterface):
    def __init__(
        self,
        model_dir="data",
        model_file="xgb_classifier.json",
        scalers_file="scalers.json",
    ):
        self._model_dir = model_dir
        self._model_file = model_file
        self._scalers_file = scalers_file

        self._model = xgb.XGBClassifier()
        self._model.load_model(os.path.join(self._model_dir, self._model_file))
        with open(os.path.join(self._model_dir, self._scalers_file), "rb") as fp:
            self._scalers = pickle.load(fp)
            assert self._scalers is not None

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        X_transformed, scalers_returned = preprocessing.transform_features(
            X=X, scalers=self._scalers, noise_level=0
        )
        assert scalers_returned is self._scalers
        
        return self._model.predict_proba(X_transformed)[:, 1], X_transformed
