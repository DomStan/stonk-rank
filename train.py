import os

import pickle
import xgboost as xgb
import sklearn

import numpy as np
import pandas as pd

from typing import Dict
from typing import Any
from typing import Tuple

import preprocessing

def train_production_xgb(
    df: pd.DataFrame, params: Dict[str, Any], noise_level: float = 0
) -> Tuple[xgb.XGBClassifier, sklearn.base.TransformerMixin]:
    X_train, scalers = preprocessing.transform_features(df, noise_level=noise_level)
    y_train = df["label"]

    clf = xgb.XGBClassifier(**params)

    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)])
    clf.save_model(os.path.join("data", "xgb_classifier.json"))

    with open(os.path.join("data", "scalers.json"), "wb") as fp:
        pickle.dump(scalers, fp)

    return clf, scalers