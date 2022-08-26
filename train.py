import os

import pickle
import xgboost as xgb
import sklearn

import numpy as np
import pandas as pd

from hyperopt import STATUS_OK, STATUS_FAIL, Trials, fmin, hp, tpe, atpe, rand

from typing import Dict
from typing import Any
from typing import Tuple

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import preprocessing
import evaluate


def model_hp_search(
    data_splits: Dict[str, pd.DataFrame],
    n_evals: int,
    trial_name: str,
    additive_random_noise: float,
    write_csv: bool = True,
    random_state: int = 1,
    data_dir: str = "data",
    output_dir: str = "experiments",
) -> pd.DataFrame:
    _X_train, scalers = preprocessing.transform_features(
        data_splits["train"], noise_level=additive_random_noise
    )

    _X_valid, _ = preprocessing.transform_features(
        data_splits["validation"], scalers=scalers, noise_level=0
    )

    _y_train = data_splits["train"]["label"]
    _y_valid = data_splits["validation"]["label"]

    _hyperparameter_space = {
        "gamma": hp.uniform("gamma", 0, 5),
        "scale_pos_weight": hp.uniform("scale_pos_weight", 3, 7),
        "max_depth": hp.quniform("max_depth", 3, 9, 1),
        "min_child_weight": hp.quniform("min_child_weight", 1, 8, 1),
        "max_delta_step": hp.quniform("max_delta_step", 1, 4, 1),
        "n_estimators": hp.quniform("n_estimators", 25, 85, 1),
        # "subsample": hp.uniform("subsample", 0.9, 1),
        # "colsample_bylevel" : hp.uniform("colsample_bylevel", 0.5, 1),
    }

    def _optimization_objective(space):
        clf = xgb.XGBClassifier(
            # Uniform floating point
            gamma=space["gamma"],
            scale_pos_weight=space["scale_pos_weight"],
            subsample=1,
            colsample_bylevel=1,
            # subsample = space['subsample'],
            # colsample_bylevel = space['colsample_bylevel'],
            # Uniform integer
            max_depth=int(space["max_depth"]),
            min_child_weight=int(space["min_child_weight"]),
            max_delta_step=int(space["max_delta_step"]),
            n_estimators=int(space["n_estimators"]),
            # Constant
            learning_rate=0.1,
            tree_method="hist",
            enable_categorical=True,
            max_cat_to_onehot=1,
            random_state=random_state,
        )

        clf.fit(
            _X_train,
            _y_train,
            verbose=False,
        )

        y_score = clf.predict_proba(_X_valid)[:, 1]
        y_preds = y_score > 0.5

        f1 = f1_score(_y_valid, y_preds, zero_division=0)
        precision = precision_score(_y_valid, y_preds, zero_division=0)
        ap = evaluate.average_precision_from_cutoff(_y_valid, y_score, 0.5)
        roc = roc_auc_score(_y_valid, y_score)

        pos_preds = int(y_preds.sum())
        pos_labels = int(_y_valid.sum())

        ap = ap if pos_preds >= pos_labels else 0

        if f1 == 0 or precision == 0:
            return {
                "loss": 100,
                "precision": precision,
                "f1_score": f1,
                "ap": ap,
                "auc": roc,
                "pos_preds": pos_preds,
                "status": STATUS_FAIL,
            }
        else:
            return {
                "loss": -ap,
                "precision": precision,
                "f1_score": f1,
                "ap": ap,
                "auc": roc,
                "pos_preds": pos_preds,
                "status": STATUS_OK,
            }

    trials = Trials()

    best_hyperparams = fmin(
        fn=_optimization_objective,
        space=_hyperparameter_space,
        algo=tpe.suggest,
        max_evals=n_evals,
        trials=trials,
    )

    trial_vals = trials.vals
    trial_vals["f1_score"] = list(map(lambda x: x["f1_score"], trials.results))
    trial_vals["precision"] = list(map(lambda x: x["precision"], trials.results))
    trial_vals["ap"] = list(map(lambda x: x["ap"], trials.results))
    trial_vals["auc"] = list(map(lambda x: x["auc"], trials.results))
    trial_vals["pos_preds"] = list(map(lambda x: x["pos_preds"], trials.results))

    df_trials = pd.DataFrame.from_dict(trial_vals).sort_values("ap", ascending=False)

    if write_csv:
        output_path = os.path.join(data_dir, output_dir, trial_name + ".csv")
        df_trials.to_csv(output_path, index=False)

    return df_trials


def train_production_xgb(
    dataset: pd.DataFrame,
    params: Dict[str, Any],
    noise_level: float = 0,
    verbose: bool = True,
    data_dir: str = "data",
) -> Tuple[xgb.XGBClassifier, sklearn.base.TransformerMixin]:
    X_train, scalers = preprocessing.transform_features(
        dataset, noise_level=noise_level
    )
    y_train = dataset["label"]

    clf = xgb.XGBClassifier(**params)

    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=verbose)
    clf.save_model(os.path.join(data_dir, "xgb_classifier.json"))

    with open(os.path.join(data_dir, "scalers.json"), "wb") as fp:
        pickle.dump(scalers, fp)

    return clf, scalers
