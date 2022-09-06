import numpy as np
import pandas as pd

from typing import Optional
from typing import Dict
from typing import Union
from typing import Tuple

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def split_data(
    df: pd.DataFrame,
    date_count_train: int,
    date_count_valid: int,
    date_count_gap: int,
    random_state: int = None,
) -> Dict[str, pd.DataFrame]:
    df_copy = df.copy()

    # Shuffle data
    df_copy = df_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)

    dates_sorted = np.sort(df["trade_date"].unique())

    total_date_count = len(dates_sorted)

    total_date_count_train = total_date_count - date_count_valid - date_count_gap

    assert total_date_count_train > 0

    assert date_count_train > 0 and date_count_train <= total_date_count_train

    dates_valid = dates_sorted[
        total_date_count_train
        + date_count_gap : total_date_count_train
        + date_count_gap
        + date_count_valid
    ]
    dates_train = dates_sorted[:total_date_count_train]
    dates_train = dates_train[-date_count_train:]

    assert len(dates_valid) == date_count_valid
    assert len(dates_train) == date_count_train

    assert all([x > dates_train[-1] for x in dates_valid])
    assert all([x < dates_valid[0] for x in dates_train])

    # Sliced dates should not overlap
    assert len(set(dates_valid) & set(dates_train)) == 0

    df_valid = (
        df_copy[df_copy.trade_date.isin(dates_valid)]
        .copy()
        .sample(frac=1, random_state=random_state)
    )
    df_train = (
        df_copy[df_copy.trade_date.isin(dates_train)]
        .copy()
        .sample(frac=1, random_state=random_state)
    )

    # No intersection
    assert (
        len(set(df_valid.trade_date.unique()) & set(df_train.trade_date.unique())) == 0
    )

    assert not np.any(df_valid.index.isin(df_train.index))

    return {"train": df_train, "validation": df_valid}


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    def label_one_example(example):
        success_one_month = (
            example["return_one_month"] >= 0.065
            and np.abs(example["last_residual"] - example["residual_one_month"]) >= 2
        )
        success_two_month = (
            example["return_two_month"] >= 0.065
            and np.abs(example["last_residual"] - example["residual_two_month"]) >= 1.5
        ) or (
            example["return_two_month"] > 0
            and np.abs(example["last_residual"] - example["residual_two_month"]) >= 2
        )
        success_three_month = (
            example["return_three_month"] >= 0.095
            and np.abs(example["last_residual"] - example["residual_three_month"])
            >= 1.5
        ) or (
            example["return_three_month"] > 0
            and np.abs(example["last_residual"] - example["residual_three_month"]) >= 2
        )
        label_positive = int(
            any([success_one_month, success_two_month, success_three_month])
        )
        return label_positive

    df_copy = df.copy()
    df_copy["label"] = df_copy.apply(label_one_example, axis=1)
    return df_copy


def transform_features(
    X: pd.DataFrame,
    scalers: Optional[Dict[str, sklearn.base.TransformerMixin]] = None,
    noise_level: Optional[float] = 0,
) -> Tuple[pd.DataFrame, Union[None, Dict[str, sklearn.base.TransformerMixin]],]:

    data_types = {
        "adf_pass_rate": np.float32,
        "last_residual": np.float32,
        "residual_mean_max": np.float32,
        "vix": np.float32,
        "betas_rsquared": np.float32,
        "arima_forecast_normalized": np.float32,
    }

    df_copy = X.copy()
    n_x = df_copy.shape[0]

    # Select features
    df_copy = df_copy[
        [
            "adf_pass_rate",
            "last_residual",
            "residual_mean_max",
            "industry",
            "vix",
            "betas_rsquared",
            "arima_forecast_normalized",
        ]
    ].astype(data_types)
    df_copy["industry"] = df_copy["industry"].astype("category")

    # Residual transform
    df_copy.loc[:, "last_residual"] = df_copy["last_residual"].abs()
    #

    # Feature cross
    df_copy.loc[:, "residual_inter"] = (
        df_copy["last_residual"] / df_copy["residual_mean_max"]
    )
    #

    # Additive random noise
    if noise_level > 0:
        df_copy.loc[:, "adf_pass_rate"] += np.random.normal(0, noise_level, n_x)
        df_copy.loc[:, "last_residual"] += np.random.normal(0, noise_level, n_x)
        df_copy.loc[:, "residual_mean_max"] += np.random.normal(0, noise_level, n_x)
        df_copy.loc[:, "vix"] += np.random.normal(0, noise_level, n_x)
    #

    # Feature scalers fit
    if scalers is None:
        scalers = dict()
        scalers["adf_pass_rate"] = _get_new_scaler("minmax").fit(
            df_copy["adf_pass_rate"].to_numpy().reshape(-1, 1)
        )
        scalers["vix"] = _get_new_scaler("standard").fit(
            df_copy["vix"].to_numpy().reshape(-1, 1)
        )
    #

    # Feature scaling
    ## ADF pass rate scaling
    scaler = scalers["adf_pass_rate"]
    df_copy.loc[:, "adf_pass_rate"] = scaler.transform(
        df_copy["adf_pass_rate"].to_numpy().reshape(-1, 1)
    )
    ##

    ## Vix index scaling
    scaler = scalers["vix"]
    df_copy.loc[:, "vix"] = scaler.transform(df_copy["vix"].to_numpy().reshape(-1, 1))
    ##

    return df_copy, scalers


def _get_new_scaler(scaling):
    if scaling == "minmax":
        return MinMaxScaler()
    elif scaling == "standard":
        return StandardScaler()
