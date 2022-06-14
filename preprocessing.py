import numpy as np
import pandas as pd

from typing import Optional
from typing import Dict
from typing import Union
from typing import Tuple

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def split_data_by_time(
    df: pd.DataFrame, hold_out_time_slices: int, seed: Optional[int] = 420
) -> Dict[str, pd.DataFrame]:
    df_copy = df.copy()
    t_test = hold_out_time_slices // 2

    # Shuffle data
    df_copy = df_copy.sample(frac=1, random_state=seed).reset_index(drop=True)

    dates_sorted = np.sort(df["trade_date"].unique())

    dates_valid = dates_sorted[-hold_out_time_slices:-t_test]
    dates_test = dates_sorted[-t_test:]
    dates_train = dates_sorted[:-hold_out_time_slices]

    # Sliced dates should not overlap
    assert len(set(dates_valid) & set(dates_test) & set(dates_train)) == 0

    df_validation = (
        df_copy[df_copy.trade_date.isin(dates_valid)]
        .copy()
        .sample(frac=1, random_state=seed)
    )
    df_test = (
        df_copy[df_copy.trade_date.isin(dates_test)]
        .copy()
        .sample(frac=1, random_state=seed)
    )
    df_train = (
        df_copy[df_copy.trade_date.isin(dates_train)]
        .copy()
        .sample(frac=1, random_state=seed)
    )

    # No intersection
    assert (
        len(
            set(df_validation.trade_date.unique())
            & set(df_test.trade_date.unique())
            & set(df_train.trade_date.unique())
        )
        == 0
    )

    assert len(df) == sum([len(df_train), len(df_validation), len(df_test)])

    return {"train": df_train, "validation": df_validation, "test": df_test}


def split_data(
    df: pd.DataFrame,
    date_count_valid: int,
    date_count_gap: int,
    date_remove_train: int = 0,
    random_state: int = None
) -> Dict[str, pd.DataFrame]:
    df_copy = df.copy()

    # Shuffle data
    df_copy = df_copy.sample(frac=1, random_state=random_state).reset_index(drop=True)

    dates_sorted = np.sort(df["trade_date"].unique())
    
    total_date_count = len(dates_sorted)
    
    date_count_train = total_date_count - date_count_valid - date_count_gap

    assert date_count_train > 0 and sum([date_count_train, date_count_valid, date_count_gap]) <= total_date_count

    dates_valid = dates_sorted[date_count_train+date_count_gap : date_count_train+date_count_gap+date_count_valid]
    dates_train = dates_sorted[date_remove_train:date_count_train]
    
    assert len(dates_valid) == date_count_valid and len(dates_train) == date_count_train - date_remove_train
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
    assert len(set(df_valid.trade_date.unique()) & set(df_train.trade_date.unique())) == 0

    assert not np.any(df_valid.index.isin(df_train.index))

    return {"train": df_train, "validation": df_valid}


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    def label_one_example(example):
        success_one_month = (
            example["return_one_month"] >= 0.045
            and np.abs(example["last_residual"] - example["residual_one_month"]) >= 2
        )
        success_two_month = (
            example["return_two_month"] >= 0.065
            and np.abs(example["last_residual"] - example["residual_two_month"]) >= 2
        )
        success_three_month = (
            example["return_three_month"] >= 0.095
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
    df: pd.DataFrame,
    scalers: Optional[Dict[str, sklearn.base.TransformerMixin]] = None,
    scaling: Optional[str] = None,
    add_noise: Optional[bool] = False,
) -> Tuple[
    pd.DataFrame, Union[None, Dict[str, sklearn.base.TransformerMixin]],
]:
    def _map_industry(example):
        mappings = {
            "chemicals": "materials",
            "construction_materials": "materials",
            "containers_and_packaging": "materials",
            "metals_and_mining": "materials",
            "paper_and_forest_products": "materials",
            #
            "capital_goods": "industrials",
            "commercial_and_professional_services": "industrials",
            "transportation": "industrials",
            #
            "automobiles_and_components": "consumer_discretionary",
            "consumer_durables_and_apparel": "consumer_discretionary",
            "consumer_services": "consumer_discretionary",
            "retailing": "consumer_discretionary",
            #
            "health_care_equipment_and_services": "health_care",
            "pharmaceuticals_biotechnology_and_life_sciences": "health_care",
            #
            "banks": "financials",
            "diversified_financials": "financials",
            "insurance": "financials",
            #
            "software_and_services": "information_technology",
            "technology_hardware_and_equipment": "information_technology",
            "semiconductors_and_semiconductor_equipment": "information_technology",
            #
            "telecommunication_services": "communication_services",
            "media_and_entertainment": "communication_services",
            #
            "energy": "energy",
            "utilities": "utilities",
            "real_estate": "real_estate",
            "consumer_staples": "consumer_staples",
        }
        return mappings[example["subindustry"]]

    if scalers is None and scaling is not None:
        FIT_SCALERS = True
        scalers = {}
    else:
        FIT_SCALERS = False

    def _get_new_scaler():
        if scaling == "minmax":
            return MinMaxScaler()
        elif scaling == "standard":
            return StandardScaler()

    data_types = {
        "adf_pass_rate": np.float32,
        "last_residual": np.float32,
        "residual_mean_max": np.float32,
    }

    df_copy = df.copy()
    n_x = df_copy.shape[0]

    # Select features
    df_copy = df_copy[
        ["adf_pass_rate", "last_residual", "residual_mean_max", "subindustry"]
    ].astype(data_types)

    # Industry transform
    df_copy.loc[:, "industry"] = df_copy.apply(_map_industry, axis=1)
    df_copy = df_copy.drop(columns="subindustry")
    df_copy["industry"] = df_copy["industry"].astype("category")

    # Residual transform
    df_copy.loc[:, "last_residual"] = df_copy["last_residual"].abs()

    # Feature crossing
    df_copy.loc[:, "residual_inter"] = (
        df_copy["last_residual"] / df_copy["residual_mean_max"]
    )

    if add_noise:
        df_copy.loc[:, "adf_pass_rate"] += np.random.normal(0, 0.01, n_x)
        df_copy.loc[:, "last_residual"] += np.random.normal(0, 0.01, n_x)
        df_copy.loc[:, "residual_mean_max"] += np.random.normal(0, 0.01, n_x)

    # Scale features
    if scalers is not None:
        # Last residual scaling
        if FIT_SCALERS:
            scalers["last_residual"] = _get_new_scaler().fit(
                df_copy["last_residual"].to_numpy().reshape(-1, 1)
            )
            scalers["adf_pass_rate"] = _get_new_scaler().fit(
                df_copy["adf_pass_rate"].to_numpy().reshape(-1, 1)
            )
            scalers["residual_mean_max"] = _get_new_scaler().fit(
                df_copy["residual_mean_max"].to_numpy().reshape(-1, 1)
            )

        scaler = scalers["last_residual"]
        df_copy.loc[:, "last_residual"] = scaler.transform(
            df_copy["last_residual"].to_numpy().reshape(-1, 1)
        )

        # ADF pass rate scaling
        scaler = scalers["adf_pass_rate"]
        df_copy.loc[:, "adf_pass_rate"] = scaler.transform(
            df_copy["adf_pass_rate"].to_numpy().reshape(-1, 1)
        )

        # Residual mean max scaling
        scaler = scalers["residual_mean_max"]
        df_copy.loc[:, "residual_mean_max"] = scaler.transform(
            df_copy["residual_mean_max"].to_numpy().reshape(-1, 1)
        )

    return df_copy, scalers
