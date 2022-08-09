import os
import pickle

from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import Dict
from typing import Any
from typing import Tuple
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn

import processing
import preprocessing
import utils

from processing import DAYS_IN_TRADING_YEAR
from processing import DAYS_IN_TRADING_MONTH
from processing import DAYS_IN_TRADING_WEEK


def data_collection_rolling_pipeline(
    stonk_prices: pd.DataFrame,
    l_reg: int,
    l_roll: int,
    dt: int,
    market_cap_min_mm: int,
    market_cap_max_mm: int,
    last_residual_cutoff: float,
    mean_max_residual_dt: int,
    adf_pval_cutoff: float,
    adf_pass_rate_filter: float,
    arima_forecast_months: int,
    arima_eval_models: int,
    trade_length_months: int,
    trading_interval_weeks: int,
    remove_industries: Optional[Iterable[str]] = None,
    first_n_windows: Optional[int] = None,
    data_dir: Optional[str] = "data",
    verbose: Optional[bool] = False,
) -> None:

    market_cap_max_string = (
        "max" if market_cap_max_mm is None else str(market_cap_max_mm)
    )
    data_output_dir = os.path.join(
        data_dir,
        "data_collection_pipeline",
        str(market_cap_min_mm) + "_to_" + market_cap_max_string,
    )

    # Adjust days so that they are divisible by dt
    l_reg_days = int(DAYS_IN_TRADING_YEAR * l_reg)
    l_reg_days -= l_reg_days % dt
    l_roll_days = int(DAYS_IN_TRADING_YEAR * l_roll)
    l_roll_days -= l_roll_days % dt

    total_days = l_reg_days + l_roll_days
    trade_length_days = DAYS_IN_TRADING_MONTH * trade_length_months
    trading_interval_days = DAYS_IN_TRADING_WEEK * trading_interval_weeks
    total_backtest_days = total_days + trade_length_days

    tickers = utils.get_ticker_names(
        market_cap_min_mm,
        market_cap_max_mm,
        data_dir=data_dir,
        remove_industries=remove_industries,
    )
    industries = list(tickers["subindustry"].unique())
    market_indexes = utils.get_market_indexes()

    data_range = range(
        stonk_prices.shape[1], total_backtest_days, -trading_interval_days
    )

    total_data_windows = len(list(data_range))
    total_industries = len(industries)

    if first_n_windows is not None:
        total_data_windows = min(first_n_windows, total_data_windows)

    print("Total data windows: " + str(total_data_windows))

    for index_end in data_range:
        index_start = index_end - total_backtest_days
        price_data_window = stonk_prices.iloc[:, index_start:index_end]
        assert price_data_window.shape[1] == total_backtest_days

        if verbose:
            print(
                "Period "
                + str(price_data_window.columns[0])
                + " to "
                + str(price_data_window.columns[-1])
            )

        collected_data_all_industries = []
        for industry in industries:
            tickers_by_industry = tickers[tickers["subindustry"] == industry]
            if len(tickers_by_industry) <= 1:
                continue
            price_data_window_by_industry = price_data_window[
                price_data_window.index.isin(tickers_by_industry.index)
            ]
            X, Y = processing.combine_stonk_pairs(price_data_window_by_industry)

            collected_data = pd.DataFrame(
                _data_collection_step(
                    X=X,
                    Y=Y,
                    market_indexes=market_indexes,
                    l_reg=l_reg,
                    l_roll=l_roll,
                    dt=dt,
                    last_residual_cutoff=last_residual_cutoff,
                    mean_max_residual_dt=mean_max_residual_dt,
                    adf_pval_cutoff=adf_pval_cutoff,
                    adf_pass_rate_filter=adf_pass_rate_filter,
                    arima_forecast_months=arima_forecast_months,
                    arima_eval_models=arima_eval_models,
                    trade_length_months=trade_length_months,
                )
            )

            collected_data["data_window_start"] = np.full(
                collected_data.shape[0], X.columns[0]
            )
            collected_data["subindustry"] = np.full(collected_data.shape[0], industry)

            collected_data_all_industries.append(collected_data)

            if verbose:
                print(
                    "Industries "
                    + str(len(collected_data_all_industries))
                    + "/"
                    + str(total_industries)
                )

        collected_data_all_industries = pd.concat(
            collected_data_all_industries, ignore_index=True
        )

        filename = (
            "_".join(
                [
                    str(data_window.columns[0]),
                    str(data_window.columns[-1]),
                    str(l_reg),
                    str(l_roll),
                    str(dt),
                    str(market_cap_min_mm),
                    str(market_cap_max_mm),
                    str(last_residual_cutoff),
                    str(mean_max_residual_dt),
                    str(adf_pval_cutoff),
                    str(adf_pass_rate_filter),
                    str(arima_forecast_months),
                    str(arima_eval_models),
                    str(trade_length_months),
                    str(trading_interval_weeks),
                ]
            )
            + ".csv"
        )

        data_output_file_path = os.path.join(data_output_dir, filename)

        collected_data_all_industries.to_csv(
            data_output_file_path, header=True, index=False
        )

        total_data_windows -= 1

        if verbose:
            print("Remaining data windows: " + str(total_data_windows))

        if total_data_windows == 0:
            print("All done")
            return


def _data_collection_step(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    market_indexes: pd.DataFrame,
    l_reg: int,
    l_roll: int,
    dt: int,
    last_residual_cutoff: float,
    mean_max_residual_dt: int,
    adf_pval_cutoff: float,
    adf_pass_rate_filter: float,
    arima_forecast_months: int,
    arima_eval_models: int,
    trade_length_months: int,
) -> Dict[str, ArrayLike]:
    assert X.shape == Y.shape

    output = {}

    trade_length_days = trade_length_months * DAYS_IN_TRADING_MONTH

    X_until_T = X.iloc[:, :-trade_length_days].copy()
    Y_until_T = Y.iloc[:, :-trade_length_days].copy()

    X_from_T = (
        X.iloc[:, -trade_length_days - 1 :]
        .reset_index()
        .drop_duplicates(subset="index")
        .set_index("index")
        .copy()
    )
    Y_from_T = (
        Y.iloc[:, -trade_length_days - 1 :]
        .reset_index()
        .drop_duplicates(subset="index")
        .set_index("index")
        .copy()
    )

    # X and Y dimensions must match
    assert X_from_T.shape == Y_from_T.shape and X_until_T.shape == Y_until_T.shape

    # Check whether enough days' worth of data was given
    assert (
        X_from_T.shape[1] == trade_length_days + 1
        and X.shape[1] == X_until_T.shape[1] + X_from_T.shape[1] - 1
    )

    del X
    del Y

    features = process_features_from_price_data(
        X=X_until_T,
        Y=Y_until_T,
        market_indexes=market_indexes,
        l_reg=l_reg,
        l_roll=l_roll,
        dt=dt,
        last_residual_cutoff=last_residual_cutoff,
        adf_pval_cutoff=adf_pval_cutoff,
        adf_pass_rate_filter=adf_pass_rate_filter,
        mean_max_residual_dt=mean_max_residual_dt,
        arima_forecast_months=arima_forecast_months,
        arima_eval_models=arima_eval_models,
    )

    ### Trade returns calculations
    # True for trades where we buy X and short Y
    buy_X = features["std_residuals"].iloc[:, -1] > 0

    selected_trades_tickers = utils.separate_pair_index(features["std_residuals"].index)
    X_from_T = X_from_T.loc[selected_trades_tickers["x"]]
    Y_from_T = Y_from_T.loc[selected_trades_tickers["y"]]

    trade_returns = processing.get_trades_returns(
        prices_X=X_from_T.to_numpy(),
        prices_Y=Y_from_T.to_numpy(),
        betas_YX=features["betas_last"].to_numpy(),
        buy_X=buy_X.to_numpy(),
    )
    trade_residuals = processing.get_trades_residuals(
        prices_X=X_from_T.to_numpy(),
        prices_Y=Y_from_T.to_numpy(),
        betas_YX=features["betas_last"].to_numpy(),
        intercepts_YX=features["intercepts_last"].to_numpy(),
        means_YX=features["means"].to_numpy(),
        stds_YX=features["stds"].to_numpy(),
    )
    ###

    output_length = len(features["std_residuals"])
    output["ticker_x"] = X_from_T.index
    output["ticker_y"] = Y_from_T.index
    output["trade_date"] = np.full(output_length, X_from_T.columns[0])

    output["adf_pass_rate"] = features["adfs"][0].values.round(3)
    output["last_residual"] = features["std_residuals"].iloc[:, -1].to_numpy().round(3)
    output["beta"] = features["betas_last"][0].to_numpy().round(3)
    output["intercept"] = features["intercepts_last"][0].to_numpy().round(3)
    output["residual_mean_max"] = np.full(output_length, features["residuals_max_mean"])
    output["betas_rsquared"] = features["beta_stability_rsquared_vals"][0].to_numpy()
    output["arima_forecast"] = features["arima_forecasts"][0].to_numpy()

    for corr_feature in features["market_correlations"].columns:
        output[corr_feature] = features["market_correlations"].loc[:, corr_feature]

    output["return_one_month"] = trade_returns[:, 1 * DAYS_IN_TRADING_MONTH]
    output["residual_one_month"] = trade_residuals[:, 1 * DAYS_IN_TRADING_MONTH]
    if trade_length_months > 1:
        output["return_two_month"] = trade_returns[:, 2 * DAYS_IN_TRADING_MONTH]
        output["residual_two_month"] = trade_residuals[:, 2 * DAYS_IN_TRADING_MONTH]
    else:
        output["return_two_month"] = np.full(output_length, np.nan)
        output["residual_two_month"] = np.full(output_length, np.nan)

    if trade_length_months > 2:
        output["return_three_month"] = trade_returns[:, 3 * DAYS_IN_TRADING_MONTH]
        output["residual_three_month"] = trade_residuals[:, 3 * DAYS_IN_TRADING_MONTH]
    else:
        output["return_three_month"] = np.full(output_length, np.nan)
        output["residual_three_month"] = np.full(output_length, np.nan)

    del features
    return output


def process_features_from_price_data(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    market_indexes: pd.DataFrame,
    l_reg: int,
    l_roll: int,
    dt: int,
    last_residual_cutoff: float,
    adf_pval_cutoff: float,
    adf_pass_rate_filter: float,
    mean_max_residual_dt: float,
    arima_forecast_months: int,
    arima_eval_models: int,
) -> Dict[str, ArrayLike]:
    assert X.shape == Y.shape and len(X.shape) == 2

    residuals, betas, intercepts, dates_index = processing.get_rolling_residuals(
        X=X, Y=Y, l_reg=l_reg, l_roll=l_roll, dt=dt
    )

    std_residuals, means, stds = processing.get_standardized_residuals(residuals)

    # Filter residuals to only select relevant trades that fit the theoretical assumptions
    std_residuals = std_residuals[
        std_residuals.iloc[:, -1].abs() >= last_residual_cutoff
    ]

    if len(std_residuals) == 0:
        return {}

    residuals = residuals.loc[std_residuals.index]
    betas = betas.loc[std_residuals.index]
    intercepts = intercepts.loc[std_residuals.index]
    dates_index = dates_index.loc[std_residuals.index]

    adfs, adfs_raw = processing.get_aggregate_adfs(
        residuals, betas=betas, cutoff=adf_pval_cutoff
    )

    # Select betas and intercepts from the most recent regressions
    betas_last = processing.get_last_pairs(betas)
    intercepts_last = processing.get_last_pairs(intercepts)

    assert np.all(std_residuals.index == betas_last.index)
    assert np.all(adfs.index == std_residuals.index)
    assert np.all(intercepts_last.index == betas_last.index)

    # Select trades that are above the specified ADF pass rate
    selected_by_adf = (adfs >= adf_pass_rate_filter).values

    adfs = adfs[selected_by_adf]
    std_residuals = std_residuals[selected_by_adf]

    if len(std_residuals) == 0:
        return {}

    residuals = residuals.loc[adfs.index]
    adfs_raw = adfs_raw.loc[adfs.index]
    betas = betas.loc[adfs.index]
    dates_index = dates_index.loc[adfs.index]

    betas_last = betas_last.loc[adfs.index]
    intercepts_last = intercepts_last.loc[adfs.index]

    means = means.loc[adfs.index]
    stds = stds.loc[adfs.index]

    ### Residuals mean-max calculations
    residuals_max_mean = processing.get_mean_residual_magnitude(
        std_residuals.to_numpy(), dt=mean_max_residual_dt
    )
    ###

    ### Beta stability calculations
    beta_stability_rsquared_vals = processing.calculate_beta_stability_rsquared(
        prices_X=X, prices_Y=Y, betas=betas, dates_index=dates_index
    )
    assert np.all(beta_stability_rsquared_vals.index == std_residuals.index)
    ###

    ### ARIMA model forecast calculations
    arima_forecasts = processing.calculate_arima_forecast(
        std_residuals=std_residuals,
        forecast_months=arima_forecast_months,
        eval_models=arima_eval_models,
    )
    ###

    ### Market correlations calculations
    correlations_market_research = (
        processing.get_correlation_with_historical_market_factors_research(
            std_residuals=std_residuals, dates_index=dates_index
        )
    )
    correlations_market_indexes = processing.get_correlation_with_live_market_indexes(
        std_residuals=std_residuals,
        dates_index=dates_index,
        market_indexes=market_indexes,
    )

    market_correlations = pd.concat(
        (correlations_market_research, correlations_market_indexes), axis="columns"
    )
    ###

    return {
        "residuals": residuals,
        "std_residuals": std_residuals,
        "betas": betas,
        "betas_last": betas_last,
        "intercepts": intercepts,
        "intercepts_last": intercepts_last,
        "adfs": adfs,
        "adfs_raw": adfs_raw,
        "means": means,
        "stds": stds,
        "dates_index": dates_index,
        "residuals_max_mean": residuals_max_mean,
        "beta_stability_rsquared_vals": beta_stability_rsquared_vals,
        "arima_forecasts": arima_forecasts,
        "market_correlations": market_correlations,
    }


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
