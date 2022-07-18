import os

from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import Dict
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd

import processing
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
    first_n_windows: Optional[int] = None,
    data_dir: Optional[str] = "data",
) -> None:

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
        market_cap_min_mm, market_cap_max_mm, data_dir=data_dir
    )
    industries = list(tickers["subindustry"].unique())

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
        data_window = stonk_prices.iloc[:, index_start:index_end]
        assert data_window.shape[1] == total_backtest_days

        print(
            "Period "
            + str(data_window.columns[0])
            + " to "
            + str(data_window.columns[-1])
        )

        data_all_industries = []
        for industry in industries:
            tickers_by_industry = tickers[tickers["subindustry"] == industry]
            data_window_by_industry = data_window[
                data_window.index.isin(tickers_by_industry.index)
            ]
            X, Y = processing.combine_stonk_pairs(data_window_by_industry)

            data = pd.DataFrame(
                _data_collection_step(
                    X=X,
                    Y=Y,
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

            data["data_window_start"] = np.full(data.shape[0], X.columns[0])
            data["subindustry"] = np.full(data.shape[0], industry)

            data_all_industries.append(data)

            print(
                "Industries "
                + str(len(data_all_industries))
                + "/"
                + str(total_industries)
            )

        data_all_industries = pd.concat(data_all_industries, ignore_index=True)

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

        data_output_path = os.path.join(data_dir, "trades", filename)

        data_all_industries.to_csv(data_output_path, header=True, index=False)

        total_data_windows -= 1
        print("Remaining data windows: " + str(total_data_windows))

        if total_data_windows == 0:
            print("All done")
            return


def _data_collection_step(
    X: pd.DataFrame,
    Y: pd.DataFrame,
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

    residuals, betas, intercepts, dates_index = processing.get_rolling_residuals(
        X=X_until_T, Y=Y_until_T, l_reg=l_reg, l_roll=l_roll, dt=dt
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
    del adfs_raw
    del residuals

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

    betas_last = betas_last.loc[std_residuals.index]
    betas = betas.loc[std_residuals.index]
    intercepts_last = intercepts_last.loc[std_residuals.index]
    dates_index = dates_index.loc[std_residuals.index]

    means = means.loc[std_residuals.index]
    stds = stds.loc[std_residuals.index]

    ### Residuals mean-max calculations
    residuals_max_mean = processing.get_mean_residual_magnitude(
        std_residuals.to_numpy(), dt=mean_max_residual_dt
    )
    ###

    ### Beta stability calculations
    beta_stability_rsquared_vals = processing.calculate_beta_stability_rsquared(
        prices_X=X_until_T, prices_Y=Y_until_T, betas=betas, dates_index=dates_index
    )
    assert np.all(beta_stability_rsquared_vals.index == std_residuals.index)
    ###

    ### ARIMA model forecast calculations
    arima_forecast_diffs = processing.calculate_arima_forecast_diff(
        std_residuals=std_residuals,
        forecast_months=arima_forecast_months,
        eval_models=arima_eval_models,
    )
    ###

    ### Trade returns calculations
    # True for trades where we buy X and short Y
    buy_X = std_residuals.iloc[:, -1] > 0

    selected_trades_tickers = utils.separate_pair_index(std_residuals.index)
    X_from_T = X_from_T.loc[selected_trades_tickers["x"]]
    Y_from_T = Y_from_T.loc[selected_trades_tickers["y"]]

    trade_returns = processing.get_trades_returns(
        prices_X=X_from_T.to_numpy(),
        prices_Y=Y_from_T.to_numpy(),
        betas_YX=betas_last.to_numpy(),
        buy_X=buy_X.to_numpy(),
    )
    trade_residuals = processing.get_trades_residuals(
        prices_X=X_from_T.to_numpy(),
        prices_Y=Y_from_T.to_numpy(),
        betas_YX=betas_last.to_numpy(),
        intercepts_YX=intercepts_last.to_numpy(),
        means_YX=means.to_numpy(),
        stds_YX=stds.to_numpy(),
    )
    ###

    output_length = len(std_residuals)
    output["ticker_x"] = X_from_T.index
    output["ticker_y"] = Y_from_T.index
    output["trade_date"] = np.full(output_length, X_from_T.columns[0])

    output["adf_pass_rate"] = adfs[0].values.round(3)
    output["last_residual"] = std_residuals.iloc[:, -1].to_numpy().round(3)
    output["beta"] = betas_last[0].to_numpy().round(3)
    output["intercept"] = intercepts_last[0].to_numpy().round(3)
    output["residual_mean_max"] = np.full(output_length, residuals_max_mean)
    output["betas_rsquared"] = beta_stability_rsquared_vals[0].to_numpy()
    output["arima_forecast_diff"] = arima_forecast_diffs[0].to_numpy()

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

    return output
