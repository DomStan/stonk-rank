import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from pmdarima.arima import auto_arima

from numpy.typing import ArrayLike
from typing import Tuple

import utils

DAYS_IN_TRADING_YEAR = 252
DAYS_IN_TRADING_MONTH = 21
DAYS_IN_TRADING_WEEK = 5


def combine_stonk_pairs(
    stonks_prices: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO: make combinations with numpy
    # All ticker names must be unique
    assert all(stonks_prices.index.unique() == stonks_prices.index)

    # Check that there aren't too many combinations that would cause extreme lag
    assert len(stonks_prices) < 400

    combs = np.asarray(list(combinations(stonks_prices.index.unique(), 2)))

    return stonks_prices.loc[combs[:, 0]], stonks_prices.loc[combs[:, 1]]


def get_residuals_many(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[np.ndarray, np.array, np.array]:
    """Vectorized calculation of residuals from many univariate linear regressions.
    Args:
        X (numpy array of shape (n_pairs, d_time)): matrix of LR inputs X, each row represents a different regression, corresponding to the same rows in Y
        Y (numpy array of shape (n_pairs, d_time)): matrix of LR inputs Y, each row represents a different regression, corresponding to the same rows in X
    Returns:
        residuals (numpy array of shape (n_pairs, d_time)): matrix of resulting residuals between vectorized pairs of X and Y
        betas (numpy array of shape (n_pairs, 1)): beta coefficients for each linear regression
        Y_hat (numpy array of shape (n_pairs, d_time)): predictions using X
    """
    # Stack 2D matrices into 3D matrices
    X = X.reshape(np.shape(X)[0], np.shape(X)[1], -1)
    Y = Y.reshape(np.shape(Y)[0], np.shape(Y)[1], -1)

    # Add bias/intercept in the form (Xi, 1)
    Z = np.concatenate(
        [X, np.ones((np.shape(X)[0], np.shape(X)[1], 1), dtype=np.float32)],
        axis=2,
    )
    

    # Save the transpose as it's used a couple of times
    Z_t = Z.transpose(0, 2, 1)

    # Linear Regression equation solutions w.r.t. weight matrix
    # W contains (beta_coef, a_intercept) for each regression
    try:
        W = np.matmul(np.linalg.inv(np.matmul(Z_t, Z)), np.matmul(Z_t, Y))
    except:
        # Fallback to non-vectorized calculation using sklearn
        return _get_sklearn_residuals_many(X=X, Y=Y)
        
    del X
    del Z_t

    # Predictions and residuals
    # Y_hat = np.matmul(Z, W).round(2)
    residuals = (Y - np.matmul(Z, W)).round(3)
    del Y
    del Z
    assert residuals.dtype == np.float32

    # Y_hat returned for debugging purposes
    # return (residuals[:, :, 0], W[:, 0, 0], Y_hat[:, :, 0])
    return (residuals[:, :, 0], W[:, 0, 0], W[:, 1, 0])

def _get_sklearn_residuals_many(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.array, np.array]:
    lr = LinearRegression(n_jobs=-1, fit_intercept=True)
    X = X.reshape((X.shape[0], X.shape[1], -1))
    Y = Y.reshape((Y.shape[0], Y.shape[1], -1))
    
    preds = []
    res = []
    betas = []
    intercepts = []
    for i in range(X.shape[0]):
        lr.fit(X[i], Y[i])
        preds.append(lr.predict(X[i]).round(3))
        res.append(Y[i]-preds[-1])
        betas.append(lr.coef_[0][0])
        intercepts.append(lr.intercept_)
    return (np.asarray(res)[:,:,0], np.asarray(betas).ravel(), np.asarray(intercepts).ravel())


def get_rolling_residuals(
    X: pd.DataFrame, Y: pd.DataFrame, l_reg: int, l_roll: int, dt: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculates rolling window residuals in vectorized form. Returns the result as an array that repeats each ticker for the number of regressions calculated.
    For example, if the inputs are (Pair A, Pair B, Pair C) and l_roll / dt = 3, then the returned results will have the form as follows:
    (Pair A, Pair A, Pair A, Pair B, Pair B, Pair B, Pair C, Pair C, Pair C)
    Works best when l_reg and l_roll are integers.
        Args:
        - X, Y (DataFrame of shape (n_pairs, >= l_reg + l_roll)): matrix of LR inputs X, Y; each row containing at least the complete data period for rolling regressions (can be longer)
        - l_reg (float): length of each LR to calculate residuals, in years; will be multiplied by the adjusted number of days in a trading year
        - l_roll (float): length of rolling window, in years; will be multipled by the adjusted number of days in a trading year
        - dt (int): rolling window step size, in trading days; total trading year days will be reduced to be divisible by dt (by not more than the value of dt)
        Returns:
        - residuals (numpy array of shape (n_pairs * (l_roll/dt)+1, l_reg + l_roll)): matrix of resulting residuals between vectorized pairs of X and Y
        - betas (numpy array of shape (n_pairs * (l_roll/dt)+1, 1)): beta coefficients for each linear regression
        - Y_hat (numpy array of shape (n_pairs * (l_roll/dt)+1, l_reg + l_roll)): predictions using X
    """

    # Adjust days so that they are divisible by dt
    l_reg_days = int(DAYS_IN_TRADING_YEAR * l_reg)
    l_reg_days -= l_reg_days % dt
    l_roll_days = int(DAYS_IN_TRADING_YEAR * l_roll)
    l_roll_days -= l_roll_days % dt

    total_days = l_reg_days + l_roll_days

    # Number of regressions for each ticker
    n_windows = (l_roll_days // dt) + 1

    # Number of tickers
    n_x = X.shape[0]

    # Take the dates, create an empty array for windowed dates
    date_index = X.columns[-total_days:]
    date_index_windowed = np.empty(shape=(n_x * n_windows, 2), dtype="O")

    # Repeat each ticker name times n_windows
    X_index = np.repeat(X.index, n_windows)
    Y_index = np.repeat(Y.index, n_windows)

    # X and Y must have the same dates
    assert np.array_equal(X.columns, Y.columns)

    X = X.to_numpy(dtype=np.float32)
    Y = Y.to_numpy(dtype=np.float32)

    # Rolling window length must be divisible by dt
    assert (l_roll_days % dt) == 0

    # There has to be enough days' worth of data in X (and Y) and their shapes must match
    assert X.shape == Y.shape and X.shape[1] >= total_days

    # Take the total_days from the end of the arrays (most recent days first, oldest days at the end are cut off)
    X = X[:, -total_days:]
    Y = Y[:, -total_days:]

    # Create empty arrays that will contain windowed slices of our data
    X_windows = np.empty(shape=(n_x * n_windows, l_reg_days), dtype=np.float32)
    Y_windows = np.empty(shape=(n_x * n_windows, l_reg_days), dtype=np.float32)

    # Take windowed slices and place them into the created empty arrays
    for n in range(n_x):
        for i in range(n_windows):
            n_i = (n * n_windows) + i
            t_i = i * dt
            t_y = t_i + l_reg_days

            X_windows[n_i] = X[n, t_i:t_y]
            Y_windows[n_i] = Y[n, t_i:t_y]
            date_index_windowed[n_i, 0] = date_index[t_i]
            date_index_windowed[n_i, 1] = date_index[t_y - 1]

    # Make sure we've got the windowing dimensions right
    assert X_windows.shape == (n_x * n_windows, l_reg_days,) and Y_windows.shape == (
        n_x * n_windows,
        l_reg_days,
    )

    # Sanity checks
    assert all(
        [
            X[0, -1] == X_windows[n_windows - 1, -1],
            Y[0, -1] == Y_windows[n_windows - 1, -1],
            X[-1, -1] == X_windows[-1, -1],
            Y[-1, -1] == Y_windows[-1, -1],
        ]
    )

    # Construct ticker pair index column
    pair_index = np.array(
        pd.DataFrame(np.array([Y_index, X_index])).apply("_".join, axis=0, raw=True)
    )

    # Construct regression date range index column
    date_index = np.array(
        pd.DataFrame(
            np.array([date_index_windowed[:, 0], date_index_windowed[:, 1]])
        ).apply("_".join, axis=0, raw=True)
    )

    # Lengths of indexes must match
    assert len(pair_index) == len(date_index)

    # Calculate and return the residuals
    residuals, betas, intercepts = get_residuals_many(X_windows, Y_windows)

    residuals = pd.DataFrame(residuals, index=pair_index)
    betas = pd.DataFrame(betas, index=pair_index)
    intercepts = pd.DataFrame(intercepts, index=pair_index)
    date_index = pd.DataFrame(date_index, index=pair_index)

    return residuals, betas, intercepts, date_index


def get_aggregate_adfs(
    residuals: pd.DataFrame,
    betas: pd.DataFrame = None,
    cutoff: float = 0.1,
    adf_regression: str = "c",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _get_adfs(residuals: np.ndarray, adf_regression: str) -> np.ndarray:
        # Get ADF test p-values for each row of the residuals array. No autolag (maxlag always used)
        assert residuals.dtype == np.float32
        return np.apply_along_axis(
            lambda x: adfuller(x, regression=adf_regression, autolag=None)[1],
            axis=1,
            arr=residuals,
        )

    # Get ADF p-values
    adfs = _get_adfs(
        residuals.to_numpy(),
        adf_regression=adf_regression,
    ).reshape((-1, 1))

    # Add ones to ADF values where betas are negative, if betas are given
    if betas is not None:
        # Must be the same number of columns
        assert adfs.shape[0] == betas.shape[0]
        # Residuals and betas must have the same index names
        assert np.all(residuals.index == betas.index)
        # Add 1's to p-values where betas are negative
        adfs = adfs + (betas[0].to_numpy() <= 0).reshape((-1, 1))

    # Make a copy for returning, CSV output
    adfs_raw = pd.DataFrame(adfs.copy(), index=residuals.index)

    # All unique ticker pairs, in original order
    unique_pairs = residuals.index.unique()

    # Number of regressions for one pair
    pairs_per_index = len(residuals) // len(unique_pairs)

    # Reshape into a 3D array for averaging ADF values along the second axis
    adfs = adfs.reshape((len(unique_pairs), pairs_per_index, 1))

    # Takes cutoff, averages along the pairs_per_index (second) axis
    adfs = (adfs <= cutoff).mean(axis=1)

    # Probably always true, but just in case
    assert adfs.shape[0] == len(unique_pairs)

    # Back to a DataFrame with named indexes
    adfs = pd.DataFrame(adfs, index=unique_pairs)

    return adfs, adfs_raw


def get_last_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    # Get unique ticker pairs, in preserved order
    unique_pairs = pairs.index.unique()

    # Number of samples per ticker pair
    pairs_per_index = len(pairs) // len(unique_pairs)

    # Must be an equal number of pairs per index
    assert pairs_per_index * len(unique_pairs) == len(pairs)

    # Slice taking only the last element for each ticker pair
    last_pairs = pairs.iloc[pairs_per_index - 1 : len(pairs) : pairs_per_index].copy()

    # Make sure we got the slices right
    assert np.all(last_pairs.index == unique_pairs) and np.all(
        pairs.iloc[-1] == last_pairs.iloc[-1]
    )

    return last_pairs


def get_standardized_residuals(
    residuals: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Get the last regression for each spread
    last_reg_pairs = get_last_pairs(residuals)

    # Get unique ticker pairs
    unique_pairs = last_reg_pairs.index.copy()

    # Convert to numpy
    last_reg_pairs = last_reg_pairs.to_numpy()

    # Standardize
    means = last_reg_pairs.mean(axis=1, keepdims=True)
    stds = last_reg_pairs.std(axis=1, keepdims=True)
    last_reg_pairs = (last_reg_pairs - means) / stds

    # Back to a DataFrame with named indexes
    last_reg_pairs = pd.DataFrame(last_reg_pairs, index=unique_pairs)
    means = pd.DataFrame(means, index=unique_pairs)
    stds = pd.DataFrame(stds, index=unique_pairs)

    return (last_reg_pairs, means, stds)


def get_mean_residual_magnitude(std_residuals: np.ndarray, dt: int) -> float:
    # Assume there is enough days' worth of data for averaging over dt days
    assert std_residuals.shape[1] >= dt

    # Take the absolute maximum for each day, over all tickers, mean over the results
    mean_magnitude = np.round(np.abs(std_residuals[:, -dt:]).max(axis=0).mean(), 2)

    return mean_magnitude


def get_trades_returns(
    prices_X: np.ndarray,
    prices_Y: np.ndarray,
    betas_YX: np.ndarray,
    buy_X: np.ndarray,
) -> np.ndarray:
    # Sanity checks
    assert all([prices_X.shape == prices_Y.shape, len(buy_X) == betas_YX.shape[0]])

    # Save entering prices at t=0
    initial_prices_X = prices_X[:, [0]]
    initial_prices_Y = prices_Y[:, [0]]

    # Initial proportional values of trades at t=0, X prices scaled by beta
    initial_trade_values = (initial_prices_X * betas_YX) + initial_prices_Y

    # Returns for X, Y trades each day. X prices scaled by beta
    returns_X = betas_YX * (prices_X - initial_prices_X)
    returns_Y = prices_Y - initial_prices_Y

    # Negate short trades
    returns_X[~buy_X] = -returns_X[~buy_X]
    returns_Y[buy_X] = -returns_Y[buy_X]

    # Add the trade returns for X, Y, divide by initial investment values to get profit/loss %
    trade_returns = ((returns_X + returns_Y) / initial_trade_values).round(3)

    return trade_returns


def get_trades_residuals(
    prices_X: np.ndarray,
    prices_Y: np.ndarray,
    betas_YX: np.ndarray,
    intercepts_YX: np.ndarray,
    means_YX: np.ndarray,
    stds_YX: np.ndarray,
) -> np.ndarray:

    # Sanity checks
    assert all(
        [
            prices_X.shape == prices_Y.shape,
            betas_YX.shape[0]
            == intercepts_YX.shape[0]
            == means_YX.shape[0]
            == stds_YX.shape[0]
            == prices_X.shape[0],
            betas_YX.shape[1]
            == intercepts_YX.shape[1]
            == means_YX.shape[1]
            == stds_YX.shape[1]
            == 1,
        ]
    )

    # Calculate residuals from regression form: Y = bX + a
    trade_residuals = prices_Y - ((betas_YX * prices_X) + intercepts_YX)

    # Standardize the residuals
    trade_residuals = ((trade_residuals - means_YX) / stds_YX).round(3)

    return trade_residuals


def calculate_beta_stability_rsquared(
    prices_X: pd.DataFrame,
    prices_Y: pd.DataFrame,
    betas: pd.DataFrame,
    dates_index: pd.DataFrame,
) -> np.array:
    betas = betas.copy()

    betas["dates"] = dates_index.values
    first_reg_date = betas["dates"][0].split("_")[0]
    last_reg_date = betas["dates"][-1].split("_")[-1]

    betas["dates_end"] = betas["dates"].map(lambda x: x.split("_")[-1])
    betas = betas.drop(columns="dates")
    betas_original_order = betas.index.unique()

    selected_tickers = utils.separate_pair_index(betas_original_order)
    prices_X_selected = (
        prices_X.reset_index()
        .drop_duplicates(subset="index")
        .set_index("index")
        .copy()
        .loc[selected_tickers["x"], first_reg_date:last_reg_date]
    )
    prices_Y_selected = (
        prices_Y.reset_index()
        .drop_duplicates(subset="index")
        .set_index("index")
        .copy()
        .loc[selected_tickers["y"], first_reg_date:last_reg_date]
    )

    betas = (
        betas.reset_index()
        .pivot(index="index", columns="dates_end", values=0)
        .loc[betas_original_order]
    )

    assert np.all(
        (prices_Y_selected.index + "_" + prices_X_selected.index) == betas.index
    )

    # Finds the index of the closest earlier (ffill) date than the one given in pandas_index
    def _get_closest_loc(pandas_index: pd.Index, value: str) -> int:
        try:
            return pandas_index.get_loc(value, method="ffill")
        except KeyError:
            return 0

    betas = betas.iloc[
        :, prices_X_selected.columns.map(lambda x: _get_closest_loc(betas.columns, x))
    ].values

    assert betas.shape == prices_X_selected.shape

    last_betas = betas[:, -1].copy().reshape(-1, 1)

    assert all(
        [
            prices_X_selected.shape == prices_Y_selected.shape,
            len(last_betas) == len(prices_X_selected),
        ]
    )

    spreads_X = prices_Y_selected.values - (betas * prices_X_selected.values)
    spreads_Y = prices_Y_selected.values - (last_betas * prices_X_selected.values)

    lr = LinearRegression(n_jobs=-1)
    rsquared_vals = pd.DataFrame(
        [
            lr.fit(x.reshape(-1, 1), y.reshape(-1, 1)).score(
                x.reshape(-1, 1), y.reshape(-1, 1)
            )
            for x, y in zip(spreads_X, spreads_Y)
        ],
        index=betas_original_order,
        dtype=np.float32,
    )
    return rsquared_vals


def calculate_arima_forecast(
    std_residuals: pd.DataFrame, forecast_months: int = 3, eval_models: int = 5
) -> np.array:
    forecast_length = forecast_months * DAYS_IN_TRADING_MONTH

    arima_forecasts = np.apply_along_axis(
        lambda x: auto_arima(
            y=x,
            seasonal=False,
            stationary=True,
            information_criterion="aic",
            with_intercept=False,
            maxiter=eval_models,
            d=0,
        ).fit_predict(y=x, n_periods=forecast_length)[-1],
        axis=1,
        arr=std_residuals.values,
    )

    arima_forecasts = pd.DataFrame(
        arima_forecasts,
        index=std_residuals.index,
        dtype=np.float32,
    )

    return arima_forecasts
