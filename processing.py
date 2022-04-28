import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import adfuller

DAYS_IN_TRADING_YEAR = 252
DAYS_IN_TRADING_MONTH = 21
DAYS_IN_TRADING_WEEK = 5


def combine_stonk_pairs(stonks_prices):
    # TODO: make combinations with numpy
    # All ticker names must be unique
    assert all(stonks_prices.index.unique() == stonks_prices.index)
    
    # Check that there isn't too many combinations that would cause lag
    assert(len(stonks_prices) < 400)
    
    combs = np.asarray(list(combinations(stonks_prices.index.unique(), 2)))
    
    return stonks_prices.loc[combs[:, 0]], stonks_prices.loc[combs[:, 1]]


def get_residuals_many(X, Y):
    '''Vectorized calculation of residuals from many univariate linear regressions.
    Args:
        X (numpy array of shape (n_pairs, d_time)): matrix of LR inputs X, each row represents a different regression, corresponding to the same rows in Y
        Y (numpy array of shape (n_pairs, d_time)): matrix of LR inputs Y, each row represents a different regression, corresponding to the same rows in X
    Returns:
        residuals (numpy array of shape (n_pairs, d_time)): matrix of resulting residuals between vectorized pairs of X and Y
        betas (numpy array of shape (n_pairs, 1)): beta coefficients for each linear regression
        Y_hat (numpy array of shape (n_pairs, d_time)): predictions using X
    '''
    # Stack 2D matrices into 3D matrices
    X = X.reshape(np.shape(X)[0], np.shape(X)[1], -1)
    Y = Y.reshape(np.shape(Y)[0], np.shape(Y)[1], -1)
    
    # Add bias/intercept in the form (Xi, 1)
    Z = np.concatenate([X, np.ones((np.shape(X)[0], np.shape(X)[1], 1), dtype=np.float32)], axis=2)
    del X
    
    # Save the transpose as it's used a couple of times
    Z_t = Z.transpose(0, 2, 1)
    
    # Linear Regression equation solutions w.r.t. weight matrix
    # W contains (beta_coef, a_intercept) for each regression
    W = np.matmul(np.linalg.inv(np.matmul(Z_t, Z)),  np.matmul(Z_t, Y))
    del Z_t
    
    # Predictions and residuals
    # Y_hat = np.matmul(Z, W).round(2)
    residuals = (Y - np.matmul(Z, W)).round(2)
    del Y
    del Z
    assert residuals.dtype == np.float32
    
    # Y_hat returned for debugging purposes
    # return (residuals[:, :, 0], W[:, 0, 0], Y_hat[:, :, 0])
    return (residuals[:, :, 0], W[:, 0, 0],  W[:, 1, 0])


def get_rolling_residuals(X, Y, l_reg, l_roll, dt):
    '''
    Calculates rolling window residuals in vectorized form. Returns the result as an array that repeats each ticker for the number of regressions calculated.
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
    '''
    
    # Adjust days so that they are divisible by dt
    l_reg_days = int(DAYS_IN_TRADING_YEAR * l_reg)
    l_reg_days-= l_reg_days % dt
    l_roll_days = int(DAYS_IN_TRADING_YEAR * l_roll)
    l_roll_days-= l_roll_days % dt
    
    total_days = l_reg_days + l_roll_days
    
    # Number of regressions for each ticker
    n_windows = (l_roll_days // dt) + 1
    
    # Number of tickers
    n_x = X.shape[0]
    
    # Take the dates, create an empty array for windowed dates
    date_index = X.columns[-total_days:]
    date_index_windowed = np.empty(shape=(n_x*n_windows, 2), dtype='O')
    
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
    X_windows = np.empty(shape=(n_x*n_windows, l_reg_days), dtype=np.float32)
    Y_windows = np.empty(shape=(n_x*n_windows, l_reg_days), dtype=np.float32)
    
    # Take windowed slices and place them into the created empty arrays
    for n in range(n_x):
        for i in range(n_windows):
            n_i = (n*n_windows)+i
            t_i = i*dt
            t_y = t_i + l_reg_days
            
            X_windows[n_i] = X[n, t_i:t_y]
            Y_windows[n_i] = Y[n, t_i:t_y]
            date_index_windowed[n_i, 0] = date_index[t_i]
            date_index_windowed[n_i, 1] = date_index[t_y-1]
    
    # Make sure we've got the windowing dimensions right
    assert X_windows.shape == (n_x*n_windows, l_reg_days) and Y_windows.shape == (n_x*n_windows, l_reg_days)
    
    # Sanity checks
    assert all([
        X[0, -1] == X_windows[n_windows-1, -1],
        Y[0, -1] == Y_windows[n_windows-1, -1],
        X[-1, -1] == X_windows[-1, -1],
        Y[-1, -1] == Y_windows[-1, -1],
    ])
    
    # Construct ticker pair index column
    pair_index = np.array(pd.DataFrame(np.array([Y_index, X_index])).apply('_'.join, axis=0, raw=True))
    
    # Construct regression date range index column
    date_index = np.array(pd.DataFrame(np.array([date_index_windowed[:, 0], date_index_windowed[:, 1]])).apply('_'.join, axis=0, raw=True))
    
    # Lengths of indexes must match
    assert len(pair_index) == len(date_index)
    
    # Calculate and return the residuals
    res, betas, intercepts = get_residuals_many(X_windows, Y_windows)
    
    res = pd.DataFrame(res, index=pair_index)
    res.insert(0, 'dates', date_index)
    
    betas = pd.DataFrame(betas, index=pair_index)
    betas.insert(0, 'dates', date_index)
    
    intercepts = pd.DataFrame(intercepts, index=pair_index)
    
    return res, betas, intercepts


def get_adfs(residuals, adf_regression):
    # Get ADF test p-values for each row of the residuals array. No autolag (maxlag always used)
    assert residuals.dtype == np.float32
    return np.apply_along_axis(lambda x: adfuller(x, regression=adf_regression, autolag=None)[1], axis=1, arr=residuals)


def get_aggregate_adfs(residuals, betas=None, cutoff=0.1, adf_regression='c'):
    # Get ADF p-values
    adfs = get_adfs(residuals.drop(columns='dates').to_numpy(), adf_regression=adf_regression).reshape((-1, 1))
    
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


def get_last_pairs(pairs):
    # Get unique ticker pairs, in preserved order
    unique_pairs = pairs.index.unique()
    
    # Number of samples per ticker pair
    pairs_per_index = len(pairs) // len(unique_pairs)
    
    # Must be an equal number of pairs per index
    assert pairs_per_index * len(unique_pairs) == len(pairs)
    
    # Slice taking only the last regression for each ticker pair
    last_pairs = pairs.iloc[pairs_per_index-1:len(pairs):pairs_per_index].copy()
    
    # Make sure we got the slices right
    assert np.all(last_pairs.index == unique_pairs) and np.all(pairs.iloc[-1] == last_pairs.iloc[-1])
        
    return last_pairs


def get_standardized_residuals(residuals):
    # Dates aren't needed anymore, as we're using the latest regressions
    residuals = residuals.drop(columns='dates')
    
    # Get the last regression for each spread
    last_reg_pairs = get_last_pairs(residuals)
    
    # Get unique ticker pairs
    unique_pairs = last_reg_pairs.index
    
    # Convert to numpy
    last_reg_pairs = last_reg_pairs.to_numpy(dtype=np.float32)
    
    # Standardize
    means = last_reg_pairs.mean(axis=1, keepdims=True)
    stds = last_reg_pairs.std(axis=1, keepdims=True)
    last_reg_pairs = (last_reg_pairs - means) / stds
    
    # Back to a DataFrame with named indexes
    last_reg_pairs = pd.DataFrame(last_reg_pairs, index=unique_pairs)
    means = pd.DataFrame(means, index=unique_pairs)
    stds = pd.DataFrame(stds, index=unique_pairs)
        
    return (last_reg_pairs, means, stds)


def get_mean_residual_magnitude(std_residuals, dt):
    # Assume there is enough days' worth of data for averaging over dt days
    assert std_residuals.shape[1] >= dt
    
    # Select the last dt days from the right
    std_residuals = std_residuals.to_numpy(dtype=np.float32)[:, -dt:]
    
    # Take the absolute maximum for each day, over all tickers, mean over the results
    mean_magnitude = np.round(np.abs(std_residuals).max(axis=0).mean(), 2)
    
    return mean_magnitude


def get_trades_returns(prices_X, prices_Y, betas_YX, buy_X):

    # Save spread indexes for later
    pairs_indexes = betas_YX.index.copy()
    
    # Take numpy betas
    betas_YX = betas_YX.drop(columns='dates').to_numpy().copy()

    # Take numpy buy list for X
    buy_X = buy_X.values.copy()

    # Sanity checks
    assert all([
        prices_X.shape == prices_Y.shape,
        len(buy_X) == betas_YX.shape[0]
    ])

    # Prices to numpy
    prices_X = prices_X.to_numpy().copy()
    prices_Y = prices_Y.to_numpy().copy()

    # Save entering prices at t=0
    initial_prices_X = prices_X[:, [0]].copy()
    initial_prices_Y = prices_Y[:, [0]].copy()

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

    # Back to dataframe with indexes
    trade_returns = pd.DataFrame(trade_returns, index=pairs_indexes)
    
    return trade_returns


def get_trades_residuals(prices_X, prices_Y, betas_YX, intercepts_YX, means_YX, stds_YX):

    # Sanity checks
    assert all([
        np.all(betas_YX.index == intercepts_YX.index),
        np.all(means_YX.index == stds_YX.index),
        np.all(betas_YX.index == stds_YX.index)
    ])
    
    # Save indexes for later
    indexes = betas_YX.index.copy()
    
    betas_YX = betas_YX.drop(columns='dates').to_numpy().copy()
    intercepts_YX = intercepts_YX.to_numpy().copy()
    
    means_YX = means_YX.to_numpy().copy()
    stds_YX = stds_YX.to_numpy().copy()
    
    prices_X = prices_X.to_numpy().copy()
    prices_Y = prices_Y.to_numpy().copy()

    # Sanity checks
    assert all([
        prices_X.shape == prices_Y.shape,
        betas_YX.shape[0] == intercepts_YX.shape[0] == means_YX.shape[0] == stds_YX.shape[0] == prices_X.shape[0],
        betas_YX.shape[1] == intercepts_YX.shape[1] == means_YX.shape[1] == stds_YX.shape[1] == 1,
    ])
    
    # Calculate residuals from regression form: Y = bX + a
    trade_residuals = prices_Y - ((betas_YX * prices_X) + intercepts_YX)
    
    # Standardize the residuals
    trade_residuals = ((trade_residuals - means_YX) / stds_YX).round(2)

    # Back to dataframe with indexes
    trade_residuals = pd.DataFrame(trade_residuals, index=indexes)
    
    return trade_residuals