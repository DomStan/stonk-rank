import os
from datetime import datetime
from datetime import timedelta
import time

from typing import Iterable
from typing import Dict
from typing import Tuple
from numpy.typing import ArrayLike

import pandas as pd
import numpy as np

import yfinance as yf


def download_stonk_prices(
    tickers: Iterable[str],
    period_years: float,
    date_from: datetime = None,
    date_to: datetime = None,
    interval: str = "1d",
    source: str = "yfinance",
    data_dir: str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads historical price data for given tickers from a given source.

    Args:
        tickers - iterable of stock identifiers as strings
        period_years - how many years of data to download until date_to, can be a float; roughly converted into days
        date_from - exact start date for downloading data (used instead of period_years)
        date_to - end date for downloading data, set to today if not given
        interval - valid intervals of price frequency: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        source - where to source data from; valid sources: yfinance
        data_dir - directory name where to save the downloaded data
                
    Returns:
        Tuple containing the raw and cleaned versions of downloaded data (for debugging purposes).
    """

    date_to = datetime.now() if date_to is None else date_to
    date_from = (
        date_to - (timedelta(days=int(365 * period_years)))
        if date_from is None
        else date_from
    )

    if source.lower() == "yfinance":
        stonks = yf.download(
            list(tickers),
            start=date_from,
            end=date_to,
            interval=interval,
            group_by="column",
            threads=True,
            rounding=True,
        )["Adj Close"]
        stonks.dropna(axis=0, how="all", inplace=True)
        stonks.sort_values(by="Date", inplace=True)

        stonks.index = pd.to_datetime(stonks.index).date
        stonks.index.name = "date"

        clean_stonks = stonks.dropna(
            axis=1, how="all", thresh=int(len(stonks.index) * 0.99)
        ).copy()
        clean_stonks.dropna(
            axis=0,
            how="all",
            thresh=int(len(clean_stonks.columns) * 0.99),
            inplace=True,
        )

        # Forward fill ticker columns (axis=0 for columns)
        clean_stonks.fillna(axis=0, method="ffill", inplace=True)

        clean_stonks.dropna(axis=1, how="any", inplace=True)

        # Must be no NA values left
        assert clean_stonks.isna().sum().sum() == 0
    else:
        raise ValueError("Unsupported data source")

    def _stonks_to_csv(df, is_cleaned):
        from_date_string = df.index[0]
        to_date_string = df.index[-1]

        filename = "stonks_{from_date}_to_{to_date}.csv".format(
            from_date=from_date_string, to_date=to_date_string
        )

        if is_cleaned:
            filename = "clean_" + filename

        file_path = os.path.join(data_dir, filename)

        df.to_csv(path_or_buf=file_path, header=True, index=True, na_rep="NaN")

    _stonks_to_csv(stonks, is_cleaned=False)
    _stonks_to_csv(clean_stonks, is_cleaned=True)

    return (stonks, clean_stonks)


def get_ticker_names(
    market_cap_min_mm: float,
    market_cap_max_mm: float,
    filter_industries: Iterable[str] = None,
    data_dir: str = None,
    filename: str = None,
) -> pd.DataFrame:
    """Read the CSV file containing all tickers and their subindustries and return tickers from the selected (or all) subindustries in a dataframe.
    
    Args:
        industries: if not given, return all tickers.
    Returns:
        Pandas dataframe of tickers, optionally filtered by given industries.
    """
    filename = "tickers.csv" if filename is None else filename
    data_dir = "data" if data_dir is None else data_dir

    if not market_cap_max_mm:
        market_cap_max_mm = np.iinfo(np.int32).max

    path_to_csv = os.path.join(data_dir, filename)
    tickers = pd.read_csv(path_to_csv)
    tickers = tickers[
        tickers["market_cap"].between(market_cap_min_mm, market_cap_max_mm)
    ]
    return (
        tickers.set_index("ticker")
        if not filter_industries
        else tickers[tickers["subindustry"].isin(filter_industries)].set_index("ticker")
    )


def _read_stonk_data(
    date_from: str, date_to: str, clean: bool = True, data_dir: str = None
) -> pd.DataFrame:
    data_dir = "data" if data_dir is None else data_dir
    data_prefix = "clean_stonks" if clean else "stonks"

    path = os.path.join(
        data_dir, "{}_{}_to_{}.csv".format(data_prefix, date_from, date_to)
    )
    stonks = pd.read_csv(path, header=0, index_col=0).astype(np.float32)

    if clean:
        assert stonks.isna().sum().sum() == 0

    return stonks.T


def get_stonk_data(
    date_from: str,
    date_to: str,
    market_cap_min_mm: float = 1000,
    market_cap_max_mm: float = None,
    clean: bool = True,
    filter_industries: Iterable[str] = None,
    ticker_list_filename: str = None,
    data_dir: str = None,
) -> pd.DataFrame:
    """Read the CSV file containing all stonk price data and return the tickers from the selected subindustries.
    
    Args:
        industries - if not given, return all tickers
    
    Returns:
        stonks - list of selected tickers' price data
    """
    all_stonks = _read_stonk_data(date_from, date_to, data_dir=data_dir, clean=clean,)
    selected_tickers = get_ticker_names(
        market_cap_min_mm=market_cap_min_mm,
        market_cap_max_mm=market_cap_max_mm,
        filter_industries=filter_industries,
        data_dir=data_dir,
        filename=ticker_list_filename,
    )
    return all_stonks[all_stonks.index.isin(selected_tickers.index)]


def ingest_trade_pipeline_outputs(data_dir="data/trades"):
    data_list = []
    for file in os.listdir(data_dir):
        if file.endswith("csv"):
            data_list.append(pd.read_csv(os.path.join(data_dir, file)))
    df = pd.concat(data_list, ignore_index=True)
    return df


def measure_time(func):
    t1 = time.time()
    ret = func()
    t2 = time.time()
    print("Done after: " + str(int(t2 - t1)) + "s")
    return ret


def separate_pair_index(indexes: ArrayLike) -> Dict[str, np.ndarray]:
    indexes = pd.Series(indexes)
    splits = indexes.apply(lambda x: pd.Series(x.split("_")))
    y = splits[0].values
    x = splits[1].values
    return {"y": y, "x": x}


def preprocess_stock_list(
    write_csv: bool = True,
    raw_data_path: str = "data/raw_stonk_list.xls",
    output_path: str = "data/tickers.csv",
) -> pd.DataFrame:
    """Parses a raw excel file from CapitalIQ containing ticker names and their subindustries, validates
    unusual ticker names with Yahoo Finance, saving the processed data in CSV format.

    Args:
        raw_data_path: Path to the raw excel file.
        output_path: Path where to save the parsed data.
                
    Returns:
        Processed ticker names as a CSV file containing columns: ticker, market_cap_mm, subindustry
    """

    df = pd.read_excel(io=raw_data_path)

    # Drop NA rows
    df.dropna(axis=0, inplace=True)

    # Reset index and drop the first row
    df.reset_index(inplace=True, drop=True)
    df.columns = df.iloc[0]
    df.drop(index=0, axis=0, inplace=True)

    # Drop unwanted columns
    drop_columns = [
        "Security Name",
        "Company Name",
        "Trading Status",
        "Most Recent Trade Price ($USD, Historical rate)",
        "Equity Security Type",
        "Exchange Country/Region",
        "Exchanges",
    ]
    df.drop(columns=drop_columns, inplace=True)

    # Rename remaining columns
    df.columns = ["ticker", "subindustry", "market_cap"]

    # Remove the '(Primary)' tag from subindustries
    df["subindustry"] = df["subindustry"].str.replace(r" \(Primary\)", "")

    # Remove everything until (and including) the semicolon for tickers
    df["ticker"] = df["ticker"].str.replace(r"(.*:)", "")

    # Take all remaining tickers that have a dot
    dotted = df[df["ticker"].str.fullmatch(r"[A-Z]*\.[A-Z]")]

    # Replace the dots with dashes
    dashed = dotted.copy()
    dashed["ticker"] = dashed["ticker"].str.replace(r"\.", "-")

    # Remove the dots
    undotted = dotted.copy()
    undotted["ticker"] = undotted["ticker"].str.replace(r"\.", "")

    # Combine all variantas together
    all_variants = pd.concat([dotted, dashed, undotted])

    # Run all of these through Yahoo finance, get last day's price
    stonks = yf.download(
        list(all_variants["ticker"].astype("string").values),
        period="1m",
        interval="1d",
        group_by="column",
    )

    # Drop all NA tickers (that failed to download)
    valid_tickers = (
        stonks["Adj Close"].iloc[-1].dropna(axis=0, how="all").to_frame().reset_index()
    )

    # Rename columns
    valid_tickers.columns = ["ticker", "price"]

    # Add subindustries to the remaining valid tickers
    valid_tickers = valid_tickers.join(all_variants.set_index("ticker"), on="ticker")

    # Drop the price column
    valid_tickers.drop(columns=valid_tickers.columns[[1]], inplace=True)

    # Remove all tickers that have a dot from main dataframe
    df = df[~df["ticker"].str.fullmatch(r"[A-Z]*\.[A-Z]")]

    # Add the validated tickers back
    df = pd.concat([df, valid_tickers], axis=0, ignore_index=True)

    # Make the subindustry strings more code friendly
    df["subindustry"] = df["subindustry"].str.replace(" ", "_")
    df["subindustry"] = df["subindustry"].str.lower()
    df["subindustry"] = df["subindustry"].str.replace(",", "")

    if write_csv:
        df.to_csv(path_or_buf=output_path, header=True, index=False)

    return df


def build_dataset_from_live_data_by_industry(
    std_residuals: np.ndarray,
    adfs: np.ndarray,
    subindustry: str,
    mean_max_residual: float,
) -> pd.DataFrame:
    """Builds an inference-ready (before pre-processing) dataset with predefined feature names from incoming live data for one subindustry.
    
    Args:
        std_residuals - numpy array containing N rows of standardized residuals for each trade
        adfs - numpy array containing N rows of ADF test pass rates for each trade
        subindustry - string name of the subindustry
        mean_max_residual - averaged max standardized residual value for the current subindustry
    
    Returns:
        pandas dataframe with a named column for each feature which should all be of equal length
    
    """
    output_length = len(std_residuals)
    dataset = {}
    dataset["adf_pass_rate"] = adfs.round(3)
    dataset["last_residual"] = std_residuals[:, -1].round(3)
    dataset["residual_mean_max"] = np.full(output_length, mean_max_residual)
    dataset["subindustry"] = np.full(output_length, subindustry)

    # All columns must have equal length
    assert len(set([len(x) for x in dataset.values()])) == 1
    return pd.DataFrame(dataset)
