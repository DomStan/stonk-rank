import os
from datetime import datetime
from datetime import timedelta
import time

import pandas as pd
import numpy as np

import yfinance as yf



def download_stonk_prices(stonk_list, period_years, date_from=None, date_to=None, interval='1d', source='yfinance', data_dir='data', proxy=False):    
    '''Returns historical price data for the selected stonks.

    Args:
        stonk_list: List of stonk identifiers as strings, case unsensitive
        period_years: How many years of data to download until date_to, can be a floating point number
        date_from: Start date for stonk data (use instead of period_years)
        date_to: End date for stonk data
        interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        source: Where to source data from. Valid sources: yfinance
        data_dir: Folder name where to output downloaded data
        file_prefix: Prefix of CSV file containing downloaded data inside data_dir
        proxy: Whether to use a proxy connection to avoid API limits/blocks
                
    Returns: stonks (Pandas Dataframe): Pandas Dataframe containing requested ticker prices
    '''
    
    date_to = datetime.now() if date_to is None else date_to
    date_from = date_to-(timedelta(days=int(365*period_years))) if date_from is None else date_from
    
    if source.lower() == 'yfinance':
        stonks = yf.download(list(stonk_list), start=date_from, end=date_to, interval=interval, group_by='column', threads=True, rounding=True)['Adj Close']
        stonks.dropna(axis=0, how='all', inplace=True)
        stonks.sort_values(by='Date', inplace=True)
        
        stonks.index = pd.to_datetime(stonks.index).date
        stonks.index.name = 'date'

        clean_stonks = stonks.dropna(axis=1, how='all', thresh=int(len(stonks.index) * 0.99)).copy()
        clean_stonks.dropna(axis=0, how='all', thresh=int(len(clean_stonks.columns) * 0.99), inplace=True)
        
        # Forward fill ticker columns (axis=0 for columns)
        clean_stonks.fillna(axis=0, method='ffill', inplace=True)
        
        clean_stonks.dropna(axis=1, how='any', inplace=True)
        
        # Must be no NA values left
        assert clean_stonks.isna().sum().sum() == 0
    else:
        raise ValueError('Unsupported data source')
        
    def stonks_to_csv(stonks, clean):
        from_date_string = stonks.index[0]
        to_date_string = stonks.index[-1]

        filename = 'stonks_{from_date}_to_{to_date}.csv'.format(from_date=from_date_string, to_date=to_date_string)
        
        if clean:
            filename = 'clean_' + filename
            
        file_path = os.path.join(data_dir, filename)

        stonks.to_csv(path_or_buf=file_path, header=True, index=True, na_rep='NaN')
    
    stonks_to_csv(stonks, clean=False)
    stonks_to_csv(clean_stonks, clean=True)
    
    return (stonks, clean_stonks)


def get_tickers_by_industry(market_cap_min_mm, market_cap_max_mm, industries, data_dir=None, filename=None):
    '''Read the CSV file containing all tickers and their subindustries and return tickers from the selected subindustries in a list.
    
    Args:
        industries: if not given, return all tickers.
    Returns:
        tickers: list of selected ticker names
    '''
    filename = 'stonk_list.csv' if filename is None else filename
    data_dir = 'data' if data_dir is None else data_dir
    
    if not market_cap_max_mm:
        market_cap_max_mm = np.iinfo(np.int32).max
    
    path_to_csv = os.path.join(data_dir, filename)
    stonk_list = pd.read_csv(path_to_csv)
    stonk_list = stonk_list[stonk_list['market_cap'].between(market_cap_min_mm, market_cap_max_mm)]
    return stonk_list.set_index('ticker') if not industries else stonk_list[stonk_list['subindustry'].isin(industries)].set_index('ticker')


def read_stonk_data(date_from, date_to, clean=True, date_index=False, data_dir=None):
    data_dir = 'data' if data_dir is None else data_dir
    data_prefix = 'clean_stonks' if clean else 'stonks'
    
    path = os.path.join(data_dir, '{}_{}_to_{}.csv'.format(data_prefix, date_from, date_to))
    stonks = pd.read_csv(path, header=0, index_col=0).astype(np.float32)
    
    if clean:
        assert stonks.isna().sum().sum() == 0
    
    if date_index:
        return stonks
    else:
        return stonks.T
    

def get_stonk_data_by_industry(date_from, date_to, market_cap_min_mm=1000, market_cap_max_mm=None, clean=True, date_index=False, industries=None, stonk_list_filename=None, data_dir=None):
    '''Read the CSV file containing all stonk price data and return the tickers from the selected subindustries.
    
    Args:
        industries (List(string)): if not given, return all tickers
    
    Returns:
        stonks (pandas DataFrame): list of selected tickers' price data
    '''
    all_stonks = read_stonk_data(date_from, date_to, date_index=date_index, data_dir=data_dir, clean=clean)
    selected_tickers = get_tickers_by_industry(market_cap_min_mm=market_cap_min_mm, market_cap_max_mm=market_cap_max_mm, industries=industries, data_dir=data_dir, filename=stonk_list_filename)
    return all_stonks[all_stonks.index.isin(selected_tickers.index)]

    
def ingest_trade_pipeline_outputs(data_dir='data/trades'):
    data_list = []
    for file in os.listdir(data_dir):
            if file.endswith('csv'):
                data_list.append(pd.read_csv(os.path.join(data_dir, file)))
    df = pd.concat(data_list, ignore_index=True)
    return df


def measure_time(func):
    t1 = time.time()
    ret = func()
    t2 = time.time()
    print("Done after: " + str(int(t2-t1)) + 's')
    return ret


def separate_pair_index(indexes):
    indexes = pd.Series(indexes)
    splits = indexes.apply(lambda x: pd.Series(x.split('_')))
    y = splits[0].values
    x = splits[1].values
    return {'y':y, 'x':x}


def preprocess_stock_list(write_csv=True, raw_data_path='data/raw_stonk_list.xls', output_path='data/stonk_list.csv'):
    '''Parses a raw excel file from CapitalIQ containing ticker names and their subindustries, validates
    unusual ticker names with Yahoo Finance, saving the processed data in CSV format.

    Args:
        raw_data_path: Path to the raw excel file.
        output_path: Path where to save the parsed data.
                
    Returns:
        Processed ticker names as a CSV file containing columns: ticker, market_cap_mm, subindustry
    '''
    
    df = pd.read_excel(io=raw_data_path)

    # Drop NA rows
    df.dropna(axis=0, inplace=True)

    # Reset index and drop the first row
    df.reset_index(inplace=True, drop=True)
    df.columns = df.iloc[0]
    df.drop(index=0, axis=0, inplace=True)

    # Drop unwanted columns
    drop_columns = [
        'Security Name',
        'Company Name',
        'Trading Status',
        'Most Recent Trade Price ($USD, Historical rate)',
        'Equity Security Type',
        'Exchange Country/Region',
        'Exchanges'
    ]
    df.drop(columns=drop_columns, inplace=True)

    # Rename remaining columns
    df.columns = ['ticker', 'subindustry', 'market_cap']
    
    # Remove the '(Primary)' tag from subindustries
    df['subindustry'] = df['subindustry'].str.replace(r' \(Primary\)', '')
    
    # Remove everything until (and including) the semicolon for tickers
    df['ticker'] = df['ticker'].str.replace(r'(.*:)', '')
    
    # Take all remaining tickers that have a dot
    dotted = df[df['ticker'].str.fullmatch(r'[A-Z]*\.[A-Z]')]
    
    # Replace the dots with dashes
    dashed = dotted.copy()
    dashed['ticker'] = dashed['ticker'].str.replace(r'\.', '-')
    
    # Remove the dots
    undotted = dotted.copy()
    undotted['ticker'] = undotted['ticker'].str.replace(r'\.', '')

    # Combine all variantas together
    all_variants = pd.concat([dotted, dashed, undotted])
    
    # Run all of these through Yahoo finance, get last day's price
    stonks = yf.download(list(all_variants['ticker'].astype('string').values), period='1m', interval='1d', group_by='column')
    
    # Drop all NA tickers (that failed to download)
    valid_tickers = stonks['Adj Close'].iloc[-1].dropna(axis=0, how='all').to_frame().reset_index()
    
    # Rename columns
    valid_tickers.columns = ['ticker', 'price']
    
    # Add subindustries to the remaining valid tickers
    valid_tickers = valid_tickers.join(all_variants.set_index('ticker'), on='ticker')
    
    # Drop the price column
    valid_tickers.drop(columns=valid_tickers.columns[[1]], inplace=True)
    
    # Remove all tickers that have a dot from main dataframe
    df = df[~df['ticker'].str.fullmatch(r'[A-Z]*\.[A-Z]')]
    
    # Add the validated tickers back
    df = pd.concat([df, valid_tickers], axis=0, ignore_index=True)
    
    # Make the subindustry strings more code friendly
    df['subindustry'] = df['subindustry'].str.replace(' ', '_')
    df['subindustry'] = df['subindustry'].str.lower()
    df['subindustry'] = df['subindustry'].str.replace(',', '')
    
    if write_csv:
        df.to_csv(path_or_buf=output_path, header=True, index=False)
    
    return df