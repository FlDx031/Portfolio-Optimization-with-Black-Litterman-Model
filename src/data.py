"""src/data.py

Data utilities for the Black-Litterman project.

Provides:
- get_prices: robust yfinance wrapper handling single/multi tickers and MultiIndex
- get_sp500_tickers: helper to fetch S&P500 symbols from Wikipedia
- get_market_caps: fetch market caps (best-effort, ignores missing)
- get_cleaned_data: assemble macro factors from FRED and monthly asset returns, return X (lagged features) and y (target returns)

All functions are documented and defensive; intended to be imported by BacktestBL.py.
"""
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import pandas_datareader.data as web
from datetime import datetime


def get_sp500_tickers() -> List[str]:
    """Return list of S&P500 tickers from Wikipedia.

    Replaces dots in symbols (BRK.B -> BRK-B) to match yfinance.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "python-requests/2.x"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    table = pd.read_html(StringIO(r.text))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def get_prices(tickers, start_date: str = "2015-01-01") -> pd.DataFrame:
    """Download Close prices for tickers using yfinance.

    Handles single ticker (Series) and multiple tickers (DataFrame) robustly.
    Returns DataFrame of close prices with DatetimeIndex.
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if raw.empty:
        raise ValueError("No price data downloaded — check tickers and network")
    # yfinance returns a DataFrame with columns as MultiIndex when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Close' in raw.columns.get_level_values(0):
            close = raw['Close']
        elif ('Adj Close' in raw.columns.get_level_values(0)):
            close = raw['Adj Close']
        else:
            # fallback to last level
            close = raw.iloc[:, raw.columns.get_level_values(0) == raw.columns.get_level_values(0)[-1]]
    else:
        # Single ticker -> raw['Close'] returns Series; convert to DataFrame
        if 'Close' in raw.columns:
            close = raw['Close']
        elif 'Adj Close' in raw.columns:
            close = raw['Adj Close']
        else:
            close = raw.iloc[:, -1]
    close = pd.DataFrame(close)
    # If columns are tickers in MultiIndex, ensure names are tickers
    if close.shape[1] == 1 and isinstance(tickers, (list, tuple)) and len(tickers) == 1:
        close.columns = [tickers[0]]
    # For multi-ticker downloads, columns should be tickers already
    return close


def get_market_caps(tickers: List[str]) -> Dict[str, float]:
    """Return a dict ticker->marketCap (best-effort).

    Some tickers may return None; those are skipped.
    """
    caps = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info
            cap = info.get('marketCap')
            if cap is None:
                # Try fast info fallback
                cap = tk.fast_info.get('market_cap', None) if hasattr(tk, 'fast_info') else None
            if cap is not None:
                caps[t] = float(cap)
        except Exception:
            # best-effort: skip problematic tickers
            continue
    return caps


def get_cleaned_data(ticker: str, start_date: str = "2015-01-01") -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch macro factors (FRED) and monthly asset returns for a single ticker.

    Returns:
      X: DataFrame of features (monthly macro factors lagged by 1 month)
      y: Series of monthly asset returns aligned with X.index

    Note: Normalisation must be done after train/test split to avoid look-ahead bias.
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    macro_setup = {
        'GS10': '10Y_Yield',    # Monthly
        'CPIAUCSL': 'CPI',      # Monthly
        'VIXCLS': 'VIX',        # Daily
        'INDPRO': 'IP'          # Monthly
    }

    macro_list = []
    for code, name in macro_setup.items():
        s = web.DataReader(code, 'fred', start_date, end_date)
        s.columns = [name]
        macro_list.append(s)
    macro_df = pd.concat(macro_list, axis=1).ffill()

    # Resample to month end and stationarize
    macro_monthly_raw = macro_df.resample('ME').last()
    macro_st = pd.DataFrame(index=macro_monthly_raw.index)
    macro_st['10Y_Diff'] = macro_monthly_raw['10Y_Yield'].diff()
    macro_st['VIX_Diff'] = macro_monthly_raw['VIX'].diff()
    macro_st['CPI_Ret'] = macro_monthly_raw['CPI'].pct_change()
    macro_st['IP_Ret'] = macro_monthly_raw['IP'].pct_change()
    macro_monthly = macro_st.dropna()

    asset_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(asset_raw.columns, pd.MultiIndex):
        # case where yfinance returns multiindex: ('Close', 'TICKER')
        try:
            prices = asset_raw['Close', ticker]
        except Exception:
            prices = asset_raw['Close']
    else:
        prices = asset_raw['Close'] if 'Close' in asset_raw.columns else asset_raw.iloc[:, -1]

    # timezone
    if hasattr(prices.index, 'tz') and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    # monthly compounded returns
    asset_monthly_ret = prices.pct_change().resample('ME').apply(lambda x: (1 + x).prod() - 1)
    asset_monthly_ret.name = 'Asset_Return'

    final_df = pd.concat([asset_monthly_ret, macro_monthly], axis=1).dropna()

    # X is lagged by 1 month to avoid look-ahead
    X = final_df.drop(columns=['Asset_Return']).shift(1).dropna()
    y = final_df['Asset_Return'].loc[X.index]

    return X, y
