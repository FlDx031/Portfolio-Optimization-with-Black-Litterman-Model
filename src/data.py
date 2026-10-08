"""Price downloads and macro information actually available on each signal date."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf


def validate_prices(prices, start=None, end=None):
    """Require one complete price row for each NYSE session in the date range."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a non-empty DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must have a DatetimeIndex.")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing.")
    if prices.columns.has_duplicates or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Duplicate tickers or missing/non-finite prices. Check the download.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    if prices.index.tz is not None or not prices.index.equals(prices.index.normalize()):
        raise ValueError("Price dates must be timezone-naive session dates.")
    first = pd.Timestamp(start) if start is not None else prices.index[0]
    last = pd.Timestamp(end) if end is not None else prices.index[-1]
    expected = mcal.get_calendar("NYSE").valid_days(first, last).tz_localize(None)
    missing = expected.difference(prices.index)
    unexpected = prices.index.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f"Prices do not match NYSE sessions: missing {missing[:5].strftime('%Y-%m-%d').tolist()}, "
                         f"unexpected {unexpected[:5].strftime('%Y-%m-%d').tolist()}.")


def load_prices(tickers, start, end, cache_dir="data/cache"):
    """Adjusted daily closes, inclusive fixed dates, cached for repeat runs.

    Cached prices are an as-downloaded snapshot, not a point-in-time database.
    Delete the corresponding cache file explicitly to refresh it.
    """
    tickers = list(tickers)
    if len(set(tickers)) != len(tickers) or not tickers:
        raise ValueError("Choose a non-empty list of distinct tickers.")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end or end >= pd.Timestamp.today().normalize():
        raise ValueError("Use fixed past dates with start < end < today.")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    filename = cache / f"prices_{'_'.join(tickers)}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if filename.exists():
        prices = pd.read_csv(filename, index_col=0, parse_dates=True, float_precision="round_trip")
    else:
        raw = yf.download(tickers, start=start, end=end + pd.Timedelta(days=1),
                          auto_adjust=True, progress=False, threads=False)
        if raw.empty or "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo returned no closing prices. Check connectivity and tickers.")
        prices = raw["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
        prices = prices.reindex(columns=tickers).sort_index()
        validate_prices(prices, start, end)
        prices.to_csv(filename, index_label="date")
    prices = prices.reindex(columns=tickers)
    validate_prices(prices, start, end)
    return prices


def monthly_schedule(prices):
    """Signal at last session of month t; trade at first close of t+1.

    Only months followed by another observed month can become signal months.
    The final execution date is used only to close the last holding period.
    """
    validate_prices(prices)
    groups = list(prices.groupby(prices.index.to_period("M")))
    rows = []
    for (period, month), (next_period, next_month) in zip(groups[:-1], groups[1:]):
        if next_period.ordinal != period.ordinal + 1:
            raise ValueError("A full calendar month is missing from prices.")
        rows.append((month.index[-1], next_month.index[0]))
    return pd.DataFrame(rows, columns=["signal", "execution"])


def fred_snapshot(series_id, as_of, api_key, cache_dir="data/cache"):
    """ALFRED observations as known on one historical date; never cache the key."""
    date = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    cache = Path(cache_dir) / "alfred"
    cache.mkdir(parents=True, exist_ok=True)
    filename = cache / f"{series_id}_{date}.csv"
    if filename.exists():
        return pd.read_csv(filename, index_col=0, parse_dates=True, float_precision="round_trip")["value"]
    if not api_key:
        raise ValueError("FRED_API_KEY is required for uncached historical macro snapshots.")
    params = dict(series_id=series_id, api_key=api_key, file_type="json",
                  realtime_start=date, realtime_end=date,
                  observation_start=(pd.Timestamp(as_of) - pd.DateOffset(months=6)).strftime("%Y-%m-%d"),
                  observation_end=date)
    time.sleep(0.6)  # Keep sequential requests below FRED's rate limit.
    try:
        response = requests.get("https://api.stlouisfed.org/fred/series/observations",
                                params=params, timeout=30)
    except requests.RequestException:
        raise RuntimeError(f"FRED connection failed for {series_id} at {date}.") from None
    if response.status_code != 200:
        # Do not expose the request URL: it contains the API key.
        raise RuntimeError(f"FRED HTTP {response.status_code} for {series_id} at {date}.")
    observations = pd.DataFrame(response.json().get("observations", []))
    if observations.empty:
        raise ValueError(f"No vintage observations for {series_id} at {date}.")
    values = pd.Series(pd.to_numeric(observations["value"], errors="coerce").to_numpy(),
                       index=pd.to_datetime(observations["date"]), name="value").dropna()
    if values.empty:
        raise ValueError(f"All observations are missing for {series_id} at {date}.")
    values.to_csv(filename, index_label="date")
    return values


def load_macro_features(signal_dates, api_key, cache_dir="data/cache"):
    """Freeze features at each decision; never replace them with later revisions.

    Use information available the day BEFORE the signal to avoid intraday release
    ambiguity. Each factor uses its latest two completed observation months then
    available. Publication lags differ across series.
    """
    rows = []
    for i, signal in enumerate(pd.DatetimeIndex(signal_dates)):
        as_of = signal - pd.Timedelta(days=1)
        features = {}
        for code, name, change in [("GS10", "yield_change", "diff"),
                                    ("VIXCLS", "vix_change", "diff"),
                                    ("CPIAUCSL", "inflation", "return"),
                                    ("INDPRO", "production_growth", "return")]:
            vintage = fred_snapshot(code, as_of, api_key, cache_dir)
            monthly = vintage.groupby(vintage.index.to_period("M")).last()
            monthly = monthly[monthly.index < signal.to_period("M")]
            if len(monthly) < 2 or monthly.index[-1].ordinal < signal.to_period("M").ordinal - 3:
                raise ValueError(f"Insufficient or stale {code} vintage at {signal.date()}.")
            if monthly.index[-1].ordinal - monthly.index[-2].ordinal != 1:
                raise ValueError(f"Non-consecutive observation months for {code} at {signal.date()}.")
            previous, latest = monthly.iloc[-2:]
            features[name] = latest - previous if change == "diff" else latest / previous - 1
        rows.append(features)
        if (i + 1) % 12 == 0:
            print(f"Macro snapshots: {i + 1}/{len(signal_dates)} dates")
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(signal_dates))
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Macro features must be finite and non-empty.")
    return frame
