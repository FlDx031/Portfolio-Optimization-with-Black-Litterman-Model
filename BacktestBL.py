"""BacktestBL.py

Manual monthly walk-forward backtest for Black-Litterman strategies.

This script uses the helper modules in src/ to:
- download prices and macro data
- produce views (momentum or multi_linear_regression)
- compute Black-Litterman weights
- apply monthly rebalancing walk-forward and compute performance metrics vs SPY benchmark

Usage (from repo root):
    python BacktestBL.py

The script is documented step-by-step and prints key results. It is intentionally
simple and pedagogical so you can read the logic and adapt parameters.
"""
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict

from src.data import get_prices, get_cleaned_data, get_market_caps
from src.bl import black_litterman_opt, markowitz_opt, compute_sigma

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------
# Configuration (editable)
# --------------------------
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMZN']
MARKET = 'SPY'
START_DATE = '2018-01-01'
RISK_FREE_RATE = 0.02
TRAIN_WINDOW_MONTHS = 36  # 36 months training window
REBALANCE_FREQ = 'M'  # monthly
METHOD = 'multi_linear_regression'  # or 'momentum'
RIDGE_ALPHA = 1.0


# --------------------------
# Helpers
# --------------------------

def monthly_end_dates(prices: pd.DataFrame) -> List[pd.Timestamp]:
    """Return list of month-end dates present in prices index (sorted)."""
    return list(prices.resample('ME').last().index)


def compute_momentum_Q(prices: pd.DataFrame, k: float = 0.05) -> (np.ndarray, np.ndarray):
    """Compute P,Q,Omega from price momentum (simple implementation).

    Returns P (NxN identity), Q (N,1) monthly returns forecast, Omega diagonal.
    """
    # compute 12-month momentum excluding last month
    monthly = prices.pct_change().resample('ME').apply(lambda x: (1 + x).prod() - 1)
    # use last available 12-month window
    mom = (prices.resample('ME').last().iloc[-2] / prices.resample('ME').last().iloc[-13]) - 1
    z = (mom - mom.mean()) / mom.std()
    Q = z.values.reshape(-1, 1) * k
    P = np.eye(len(prices.columns))
    # Omega: small diagonal scaled to sample cov
    sigma = compute_sigma(prices)
    Omega = 0.05 * (P @ sigma.values @ P.T)
    return P, Q, Omega


def multi_linear_regression_views(tickers: List[str], as_of_date: pd.Timestamp, ridge_alpha: float = 1.0):
    """Fit Ridge regression per asset using data up to as_of_date and predict next-month return.

    This function uses get_cleaned_data() to fetch X,y per ticker and fits a time-series split
    using the last TRAIN_WINDOW_MONTHS months for training. Returns P,Q,Omega and diagnostics.
    """
    q_views = []
    omega_vars = []
    diags = {}

    for t in tickers:
        X, y = get_cleaned_data(t, start_date=START_DATE)
        # only keep data up to as_of_date
        X = X[X.index <= as_of_date]
        y = y.loc[X.index]
        if X.shape[0] < 12:
            # not enough data -> predict 0
            q_views.append(0.0)
            omega_vars.append(1.0)
            diags[t] = {'r2': None, 'mse': None, 'pred': 0.0}
            continue
        # use last TRAIN_WINDOW_MONTHS for training
        train_start = max(0, X.shape[0] - TRAIN_WINDOW_MONTHS)
        X_train = X.iloc[train_start:]
        y_train = y.iloc[train_start:]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train_s, y_train)
        # prediction for next period: use latest macro row
        latest_macro = X.iloc[[-1]]
        latest_scaled = scaler.transform(latest_macro)
        pred = float(model.predict(latest_scaled)[0])
        # estimate var from train residuals
        y_pred_train = model.predict(X_train_s)
        mse = float(mean_squared_error(y_train, y_pred_train))
        r2 = float(r2_score(y_train, y_pred_train))
        q_views.append(pred)
        omega_vars.append(mse)
        diags[t] = {'r2': r2, 'mse': mse, 'pred': pred}

    P = np.eye(len(tickers))
    Q = np.array(q_views).reshape(-1, 1)
    Omega = np.diag(omega_vars)
    return P, Q, Omega, diags


def apply_weights_to_next_period(weights: Dict[str, float], prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """Apply weights (dict ticker->weight) to compute portfolio return from start_date (exclusive) to end_date (inclusive).

    We compute gross returns from price series. If some tickers missing, they are treated as 0 weight.
    Returns portfolio cumulative return over the period.
    """
    # align price series
    monthly = prices.resample('ME').last()
    try:
        start_price = monthly.loc[start_date]
        end_price = monthly.loc[end_date]
    except KeyError:
        # if exact dates missing, pick closest earlier
        monthly_idx = monthly.index
        start_idx = monthly_idx[monthly_idx <= start_date][-1]
        end_idx = monthly_idx[monthly_idx <= end_date][-1]
        start_price = monthly.loc[start_idx]
        end_price = monthly.loc[end_idx]

    rets = (end_price / start_price) - 1.0
    # build weight vector aligned with rets
    w = np.array([weights.get(t, 0.0) for t in rets.index], dtype=float)
    port_ret = float((w * rets.values).sum())
    return port_ret


def compute_performance(series_returns: List[float], rf: float = RISK_FREE_RATE) -> Dict[str, float]:
    """Compute basic performance metrics given list of periodic returns (monthly).

    Returns cumulative return, annualized return, annualized vol, Sharpe, max drawdown.
    """
    arr = np.array(series_returns, dtype=float)
    cum = np.prod(1 + arr) - 1
    periods = len(arr)
    if periods == 0:
        return {}
    ann_ret = (1 + cum) ** (12 / periods) - 1
    ann_vol = arr.std(ddof=1) * np.sqrt(12)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    # drawdown
    cum_ts = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum_ts)
    dd = (cum_ts - peak) / peak
    max_dd = float(dd.min())
    return {'cumulative_return': float(cum), 'annualized_return': float(ann_ret), 'annualized_vol': float(ann_vol), 'sharpe': float(sharpe), 'max_drawdown': max_dd}


# --------------------------
# Walk-forward backtest
# --------------------------

def walk_forward_backtest(tickers: List[str], market: str, start_date: str, method: str = 'multi_linear_regression'):
    prices = get_prices(tickers, start_date)
    market_prices = get_prices(market, start_date)
    all_prices = prices.join(market_prices, how='inner')

    # choose monthly rebalancing dates (end of month) after having at least TRAIN_WINDOW_MONTHS
    medates = monthly_end_dates(all_prices)
    medates = [d for d in medates if d >= pd.Timestamp(start_date)]

    # warm-up: start after we have train window + 1 month
    warm_idx = TRAIN_WINDOW_MONTHS
    medates = medates[warm_idx:]

    pf_returns = []
    bench_returns = []
    weights_history = []

    market_caps = get_market_caps(tickers)

    for i in range(len(medates) - 1):
        as_of = medates[i]
        next_month = medates[i + 1]
        print(f"Rebalance date: {as_of.date()} -> apply to period until {next_month.date()}")

        # prices up to as_of
        hist_prices = all_prices.loc[:as_of]
        # compute views
        if method == 'momentum':
            P, Q, Omega = compute_momentum_Q(hist_prices[tickers])
            diags = {}
        else:
            P, Q, Omega, diags = multi_linear_regression_views(tickers, as_of)

        # compute weights
        try:
            weights = black_litterman_opt(tickers, hist_prices[tickers], P, Q, Omega, market_caps, risk_free_rate=RISK_FREE_RATE)
        except Exception as e:
            print('BL failed:', e)
            weights = markowitz_opt(tickers, hist_prices[tickers], risk_free_rate=RISK_FREE_RATE)

        weights_history.append((as_of, weights))

        # apply weights to next month
        pf_ret = apply_weights_to_next_period(weights, all_prices[tickers], as_of, next_month)
        bench_ret = apply_weights_to_next_period({market: 1.0}, all_prices[[market]], as_of, next_month) if market in all_prices.columns else 0.0
        pf_returns.append(pf_ret)
        bench_returns.append(bench_ret)

    pf_metrics = compute_performance(pf_returns, rf=RISK_FREE_RATE)
    bench_metrics = compute_performance(bench_returns, rf=RISK_FREE_RATE)

    print('\n=== Portfolio performance ===')
    for k, v in pf_metrics.items():
        print(f"{k}: {v}")
    print('\n=== Benchmark performance (SPY) ===')
    for k, v in bench_metrics.items():
        print(f"{k}: {v}")

    return {'pf_returns': pf_returns, 'bench_returns': bench_returns, 'weights_history': weights_history, 'pf_metrics': pf_metrics, 'bench_metrics': bench_metrics}


if __name__ == '__main__':
    print('Running a quick monthly walk-forward backtest (smoke run).')
    res = walk_forward_backtest(TICKERS, MARKET, START_DATE, method=METHOD)
    print('\nDone. Inspect res dictionary for details.')
