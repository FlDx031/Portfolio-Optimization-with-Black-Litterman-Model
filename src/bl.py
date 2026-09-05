"""src/bl.py

Black-Litterman helpers and simple Markowitz wrapper.

Functions:
- compute_sigma: wrapper for PyPortfolioOpt risk_models.risk_matrix with safe defaults
- market_prior_from_caps: wrapper to compute market implied prior returns
- black_litterman_opt: produce clean_weights given P,Q,Omega
- markowitz_opt: simple mean-historical-return + sample cov max Sharpe

The module keeps interfaces simple and documented for pedagogy.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pypfopt import BlackLittermanModel, EfficientFrontier, expected_returns, risk_models, black_litterman


def compute_sigma(prices: pd.DataFrame, method: str = 'ledoit_wolf_constant_correlation', frequency: int = 12) -> pd.DataFrame:
    """Compute covariance matrix for given prices using PyPortfolioOpt risk_models.

    prices: DataFrame of asset prices (columns are tickers). frequency: number of periods per year (12 for monthly).
    method: one of the methods supported by risk_models.risk_matrix.
    """
    method_map = {
        'sample_cov': 'sample_cov',
        'semicovariance': 'semicovariance',
        'exp_cov': 'exp_cov',
        'ledoit_wolf': 'ledoit_wolf',
        'ledoit_wolf_constant_variance': 'ledoit_wolf_constant_variance',
        'ledoit_wolf_single_factor': 'ledoit_wolf_single_factor',
        'ledoit_wolf_constant_correlation': 'ledoit_wolf_constant_correlation',
        'oracle_approximating': 'oracle_approximating'
    }
    m = method_map.get(method, 'ledoit_wolf_constant_correlation')
    sigma = risk_models.risk_matrix(prices, method=m, frequency=frequency)
    return sigma


def market_prior_from_caps(market_caps: Dict[str, float], sigma: pd.DataFrame, market_prices: Optional[pd.Series] = None, risk_free_rate: float = 0.02, frequency: int = 12) -> pd.Series:
    """Compute market-implied prior returns using PyPortfolioOpt black_litterman.market_implied_prior_returns.

    market_caps: dict ticker->market cap. sigma: covariance matrix with matching order to tickers.
    market_prices: optional Series of market index prices (used to compute implied risk aversion delta).
    If market_prices is not provided, a sensible default delta is used.

    Returns a pd.Series of implied returns aligned with sigma.index.
    """
    tickers = list(sigma.index)
    caps = np.array([market_caps.get(t, 0.0) for t in tickers], dtype=float)
    total = caps.sum()
    if total == 0:
        raise ValueError('No market caps available to compute market prior')
    # Delta: try to compute from market_prices if provided, else use a default
    if market_prices is not None:
        try:
            delta = black_litterman.market_implied_risk_aversion(market_prices, frequency=frequency, risk_free_rate=risk_free_rate)
        except Exception:
            delta = 2.5
    else:
        delta = 2.5

    cap_dict = {t: float(market_caps.get(t, 0.0)) for t in tickers}
    prior = black_litterman.market_implied_prior_returns(cap_dict, delta, sigma, risk_free_rate=risk_free_rate)
    prior_s = pd.Series(prior, index=tickers)
    return prior_s


def black_litterman_opt(tickers: List[str], prices: pd.DataFrame, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray, market_caps: Dict[str, float], market_prices: Optional[pd.Series] = None, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Return cleaned long-only weights from Black-Litterman model.

    tickers: list of tickers (order must match prices columns)
    prices: DataFrame of prices
    P, Q, Omega: BL inputs (Q expected as 1D or 2D array compatible with pypfopt)
    market_caps: dict ticker->market cap
    market_prices: optional Series (market index prices) used to compute implied risk aversion
    Returns: dict ticker->weight
    """
    sigma = compute_sigma(prices, method='ledoit_wolf_constant_correlation', frequency=12)
    # market implied prior
    try:
        market_prior = market_prior_from_caps(market_caps, sigma, market_prices=market_prices, risk_free_rate=risk_free_rate, frequency=12)
    except Exception:
        # fallback: use mean historical returns
        mu = expected_returns.mean_historical_return(prices, frequency=12)
        market_prior = mu

    bl = BlackLittermanModel(sigma, pi=market_prior, P=P, Q=Q, omega=Omega)
    posterior_returns = bl.bl_returns()
    posterior_cov = bl.bl_cov()

    ef = EfficientFrontier(posterior_returns, posterior_cov)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate / 12)
    except Exception:
        ef.max_sharpe()
    weights = ef.clean_weights()
    out = {t: float(weights.get(t, 0.0)) for t in tickers}
    return out


def markowitz_opt(tickers: List[str], prices: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Simple Markowitz (max Sharpe) using mean historical returns and sample covariance.

    Returns cleaned weights dict.
    """
    mu = expected_returns.mean_historical_return(prices, frequency=12)
    sigma = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, sigma)
    ef.max_sharpe(risk_free_rate=risk_free_rate / 12)
    weights = ef.clean_weights()
    return {t: float(weights.get(t, 0.0)) for t in tickers}
