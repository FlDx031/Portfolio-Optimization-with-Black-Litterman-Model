"""A manual monthly loop: estimate, choose weights, pay costs, hold, repeat."""

import numpy as np
import pandas as pd

from .data import monthly_schedule, validate_prices
from .model import (black_litterman, covariance, momentum_views, optimal_weights,
                    reference_prior, ridge_views)


def trading_cost(target, current, cost_rate):
    """Approximate costs as rate × total absolute traded weight (buys + sells).

    Initial cash -> portfolio costs one unit of turnover, not half a unit.
    Costs are deducted before the holding-period return; weights ignore the tiny
    second-order adjustment needed to finance fees in an exact share simulation.
    """
    turnover = float(np.abs(np.asarray(target) - np.asarray(current)).sum())
    return turnover, cost_rate * turnover


def performance(returns, rf_annual=0.02):
    """Monthly net returns; start wealth at 1 so the first loss counts in drawdown."""
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Metrics need at least two finite monthly returns above -100%.")
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    volatility = values.std(ddof=1) * np.sqrt(12)
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    return {"Total return": wealth[-1] - 1,
            "CAGR": wealth[-1] ** (12 / len(values)) - 1,
            "Volatility": volatility,
            "Sharpe": (values.mean() - rf_monthly) * 12 / volatility if volatility > 0 else np.nan,
            "Max drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min()}


def run_backtest(prices, assets, benchmark="SPY", macro=None, lookback=36,
                 ridge_window=60, min_train=24, validation_months=12,
                 rf_annual=0.02, risk_aversion=3.0, tau=0.05,
                 momentum_strength=0.005, ridge_alpha=1.0, cost_bps=10.0):
    """Compare equal weight, historical Markowitz, BL momentum, optionally BL Ridge.

    All strategies start/finish together. SPY is buy-and-hold with entry costs.
    Covariance and views are monthly, unannualized. No use of future labels,
    current capitalizations, automatic model fallback, or partial final months.
    """
    validate_prices(prices)
    assets = list(assets)
    if len(assets) < 2 or len(set(assets)) != len(assets):
        raise ValueError("Use at least two distinct portfolio assets.")
    if not set(assets + [benchmark]).issubset(prices.columns):
        raise ValueError("Missing asset or benchmark prices.")
    if lookback < 12 or min_train < 2 or validation_months < 1:
        raise ValueError("Use lookback >= 12, min_train >= 2, validation_months >= 1.")
    if ridge_window < min_train + validation_months:
        raise ValueError("ridge_window must cover training and validation.")
    if not (0 <= cost_bps < 1000) or rf_annual <= -1 or min(risk_aversion, tau, ridge_alpha) <= 0:
        raise ValueError("Invalid cost, risk-free rate or model parameters.")
    schedule = monthly_schedule(prices)
    if len(schedule) < 3:
        raise ValueError("Not enough complete months.")
    execution_prices = prices.loc[schedule.execution, assets].to_numpy()
    realized = execution_prices[1:] / execution_prices[:-1] - 1
    monthly_prices = prices.loc[schedule.signal, assets]
    month_returns = monthly_prices.pct_change(fill_method=None)
    if macro is not None:
        if macro.index.has_duplicates:
            raise ValueError("Macro dates must be unique.")
        macro = macro.reindex(pd.DatetimeIndex(schedule.signal))
        if not np.isfinite(macro.to_numpy()).all():
            raise ValueError("Macro must have a finite feature row for every signal date.")

    names = ["Equal weight", "Markowitz", "BL Momentum"]
    if macro is not None:
        names.append("BL Ridge")
    current = {name: np.zeros(len(assets)) for name in names}
    returns, turnovers, fees, weight_rows, dates, signal_rows = [], [], [], [], [], []
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    cost_rate = cost_bps / 10_000

    for i in range(len(schedule) - 1):
        signal, execution = schedule.iloc[i]
        exit_date = schedule.execution.iloc[i + 1]
        # Label j becomes known at execution[j+1], strictly before this signal.
        known = np.flatnonzero(schedule.execution.iloc[1:].to_numpy() <= signal.to_datetime64())
        history = month_returns.iloc[:i + 1].dropna().tail(lookback)
        if len(history) < lookback or len(known) < min_train + validation_months:
            continue  # Same warm-up even when macro is disabled.
        sigma = covariance(history)
        prior = reference_prior(sigma, risk_aversion)
        q, omega = momentum_views(monthly_prices.iloc[:i + 1], prior, sigma,
                                  momentum_strength, tau)
        mu_bl, sigma_bl = black_litterman(prior, sigma, q, omega, tau)
        targets = {
            "Equal weight": np.full(len(assets), 1 / len(assets)),
            "Markowitz": optimal_weights(history.mean().to_numpy() - rf_monthly, sigma, risk_aversion),
            "BL Momentum": optimal_weights(mu_bl, sigma_bl, risk_aversion),
        }
        if macro is not None:
            training = known[-ridge_window:]
            q, omega = ridge_views(macro.iloc[training], realized[training], macro.iloc[i],
                                    rf_monthly, ridge_alpha, min_train, validation_months)
            mu_bl, sigma_bl = black_litterman(prior, sigma, q, omega, tau)
            targets["BL Ridge"] = optimal_weights(mu_bl, sigma_bl, risk_aversion)

        period_returns, period_turnovers, period_fees = {}, {}, {}
        for name, weights in targets.items():
            turnover, fee = trading_cost(weights, current[name], cost_rate)
            gross_return = float(weights @ realized[i])
            period_returns[name] = (1 - fee) * (1 + gross_return) - 1
            period_turnovers[name], period_fees[name] = turnover, fee
            current[name] = weights * (1 + realized[i]) / (1 + gross_return)
            weight_rows.append(dict(strategy=name, date=execution, **dict(zip(assets, weights))))
        benchmark_return = prices.loc[exit_date, benchmark] / prices.loc[execution, benchmark] - 1
        entry = not dates
        period_returns[benchmark] = (1 - cost_rate * entry) * (1 + benchmark_return) - 1
        period_turnovers[benchmark], period_fees[benchmark] = float(entry), cost_rate * entry
        dates.append(exit_date)
        signal_rows.append(dict(signal=signal, execution=execution, exit=exit_date,
                                last_known_target_end=schedule.execution.iloc[known[-1] + 1]))
        returns.append(period_returns)
        turnovers.append(period_turnovers)
        fees.append(period_fees)

    if len(returns) < 2:
        raise ValueError("Not enough history after warm-up. Download more years.")
    returns = pd.DataFrame(returns, index=pd.DatetimeIndex(dates, name="date"))
    turnover = pd.DataFrame(turnovers, index=returns.index)
    wealth = (1 + returns).cumprod()
    first_execution = signal_rows[0]["execution"]
    wealth = pd.concat([pd.DataFrame(1.0, index=[first_execution], columns=wealth.columns), wealth])
    metrics = pd.DataFrame({name: performance(returns[name], rf_annual) for name in returns}).T
    metrics["Annual turnover"] = turnover.mean() * 12
    return {"returns": returns, "wealth": wealth, "drawdown": wealth / wealth.cummax() - 1,
            "metrics": metrics, "weights": pd.DataFrame(weight_rows).set_index(["strategy", "date"]),
            "turnover": turnover, "fees": pd.DataFrame(fees, index=returns.index),
            "timeline": pd.DataFrame(signal_rows)}
