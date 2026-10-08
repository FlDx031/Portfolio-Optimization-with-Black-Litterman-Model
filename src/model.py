"""Monthly Black–Litterman, momentum and a small Ridge regression."""

import numpy as np
from scipy.optimize import minimize


def covariance(returns, shrinkage=0.2):
    """Monthly sample covariance, shrunk 20% toward its diagonal (fixed choice)."""
    sample = np.atleast_2d(np.cov(np.asarray(returns), rowvar=False, ddof=1))
    return (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample)) + np.eye(len(sample)) * 1e-10


def reference_prior(sigma, risk_aversion=3.0):
    """Excess-return prior implied by a fixed equal-weight reference, NOT market caps."""
    return risk_aversion * sigma @ np.full(len(sigma), 1 / len(sigma))


def momentum_views(monthly_prices, prior, sigma, strength=0.005, tau=0.05):
    """11-month momentum ending one month before the signal; monthly excess views."""
    if len(monthly_prices) < 13:
        raise ValueError("Momentum needs at least 13 month-end prices.")
    prices = np.asarray(monthly_prices)
    momentum = prices[-2] / prices[-13] - 1
    spread = momentum.std()
    z = (momentum - momentum.mean()) / spread if spread > 1e-12 else np.zeros_like(momentum)
    return prior + strength * z, np.diag(np.diag(tau * sigma))


def ridge_predict(x, y, latest, alpha=1.0):
    """Standardize on training rows only; unpenalized intercept, multi-asset targets."""
    x, y, latest = np.asarray(x), np.asarray(y), np.asarray(latest)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    intercept = y.mean(axis=0)
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - intercept))
    return intercept + ((latest - mean) / scale) @ beta


def ridge_views(x, y, latest, rf_monthly, alpha=1.0, min_train=24, validation_months=12):
    """Past-only rolling-origin errors set Omega; refit on known labels for Q.

    The caller supplies only rows whose holding-period return has already ended.
    Each validation prediction is fitted before its target, including its scaler.
    """
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < min_train + validation_months:
        raise ValueError("Ridge needs training + chronological validation history.")
    errors = []
    for split in range(len(x) - validation_months, len(x)):
        prediction = ridge_predict(x[:split], y[:split], x[split], alpha)
        errors.append(y[split] - prediction)
    mse = np.maximum(np.mean(np.square(errors), axis=0), 1e-8)
    prediction = ridge_predict(x, y, latest, alpha)
    return prediction - rf_monthly, np.diag(mse)


def black_litterman(prior, sigma, views, omega, tau=0.05):
    """Absolute views (P = I). Monthly inputs; prior and views are excess returns."""
    uncertainty = tau * sigma
    gain = np.linalg.solve(uncertainty + omega, uncertainty).T
    posterior = prior + gain @ (views - prior)
    posterior_cov = sigma + uncertainty - gain @ uncertainty
    return posterior, (posterior_cov + posterior_cov.T) / 2


def optimal_weights(expected_excess, sigma, risk_aversion=3.0):
    """Long-only mean–variance utility, fully invested. No silent fallback strategy."""
    n = len(expected_excess)
    initial = np.full(n, 1 / n)
    result = minimize(
        lambda w: risk_aversion / 2 * w @ sigma @ w - w @ expected_excess,
        initial, jac=lambda w: risk_aversion * sigma @ w - expected_excess,
        bounds=[(0.0, 1.0)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                            "jac": lambda w: np.ones(n)},
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Optimization failed: {result.message}")
    if result.x.min() < -1e-7 or abs(result.x.sum() - 1) > 1e-7:
        raise RuntimeError("Optimizer returned invalid weights.")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()
