# mpt.py

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def annualize_mean_cov(
    returns: pd.DataFrame,
    trading_days: int = 252
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute annualized mean returns and covariance matrix.
    """
    mu = returns.mean() * trading_days
    cov = returns.cov() * trading_days
    return mu, cov


def portfolio_return(weights: np.ndarray, mu: pd.Series) -> float:
    return float(weights @ mu.values)


def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
    return float(np.sqrt(weights @ cov.values @ weights))


def portfolio_sharpe(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> float:
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return np.nan
    ret = portfolio_return(weights, mu)
    return (ret - rf) / vol


def portfolio_stats(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> dict:
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return {
        "expected_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


def _weight_sum_constraint(weights: np.ndarray) -> float:
    return np.sum(weights) - 1.0


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    bounds: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Maximize Sharpe ratio subject to weights summing to 1.
    """
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bnds = [bounds] * n
    constraints = [{"type": "eq", "fun": _weight_sum_constraint}]

    def objective(weights: np.ndarray) -> float:
        sharpe = portfolio_sharpe(weights, mu, cov, rf)
        return -sharpe

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bnds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")

    return result.x


def optimize_min_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    bounds: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Minimize volatility subject to weights summing to 1.
    """
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bnds = [bounds] * n
    constraints = [{"type": "eq", "fun": _weight_sum_constraint}]

    def objective(weights: np.ndarray) -> float:
        return portfolio_volatility(weights, cov)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bnds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Min-vol optimization failed: {result.message}")

    return result.x


def weights_dict_to_array(weight_map: dict[str, float], ordered_assets: list[str]) -> np.ndarray:
    return np.array([weight_map[t] for t in ordered_assets], dtype=float)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights)
    if total == 0:
        raise ValueError("Weight sum is zero.")
    return weights / total

def optimize_max_sharpe_constrained(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    tickers: list[str],
    bounds: tuple[float, float] = (0.0, 1.0),
    fixed_weights: dict[str, float] | None = None,
    min_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Maximize Sharpe ratio with custom fixed and minimum weight constraints.
    """
    fixed_weights = fixed_weights or {}
    min_weights = min_weights or {}

    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bnds = [bounds] * n

    constraints = [{"type": "eq", "fun": _weight_sum_constraint}]

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    for ticker, value in fixed_weights.items():
        idx = ticker_to_idx[ticker]
        constraints.append({
            "type": "eq",
            "fun": lambda w, idx=idx, value=value: w[idx] - value
        })

    for ticker, value in min_weights.items():
        idx = ticker_to_idx[ticker]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=idx, value=value: w[idx] - value
        })

    def objective(weights: np.ndarray) -> float:
        sharpe = portfolio_sharpe(weights, mu, cov, rf)
        return -sharpe

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bnds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Constrained Max-Sharpe optimization failed: {result.message}")

    return result.x