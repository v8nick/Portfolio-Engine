from __future__ import annotations
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models

from pypfopt import risk_models

def annualize_mean_cov(
    returns: pd.DataFrame,
    trading_days: int = 252,
    use_shrinkage: bool = False,
    shrinkage_method: str = "ledoit_wolf",
) -> tuple[pd.Series, pd.DataFrame]:
    mu = returns.mean() * trading_days

    if use_shrinkage:
        cov = risk_models.risk_matrix(
            returns,
            method=shrinkage_method,
            returns_data=True,
            frequency=trading_days,
        )
    else:
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


def weights_dict_to_array(weight_map: dict[str, float], ordered_assets: list[str]) -> np.ndarray:
    return np.array([weight_map[t] for t in ordered_assets], dtype=float)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights)
    if total == 0:
        raise ValueError("Weight sum is zero.")
    return weights / total


def _clean_weights_to_array(cleaned: dict[str, float], tickers: list[str]) -> np.ndarray:
    return np.array([cleaned.get(t, 0.0) for t in tickers], dtype=float)


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    bounds: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    ef.max_sharpe(risk_free_rate=rf)
    cleaned = ef.clean_weights()
    return _clean_weights_to_array(cleaned, list(mu.index))


def optimize_min_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    bounds: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    ef.min_volatility()
    cleaned = ef.clean_weights()
    return _clean_weights_to_array(cleaned, list(mu.index))


def optimize_max_sharpe_constrained(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    tickers: list[str],
    bounds: tuple[float, float] = (0.0, 1.0),
    fixed_weights: dict[str, float] | None = None,
    min_weights: dict[str, float] | None = None,
) -> np.ndarray:
    fixed_weights = fixed_weights or {}
    min_weights = min_weights or {}

    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)

    for ticker, value in fixed_weights.items():
        ef.add_constraint(lambda w, t=tickers.index(ticker), v=value: w[t] == v)

    for ticker, value in min_weights.items():
        ef.add_constraint(lambda w, t=tickers.index(ticker), v=value: w[t] >= v)

    ef.max_sharpe(risk_free_rate=rf)
    cleaned = ef.clean_weights()
    return _clean_weights_to_array(cleaned, tickers)