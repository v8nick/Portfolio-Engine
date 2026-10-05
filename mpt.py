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
    return np.asarray(ef.weights, dtype=float)


def optimize_min_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    bounds: tuple[float, float] = (0.0, 1.0),
    tickers: list[str] | None = None,
    fixed_weights: dict[str, float] | None = None,
    min_weights: dict[str, float] | None = None,
    max_group_weights: dict | None = None,
) -> np.ndarray:
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    _add_constraints(ef, tickers or list(mu.index), fixed_weights, min_weights, max_group_weights)
    ef.min_volatility()
    return np.asarray(ef.weights, dtype=float)


def optimize_max_sharpe_constrained(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    tickers: list[str],
    bounds: tuple[float, float] = (0.0, 1.0),
    fixed_weights: dict[str, float] | None = None,
    min_weights: dict[str, float] | None = None,
    max_group_weights: dict | None = None,
) -> np.ndarray:
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    _add_constraints(ef, tickers, fixed_weights, min_weights, max_group_weights)

    ef.max_sharpe(risk_free_rate=rf)
    return np.asarray(ef.weights, dtype=float)


def _add_constraints(optimizer, tickers, fixed_weights=None, min_weights=None, max_group_weights=None):
    for ticker, value in (fixed_weights or {}).items():
        optimizer.add_constraint(lambda w, t=tickers.index(ticker), v=value: w[t] == v)
    for ticker, value in (min_weights or {}).items():
        optimizer.add_constraint(lambda w, t=tickers.index(ticker), v=value: w[t] >= v)
    for names, cap in (max_group_weights or {}).values():
        indices = [tickers.index(t) for t in names]
        if indices:
            optimizer.add_constraint(lambda w, idx=indices, v=cap: w[idx].sum() <= v)


def _scipy_constraints(tickers, max_group_weights=None):
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    for names, cap in (max_group_weights or {}).values():
        indices = [tickers.index(t) for t in names]
        if indices:
            constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices, v=cap: v - w[idx].sum()})
    return constraints


def validate_allocation_weights(weights, tickers, bounds, max_group_weights=None, tolerance=1e-6):
    weights = np.asarray(weights, dtype=float)
    limits = np.asarray(bounds, dtype=float)
    if limits.shape == (2,):
        limits = np.tile(limits, (len(tickers), 1))
    if weights.shape != (len(tickers),) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1, atol=tolerance, rtol=0):
        raise ValueError('Allocation must be finite, aligned, and sum to one.')
    if (weights < limits[:, 0] - tolerance).any() or (weights > limits[:, 1] + tolerance).any():
        raise ValueError('Allocation violates asset bounds.')
    for names, cap in (max_group_weights or {}).values():
        if sum(weights[tickers.index(t)] for t in names) > cap + tolerance:
            raise ValueError('Allocation violates a group cap.')


def project_allocation(target, tickers, bounds, initial_weights, max_group_weights=None):
    """Closest feasible budget-balanced allocation, including fixed/band/group limits."""
    from scipy.optimize import minimize
    result = minimize(lambda w: np.sum((w - target) ** 2), initial_weights,
                      method='SLSQP', bounds=bounds,
                      constraints=_scipy_constraints(tickers, max_group_weights),
                      options={'ftol': 1e-12, 'maxiter': 500})
    if not result.success:
        raise ValueError(f'Allocation projection failed: {result.message}')
    validate_allocation_weights(result.x, tickers, bounds, max_group_weights)
    return result.x


def optimize_risk_parity(cov: pd.DataFrame, bounds, initial_weights,
                         max_group_weights=None) -> np.ndarray:
    """Approximate equal percentage risk contributions under the same hard limits.

    Bounds, fixed positions and negative covariances can prevent exact ERC.
    """
    from scipy.optimize import minimize
    sigma = cov.to_numpy()
    def objective(w):
        variance = w @ sigma @ w
        if variance <= 0:
            return 1e6
        percent = w * (sigma @ w) / variance
        return float(np.sum((percent - 1 / len(w)) ** 2))
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds,
                      constraints=_scipy_constraints(list(cov.index), max_group_weights),
                      options={'ftol': 1e-12, 'maxiter': 1000})
    if not result.success:
        raise ValueError(f'Risk parity failed: {result.message}')
    validate_allocation_weights(result.x, list(cov.index), bounds, max_group_weights)
    return result.x


def optimize_min_cvar(mu: pd.Series, returns: pd.DataFrame, bounds,
                      beta: float = .95, max_group_weights=None) -> np.ndarray:
    """PyPortfolioOpt/cvxpy empirical daily expected-shortfall minimization."""
    from pypfopt import EfficientCVaR
    ef = EfficientCVaR(mu, returns, beta=beta, weight_bounds=bounds)
    _add_constraints(ef, list(mu.index), max_group_weights=max_group_weights)
    ef.min_cvar()
    return np.asarray(ef.weights, dtype=float)
