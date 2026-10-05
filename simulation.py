#simulation.py monte carlo simulation of portfolio returns


from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_portfolio_paths(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    years: int = 10,
    n_sims: int = 10000,
    trading_days: int = 252,
    initial_value: float = 1.0,
    seed: int = 42,
    method: str = "gaussian",
) -> pd.DataFrame:
    """
    Simulate Monte Carlo portfolio value paths using multivariate normal daily returns.
    IID Gaussian simple returns with constant moments and constant weights.
    Ignores tails, regime shifts, taxes and costs; normal draws can be below -100%.
    The method parameter reserves the interface for future simulation models.
    Returns a DataFrame where each column is one simulated path.
    """
    if method != "gaussian":
        raise ValueError("Only the gaussian baseline is currently implemented.")
    if years <= 0 or n_sims <= 0 or trading_days <= 0 or initial_value <= 0:
        raise ValueError("Simulation horizon, count, frequency and initial value must be positive.")
    rng = np.random.default_rng(seed)
    n_assets = len(weights)
    n_steps = years * trading_days

    simulated_asset_returns = rng.multivariate_normal(
        mean=mu.values / trading_days,
        cov=cov.values / trading_days,
        size=(n_steps, n_sims),
    )
    # shape is (n_steps, n_sims, n_assets) if mean is vector
    simulated_asset_returns = np.asarray(simulated_asset_returns)

    portfolio_returns = np.einsum("tna,a->tn", simulated_asset_returns, weights)

    portfolio_values = initial_value * np.cumprod(1 + portfolio_returns, axis=0)

    columns = [f"sim_{i+1}" for i in range(n_sims)]
    index = pd.RangeIndex(start=1, stop=n_steps + 1, step=1, name="step")
    paths = pd.DataFrame(portfolio_values, index=index, columns=columns)
    paths.attrs.update(initial_value=initial_value, method=method, assumptions="IID Gaussian simple returns; constant moments/weights; no costs or taxes")
    return paths


def terminal_value_stats(paths: pd.DataFrame) -> dict:
    """
    Summary statistics for ending portfolio values.
    """
    terminal = paths.iloc[-1]
    initial = paths.attrs.get("initial_value", 1.0)

    return {
        "mean_terminal_value": float(terminal.mean()),
        "median_terminal_value": float(terminal.median()),
        "p5_terminal_value": float(terminal.quantile(0.05)),
        "p25_terminal_value": float(terminal.quantile(0.25)),
        "p75_terminal_value": float(terminal.quantile(0.75)),
        "p95_terminal_value": float(terminal.quantile(0.95)),
        "prob_loss": float((terminal < initial).mean()),
        "prob_double": float((terminal >= 2 * initial).mean()),
        "prob_5x": float((terminal >= 5 * initial).mean()),
    }


def path_max_drawdown(path: pd.Series, initial_value: float = 1.0) -> float:
    """
    Max drawdown for one simulated portfolio path.
    """
    peak = path.cummax().clip(lower=initial_value)
    drawdown = path / peak - 1.0
    return float(drawdown.min())


def drawdown_stats(paths: pd.DataFrame) -> dict:
    """
    Summary stats for max drawdowns across all simulations.
    """
    drawdowns = paths.apply(path_max_drawdown, axis=0, initial_value=paths.attrs.get("initial_value", 1.0))

    return {
        "mean_max_drawdown": float(drawdowns.mean()),
        "median_max_drawdown": float(drawdowns.median()),
        "p95_worst_drawdown": float(drawdowns.quantile(0.05)),
        "prob_drawdown_30": float((drawdowns <= -0.30).mean()),
        "prob_drawdown_50": float((drawdowns <= -0.50).mean()),
        "prob_drawdown_60": float((drawdowns <= -0.60).mean()),
    }