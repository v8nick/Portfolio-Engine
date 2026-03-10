# report.py

from __future__ import annotations
import numpy as np
import pandas as pd


def build_portfolio_return_series(
    returns: pd.DataFrame,
    weights: np.ndarray,
    asset_order: list[str]
) -> pd.Series:
    port = returns[asset_order].mul(weights, axis=1).sum(axis=1)
    port.name = "portfolio"
    return port


def cumulative_growth(return_series: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1 + return_series).cumprod()


def max_drawdown(return_series: pd.Series) -> float:
    wealth = cumulative_growth(return_series, 1.0)
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def annualized_return(return_series: pd.Series, trading_days: int = 252) -> float:
    n = len(return_series)
    if n == 0:
        return np.nan
    total_growth = (1 + return_series).prod()
    return float(total_growth ** (trading_days / n) - 1)


def annualized_volatility(return_series: pd.Series, trading_days: int = 252) -> float:
    return float(return_series.std() * np.sqrt(trading_days))


def sharpe_ratio(return_series: pd.Series, rf: float, trading_days: int = 252) -> float:
    ann_ret = annualized_return(return_series, trading_days)
    ann_vol = annualized_volatility(return_series, trading_days)
    if ann_vol == 0:
        return np.nan
    return float((ann_ret - rf) / ann_vol)


def beta_alpha_vs_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252
) -> dict:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]

    cov = aligned.cov().iloc[0, 1]
    var_b = aligned["benchmark"].var()
    beta = cov / var_b if var_b > 0 else np.nan

    alpha_daily = aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()
    alpha_annual = alpha_daily * trading_days

    corr = aligned["portfolio"].corr(aligned["benchmark"])

    tracking_error = (aligned["portfolio"] - aligned["benchmark"]).std() * np.sqrt(trading_days)
    active_return = (aligned["portfolio"] - aligned["benchmark"]).mean() * trading_days
    info_ratio = active_return / tracking_error if tracking_error > 0 else np.nan

    return {
        "beta": float(beta),
        "alpha_annual": float(alpha_annual),
        "correlation": float(corr),
        "tracking_error": float(tracking_error),
        "information_ratio": float(info_ratio),
    }


def summary_table(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float,
    trading_days: int = 252
) -> pd.DataFrame:
    stats = {
        "Annual Return": annualized_return(portfolio_returns, trading_days),
        "Annual Vol": annualized_volatility(portfolio_returns, trading_days),
        "Sharpe": sharpe_ratio(portfolio_returns, rf, trading_days),
        "Max Drawdown": max_drawdown(portfolio_returns),
    }

    benchmark_stats = beta_alpha_vs_benchmark(portfolio_returns, benchmark_returns, trading_days)
    stats.update({
        "Beta vs Benchmark": benchmark_stats["beta"],
        "Alpha vs Benchmark": benchmark_stats["alpha_annual"],
        "Correlation vs Benchmark": benchmark_stats["correlation"],
        "Tracking Error": benchmark_stats["tracking_error"],
        "Information Ratio": benchmark_stats["information_ratio"],
    })

    return pd.DataFrame(stats, index=["Portfolio"]).T