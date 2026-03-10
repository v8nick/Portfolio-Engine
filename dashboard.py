from __future__ import annotations
import numpy as np
import pandas as pd

from report import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    beta_alpha_vs_benchmark,
)


def current_portfolio_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float,
    trading_days: int = 252,
) -> pd.DataFrame:
    beta_stats = beta_alpha_vs_benchmark(
        portfolio_returns,
        benchmark_returns,
        trading_days=trading_days,
    )

    out = {
        "Annual Return": annualized_return(portfolio_returns, trading_days),
        "Annual Volatility": annualized_volatility(portfolio_returns, trading_days),
        "Sharpe Ratio": sharpe_ratio(portfolio_returns, rf, trading_days),
        "Max Drawdown": max_drawdown(portfolio_returns),
        "Beta vs Benchmark": beta_stats["beta"],
        "Alpha vs Benchmark": beta_stats["alpha_annual"],
        "Correlation vs Benchmark": beta_stats["correlation"],
        "Tracking Error": beta_stats["tracking_error"],
        "Information Ratio": beta_stats["information_ratio"],
    }

    return pd.DataFrame(out, index=["Current Portfolio"]).T


def optimized_portfolio_summary(
    tickers: list[str],
    weights: np.ndarray,
    expected_return: float,
    volatility: float,
    sharpe: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Optimized Weight": weights,
    }).sort_values("Optimized Weight", ascending=False)

    stats_df = pd.DataFrame({
        "Metric": ["Expected Return", "Volatility", "Sharpe Ratio"],
        "Value": [expected_return, volatility, sharpe],
    })

    return weights_df, stats_df


def suggested_weight_changes(
    tickers: list[str],
    current_weights: np.ndarray,
    optimized_weights: np.ndarray,
    threshold: float = 0.005,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "Ticker": tickers,
        "Current Weight": current_weights,
        "Optimized Weight": optimized_weights,
    })

    df["Change"] = df["Optimized Weight"] - df["Current Weight"]

    def action(x: float) -> str:
        if x > threshold:
            return "Increase"
        if x < -threshold:
            return "Decrease"
        return "Hold"

    df["Suggested Action"] = df["Change"].apply(action)
    return df.sort_values("Change", ascending=False)


def risk_contribution(
    tickers: list[str],
    weights: np.ndarray,
    cov: pd.DataFrame,
) -> pd.DataFrame:
    sigma = cov.values
    port_vol = float(np.sqrt(weights @ sigma @ weights))

    # Marginal contribution to risk
    mrc = (sigma @ weights) / port_vol

    # Component contribution to risk
    crc = weights * mrc

    # Percent contribution
    pct = crc / port_vol

    df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights,
        "Marginal Risk Contribution": mrc,
        "Component Risk Contribution": crc,
        "Percent Risk Contribution": pct,
    })

    return df.sort_values("Percent Risk Contribution", ascending=False)


def monte_carlo_risk_report(
    terminal_stats: dict,
    drawdown_stats: dict,
) -> pd.DataFrame:
    rows = {
        "Mean Terminal Value": terminal_stats["mean_terminal_value"],
        "Median Terminal Value": terminal_stats["median_terminal_value"],
        "5th Percentile Terminal Value": terminal_stats["p5_terminal_value"],
        "25th Percentile Terminal Value": terminal_stats["p25_terminal_value"],
        "75th Percentile Terminal Value": terminal_stats["p75_terminal_value"],
        "95th Percentile Terminal Value": terminal_stats["p95_terminal_value"],
        "Probability of Loss": terminal_stats["prob_loss"],
        "Probability of Double": terminal_stats["prob_double"],
        "Probability of 5x": terminal_stats["prob_5x"],
        "Mean Max Drawdown": drawdown_stats["mean_max_drawdown"],
        "Median Max Drawdown": drawdown_stats["median_max_drawdown"],
        "95th Worst Drawdown": drawdown_stats["p95_worst_drawdown"],
        "Probability of 30% Drawdown": drawdown_stats["prob_drawdown_30"],
        "Probability of 50% Drawdown": drawdown_stats["prob_drawdown_50"],
        "Probability of 60% Drawdown": drawdown_stats["prob_drawdown_60"],
    }

    return pd.DataFrame(rows, index=["Monte Carlo"]).T