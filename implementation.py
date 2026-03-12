from __future__ import annotations
import numpy as np
import pandas as pd


def compute_turnover(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """
    One-way turnover approximation.
    """
    return float(np.sum(np.abs(target_weights - current_weights)))


def implementation_cost_rate(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> float:
    """
    Approximate implementation cost as turnover times per-unit cost.
    """
    turnover = compute_turnover(current_weights, target_weights)
    per_unit_cost = (transaction_cost_bps + slippage_bps) / 10000.0
    return turnover * per_unit_cost


def penalize_expected_returns_for_turnover(
    mu: pd.Series,
    current_weights: np.ndarray,
    tickers: list[str],
    penalty_lambda: float,
) -> pd.Series:
    """
    Simple heuristic turnover penalty.
    Assets farther from current holdings become less attractive.
    """
    current_map = pd.Series(current_weights, index=tickers)
    penalty = penalty_lambda * (1.0 - current_map)
    adjusted = mu - penalty * 0.01
    adjusted.name = "mu_net_turnover_penalty"
    return adjusted


def estimate_tax_drag(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    unrealized_gains_rates: np.ndarray,
    short_term_tax_rate: float,
    long_term_tax_rate: float,
    long_term_fraction: float = 0.5,
) -> float:
    """
    Rough tax drag estimate from selling appreciated positions.
    """
    sells = np.maximum(current_weights - target_weights, 0.0)

    effective_tax_rate = (
        long_term_fraction * long_term_tax_rate
        + (1.0 - long_term_fraction) * short_term_tax_rate
    )

    taxable_realized_gain = np.sum(sells * np.maximum(unrealized_gains_rates, 0.0))
    tax_drag = float(taxable_realized_gain * effective_tax_rate)
    return tax_drag


def net_expected_return_after_all_costs(
    gross_expected_return: float,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    transaction_cost_bps: float,
    slippage_bps: float,
    unrealized_gains_rates: np.ndarray | None = None,
    short_term_tax_rate: float = 0.35,
    long_term_tax_rate: float = 0.15,
    long_term_fraction: float = 0.5,
) -> dict:
    impl_cost = implementation_cost_rate(
        current_weights=current_weights,
        target_weights=target_weights,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )

    tax_drag = 0.0
    if unrealized_gains_rates is not None:
        tax_drag = estimate_tax_drag(
            current_weights=current_weights,
            target_weights=target_weights,
            unrealized_gains_rates=unrealized_gains_rates,
            short_term_tax_rate=short_term_tax_rate,
            long_term_tax_rate=long_term_tax_rate,
            long_term_fraction=long_term_fraction,
        )

    net_ret = gross_expected_return - impl_cost - tax_drag

    return {
        "gross_expected_return": gross_expected_return,
        "implementation_cost": impl_cost,
        "tax_drag": tax_drag,
        "net_expected_return": net_ret,
        "turnover": compute_turnover(current_weights, target_weights),
    }


def implementation_summary_table(
    tickers: list[str],
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "Ticker": tickers,
        "Current Weight": current_weights,
        "Target Weight": target_weights,
    })
    df["Trade"] = df["Target Weight"] - df["Current Weight"]
    df["Action"] = np.where(
        df["Trade"] > 0, "Buy",
        np.where(df["Trade"] < 0, "Sell", "Hold")
    )
    return df


def apply_implementation_layer(
    mu: pd.Series,
    current_weights: np.ndarray,
    tickers: list[str],
    implementation_layer: bool,
    turnover_penalty_lambda: float,
) -> pd.Series:
    """
    Single wrapper for on/off behavior.
    If implementation_layer is False, returns original mu unchanged.
    """
    if not implementation_layer:
        return mu.copy()

    return penalize_expected_returns_for_turnover(
        mu=mu,
        current_weights=current_weights,
        tickers=tickers,
        penalty_lambda=turnover_penalty_lambda,
    )