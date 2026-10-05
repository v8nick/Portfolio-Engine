from __future__ import annotations

import numpy as np
import pandas as pd

from black_litterman import black_litterman_posterior, resolve_strategic_prior_weights
from config.live import (
    BENCHMARK,
    BL_ABSOLUTE_VIEWS,
    BL_STRATEGIC_PRIOR_WEIGHTS,
    BL_RELATIVE_VIEWS,
    CURRENT_WEIGHTS,
    FIXED_WEIGHTS,
    IMPLEMENTATION_LAYER,
    LOOKBACK_END_DATE,
    LOOKBACK_START_DATE,
    MIN_WEIGHTS,
    TICKERS,
    TRADE_ACTION_THRESHOLD,
    TURNOVER_PENALTY_LAMBDA,
    WEIGHT_BOUNDS,
)
from config.shared import (
    APPLY_TAX_EFFECTS,
    BL_RISK_AVERSION,
    BL_TAU,
    COV_SHRINKAGE_METHOD,
    DEFAULT_UNREALIZED_GAIN_RATE,
    LONG_TERM_FRACTION,
    LONG_TERM_TAX_RATE,
    SHORT_TERM_TAX_RATE,
    SLIPPAGE_BPS,
    TRADING_DAYS,
    TRANSACTION_COST_BPS,
    USE_BLACK_LITTERMAN,
    USE_COV_SHRINKAGE,
)
from dashboard import (
    current_portfolio_performance,
    optimized_portfolio_summary,
    risk_contribution,
    suggested_weight_changes,
)
from data import compute_returns, download_prices, download_treasury_yields, resolve_risk_free_rate
from implementation import (
    apply_implementation_layer,
    implementation_summary_table,
    net_expected_return_after_all_costs,
)
from mpt import (
    annualize_mean_cov,
    optimize_max_sharpe_constrained,
    portfolio_stats,
    weights_dict_to_array,
)
from report import build_portfolio_return_series


def print_weights(title: str, tickers: list[str], weights: np.ndarray) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker:>5}: {weight:6.2%}")


def main() -> None:
    all_tickers = list(dict.fromkeys(TICKERS + [BENCHMARK]))

    prices = download_prices(all_tickers, LOOKBACK_START_DATE, LOOKBACK_END_DATE)
    returns = compute_returns(prices)
    treasury_yields = download_treasury_yields(prices.index.min().date().isoformat(), prices.index.max().date().isoformat())
    rf_info = resolve_risk_free_rate(as_of=prices.index.max(), yields=treasury_yields)
    rf = rf_info["rate"]
    print(f"\nRisk-free rate: {rf:.2%} ({rf_info['source']}; observation {rf_info['as_of']})")

    asset_returns = returns[TICKERS].dropna()
    benchmark_returns = returns[BENCHMARK].dropna()

    hist_mu, cov = annualize_mean_cov(
        asset_returns,
        trading_days=TRADING_DAYS,
        use_shrinkage=USE_COV_SHRINKAGE,
        shrinkage_method=COV_SHRINKAGE_METHOD,
    )

    if USE_COV_SHRINKAGE:
        print(f"\nUsing covariance shrinkage: {COV_SHRINKAGE_METHOD}")
    else:
        print("\nUsing sample covariance")

    if USE_BLACK_LITTERMAN:
        market_weights = resolve_strategic_prior_weights(TICKERS, BL_STRATEGIC_PRIOR_WEIGHTS)
        mu, pi = black_litterman_posterior(
            cov=cov,
            market_weights=market_weights,
            risk_aversion=BL_RISK_AVERSION,
            tau=BL_TAU,
            absolute_views=BL_ABSOLUTE_VIEWS,
            relative_views=BL_RELATIVE_VIEWS,
            risk_free_rate=rf,
        )

        print("\nImplied Equilibrium Returns (Pi)")
        print(pi.round(4))

        print("\nBlack-Litterman Posterior Returns")
        print(mu.round(4))
    else:
        mu = hist_mu
        pi = hist_mu.copy()

    mu_compare = pd.DataFrame(
        {
            "hist_mu": hist_mu,
            "pi": pi,
            "bl_mu": mu,
        }
    )
    print("\nReturn Estimate Comparison")
    print(mu_compare.round(4))

    current_weights = weights_dict_to_array(CURRENT_WEIGHTS, TICKERS)
    current_stats = portfolio_stats(current_weights, mu, cov, rf)

    mu_for_optimization = apply_implementation_layer(
        mu=mu,
        current_weights=current_weights,
        tickers=TICKERS,
        implementation_layer=IMPLEMENTATION_LAYER,
        turnover_penalty_lambda=TURNOVER_PENALTY_LAMBDA,
    )

    target_weights = optimize_max_sharpe_constrained(
        mu=mu_for_optimization,
        cov=cov,
        rf=rf,
        tickers=TICKERS,
        bounds=WEIGHT_BOUNDS,
        fixed_weights=FIXED_WEIGHTS,
        min_weights=MIN_WEIGHTS,
    )
    target_stats = portfolio_stats(target_weights, mu, cov, rf)

    current_series = build_portfolio_return_series(asset_returns, current_weights, TICKERS)

    print_weights("Current Portfolio Weights", TICKERS, current_weights)
    print("\nNote: live holdings are modeled as weights only, not shares, cash, or tax lots.")
    print("\nCurrent Portfolio Optimization Stats")
    print(current_stats)

    if IMPLEMENTATION_LAYER:
        unrealized_gains_rates = (
            np.array([DEFAULT_UNREALIZED_GAIN_RATE] * len(TICKERS))
            if APPLY_TAX_EFFECTS
            else None
        )
        implementation_report = net_expected_return_after_all_costs(
            gross_expected_return=target_stats["expected_return"],
            current_weights=current_weights,
            target_weights=target_weights,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            unrealized_gains_rates=unrealized_gains_rates,
            short_term_tax_rate=SHORT_TERM_TAX_RATE,
            long_term_tax_rate=LONG_TERM_TAX_RATE,
            long_term_fraction=LONG_TERM_FRACTION,
        )

        print("\n" + "=" * 60)
        print("IMPLEMENTATION COST ANALYSIS")
        print("=" * 60)
        print("Tax drag is an approximate estimate using assumed gains and holding periods; no tax lots.")
        for key, value in implementation_report.items():
            print(f"{key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("CURRENT PORTFOLIO PERFORMANCE")
    print("=" * 60)
    current_perf = current_portfolio_performance(
        portfolio_returns=current_series,
        benchmark_returns=benchmark_returns,
        rf=rf,
        trading_days=TRADING_DAYS,
    )
    print(current_perf.round(4))

    print("\n" + "=" * 60)
    print("TARGET PORTFOLIO")
    print("=" * 60)
    weights_df, stats_df = optimized_portfolio_summary(
        tickers=TICKERS,
        weights=target_weights,
        expected_return=target_stats["expected_return"],
        volatility=target_stats["volatility"],
        sharpe=target_stats["sharpe"],
    )
    print(stats_df.round(4))
    print(weights_df.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("SUGGESTED WEIGHT CHANGES")
    print("=" * 60)
    changes_df = suggested_weight_changes(
        tickers=TICKERS,
        current_weights=current_weights,
        optimized_weights=target_weights,
        threshold=TRADE_ACTION_THRESHOLD,
    )
    print(changes_df.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("TRADE SHEET")
    print("=" * 60)
    trade_df = implementation_summary_table(
        tickers=TICKERS,
        current_weights=current_weights,
        target_weights=target_weights,
    )
    print(trade_df.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("RISK CONTRIBUTION")
    print("=" * 60)
    risk_df = risk_contribution(
        tickers=TICKERS,
        weights=target_weights,
        cov=cov,
    )
    print(risk_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
