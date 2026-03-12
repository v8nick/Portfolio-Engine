from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from black_litterman import black_litterman_posterior
from config.research import (
    BENCHMARK,
    BL_ABSOLUTE_VIEWS,
    BL_MARKET_WEIGHTS,
    BL_RELATIVE_VIEWS,
    END_DATE,
    FIXED_WEIGHTS,
    GENERATE_QUANTSTATS_REPORT,
    IMPLEMENTATION_LAYER,
    MC_INITIAL_VALUE,
    MC_N_SIMS,
    MC_SEED,
    MC_YEARS,
    MIN_WEIGHTS,
    QUANTSTATS_OUTPUT,
    RUN_WALKFORWARD_BACKTEST,
    START_DATE,
    STARTING_WEIGHTS,
    TICKERS,
    TURNOVER_PENALTY_LAMBDA,
    WALKFORWARD_QUANTSTATS_OUTPUT,
    WEIGHT_BOUNDS,
    WF_REBALANCE_FREQ,
    WF_WINDOW_DAYS,
)
from config.shared import (
    APPLY_TAX_EFFECTS,
    BL_RISK_AVERSION,
    BL_TAU,
    COV_SHRINKAGE_METHOD,
    DEFAULT_UNREALIZED_GAIN_RATE,
    LONG_TERM_FRACTION,
    LONG_TERM_TAX_RATE,
    RISK_FREE_RATE,
    SHORT_TERM_TAX_RATE,
    SLIPPAGE_BPS,
    TRADING_DAYS,
    TRANSACTION_COST_BPS,
    USE_BLACK_LITTERMAN,
    USE_COV_SHRINKAGE,
)
from dashboard import monte_carlo_risk_report, risk_contribution
from data import compute_returns, download_prices
from factors import factor_summary_table, fama_french_regression
from frontier import efficient_frontier
from implementation import apply_implementation_layer, net_expected_return_after_all_costs
from mpt import (
    annualize_mean_cov,
    optimize_max_sharpe_constrained,
    optimize_min_vol,
    portfolio_stats,
    weights_dict_to_array,
)
from report import (
    build_portfolio_return_series,
    cumulative_growth,
    export_quantstats_report,
    summary_table,
)
from rollingfront import rolling_statistics
from simulation import drawdown_stats, simulate_portfolio_paths, terminal_value_stats
from walkforward import rolling_black_litterman_backtest


def print_weights(title: str, tickers: list[str], weights: np.ndarray) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker:>5}: {weight:6.2%}")


def maybe_print_relative_view_note(relative_views: list[tuple[str, str, float, float]]) -> None:
    if relative_views:
        print("\nNote: BL_RELATIVE_VIEWS is configured but not yet applied by black_litterman.py.")


def main() -> None:
    all_tickers = list(dict.fromkeys(TICKERS + [BENCHMARK]))

    prices = download_prices(all_tickers, START_DATE, END_DATE)
    returns = compute_returns(prices)

    asset_returns = returns[TICKERS]
    benchmark_returns = returns[BENCHMARK]

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

    maybe_print_relative_view_note(BL_RELATIVE_VIEWS)

    if USE_BLACK_LITTERMAN:
        market_weights = weights_dict_to_array(BL_MARKET_WEIGHTS, TICKERS)
        mu, pi = black_litterman_posterior(
            cov=cov,
            market_weights=market_weights,
            risk_aversion=BL_RISK_AVERSION,
            tau=BL_TAU,
            absolute_views=BL_ABSOLUTE_VIEWS,
            relative_views=BL_RELATIVE_VIEWS,
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

    starting_weights = weights_dict_to_array(STARTING_WEIGHTS, TICKERS)

    mu_for_optimization = apply_implementation_layer(
        mu=mu,
        current_weights=starting_weights,
        tickers=TICKERS,
        implementation_layer=IMPLEMENTATION_LAYER,
        turnover_penalty_lambda=TURNOVER_PENALTY_LAMBDA,
    )

    starting_stats = portfolio_stats(starting_weights, mu, cov, RISK_FREE_RATE)
    max_sharpe_weights = optimize_max_sharpe_constrained(
        mu=mu_for_optimization,
        cov=cov,
        rf=RISK_FREE_RATE,
        tickers=TICKERS,
        bounds=WEIGHT_BOUNDS,
        fixed_weights=FIXED_WEIGHTS,
        min_weights=MIN_WEIGHTS,
    )
    min_vol_weights = optimize_min_vol(mu, cov, bounds=WEIGHT_BOUNDS)

    max_sharpe_stats = portfolio_stats(max_sharpe_weights, mu, cov, RISK_FREE_RATE)
    min_vol_stats = portfolio_stats(min_vol_weights, mu, cov, RISK_FREE_RATE)
    rolling = rolling_statistics(asset_returns, 252 * 5, TRADING_DAYS)

    print_weights("Starting Portfolio Weights", TICKERS, starting_weights)
    print("\nStarting Portfolio Stats")
    print(starting_stats)

    if IMPLEMENTATION_LAYER:
        unrealized_gains_rates = (
            np.array([DEFAULT_UNREALIZED_GAIN_RATE] * len(TICKERS))
            if APPLY_TAX_EFFECTS
            else None
        )
        implementation_report = net_expected_return_after_all_costs(
            gross_expected_return=max_sharpe_stats["expected_return"],
            current_weights=starting_weights,
            target_weights=max_sharpe_weights,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            unrealized_gains_rates=unrealized_gains_rates,
            short_term_tax_rate=SHORT_TERM_TAX_RATE,
            long_term_tax_rate=LONG_TERM_TAX_RATE,
            long_term_fraction=LONG_TERM_FRACTION,
        )

        print("\n" + "=" * 60)
        print("IMPLEMENTATION-AWARE RESEARCH COST ANALYSIS")
        print("=" * 60)
        for key, value in implementation_report.items():
            print(f"{key}: {value:.4f}")

    print_weights("Max Sharpe Weights", TICKERS, max_sharpe_weights)
    print("\nMax Sharpe Stats")
    print(max_sharpe_stats)

    print_weights("Min Vol Weights", TICKERS, min_vol_weights)
    print("\nMin Vol Stats")
    print(min_vol_stats)

    starting_series = build_portfolio_return_series(asset_returns, starting_weights, TICKERS)
    max_sharpe_series = build_portfolio_return_series(asset_returns, max_sharpe_weights, TICKERS)
    min_vol_series = build_portfolio_return_series(asset_returns, min_vol_weights, TICKERS)

    print("\nDetailed Summary: Starting Portfolio")
    print(summary_table(starting_series, benchmark_returns, RISK_FREE_RATE, TRADING_DAYS))

    print("\n" + "=" * 60)
    print("FAMA-FRENCH 5 FACTOR REGRESSION")
    print("=" * 60)
    ff5_model = fama_french_regression(starting_series, model="ff5")
    print(ff5_model.summary())

    print("\nFAMA-FRENCH 5 FACTOR TABLE")
    print(factor_summary_table(ff5_model).round(4))

    wf_series = None
    wf_weights = None
    wf_diag = None

    if RUN_WALKFORWARD_BACKTEST:
        print("\n" + "=" * 60)
        print("WALK-FORWARD OUT-OF-SAMPLE BACKTEST")
        print("=" * 60)

        wf_series, wf_weights, wf_diag = rolling_black_litterman_backtest(
            asset_returns=asset_returns,
            tickers=TICKERS,
            trading_days=TRADING_DAYS,
            risk_free_rate=RISK_FREE_RATE,
            weight_bounds=WEIGHT_BOUNDS,
            fixed_weights=FIXED_WEIGHTS,
            min_weights=MIN_WEIGHTS,
            window_days=WF_WINDOW_DAYS,
            rebalance_freq=WF_REBALANCE_FREQ,
            use_black_litterman=USE_BLACK_LITTERMAN,
            bl_market_weights=BL_MARKET_WEIGHTS,
            bl_risk_aversion=BL_RISK_AVERSION,
            bl_tau=BL_TAU,
            bl_absolute_views=BL_ABSOLUTE_VIEWS,
            use_cov_shrinkage=USE_COV_SHRINKAGE,
            cov_shrinkage_method=COV_SHRINKAGE_METHOD,
            implementation_layer=IMPLEMENTATION_LAYER,
            turnover_penalty_lambda=TURNOVER_PENALTY_LAMBDA,
            starting_weights=starting_weights,
        )

        print("\nWalk-forward summary:")
        print(
            summary_table(
                wf_series,
                benchmark_returns.loc[wf_series.index],
                RISK_FREE_RATE,
                TRADING_DAYS,
            )
        )

        if wf_weights is not None and not wf_weights.empty:
            latest_weights = wf_weights.tail(1).T.rename(
                columns={wf_weights.index[-1]: "weight"}
            )
            print("\nLatest walk-forward weights:")
            print(latest_weights.round(4))

        if wf_diag is not None and not wf_diag.empty:
            print("\nRecent rebalance diagnostics:")
            print(wf_diag.tail().round(4))

    if GENERATE_QUANTSTATS_REPORT:
        export_quantstats_report(
            portfolio_returns=starting_series,
            benchmark_returns=benchmark_returns,
            output_path=QUANTSTATS_OUTPUT,
        )
        print(f"\nQuantStats report saved to: {QUANTSTATS_OUTPUT}")

        if RUN_WALKFORWARD_BACKTEST and wf_series is not None:
            export_quantstats_report(
                portfolio_returns=wf_series,
                benchmark_returns=benchmark_returns.loc[wf_series.index],
                output_path=WALKFORWARD_QUANTSTATS_OUTPUT,
            )
            print(
                "\nWalk-forward QuantStats report saved to: "
                f"{WALKFORWARD_QUANTSTATS_OUTPUT}"
            )

    mc_paths = simulate_portfolio_paths(
        weights=starting_weights,
        mu=mu,
        cov=cov,
        years=MC_YEARS,
        n_sims=MC_N_SIMS,
        trading_days=TRADING_DAYS,
        initial_value=MC_INITIAL_VALUE,
        seed=MC_SEED,
    )
    mc_terminal = terminal_value_stats(mc_paths)
    mc_drawdowns = drawdown_stats(mc_paths)

    if RUN_WALKFORWARD_BACKTEST and wf_series is not None:
        print("\n" + "=" * 60)
        print("WALK-FORWARD FAMA-FRENCH 5 FACTOR REGRESSION")
        print("=" * 60)

        wf_ff5_model = fama_french_regression(wf_series, model="ff5")
        print(wf_ff5_model.summary())

        print("\nWALK-FORWARD FAMA-FRENCH 5 FACTOR TABLE")
        print(factor_summary_table(wf_ff5_model).round(4))

    print("\n" + "=" * 60)
    print("RISK CONTRIBUTION")
    print("=" * 60)
    risk_df = risk_contribution(
        tickers=TICKERS,
        weights=max_sharpe_weights,
        cov=cov,
    )
    print(risk_df.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("MONTE CARLO RISK")
    print("=" * 60)
    mc_report = monte_carlo_risk_report(
        terminal_stats=mc_terminal,
        drawdown_stats=mc_drawdowns,
    )
    print(mc_report.round(4))

    print("\nMonte Carlo Terminal Value Stats")
    for key, value in mc_terminal.items():
        print(f"{key}: {value:.4f}")

    print("\nMonte Carlo Drawdown Stats")
    for key, value in mc_drawdowns.items():
        print(f"{key}: {value:.4f}")

    growth_df = pd.DataFrame(
        {
            "Starting Portfolio": cumulative_growth(starting_series),
            "Max Sharpe": cumulative_growth(max_sharpe_series),
            "Min Vol": cumulative_growth(min_vol_series),
            f"{BENCHMARK} Benchmark": cumulative_growth(benchmark_returns),
        }
    )

    if RUN_WALKFORWARD_BACKTEST and wf_series is not None:
        growth_df["Walk-Forward OOS"] = cumulative_growth(wf_series)

    frontier_vol, frontier_ret, _ = efficient_frontier(
        mu,
        cov,
        bounds=WEIGHT_BOUNDS,
    )

    benchmark_ret = benchmark_returns.mean() * TRADING_DAYS
    benchmark_vol = benchmark_returns.std() * (TRADING_DAYS ** 0.5)

    max_sharpe_slope = (
        (max_sharpe_stats["expected_return"] - RISK_FREE_RATE) / max_sharpe_stats["volatility"]
    )
    cml_x = pd.Series([0.0, max(frontier_vol.max(), benchmark_vol) * 1.1])
    cml_y = RISK_FREE_RATE + max_sharpe_slope * cml_x

    plt.figure(figsize=(10, 6))
    for column in growth_df.columns:
        plt.plot(growth_df.index, growth_df[column], label=column)

    plt.title("Cumulative Growth")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(frontier_vol, frontier_ret, label="Efficient Frontier")
    plt.plot(cml_x, cml_y, linestyle="--", label="Capital Market Line")
    plt.scatter(
        starting_stats["volatility"],
        starting_stats["expected_return"],
        marker="o",
        label="Starting Portfolio",
        s=120,
    )
    plt.scatter(
        max_sharpe_stats["volatility"],
        max_sharpe_stats["expected_return"],
        marker="*",
        label="Max Sharpe",
        s=200,
    )
    plt.scatter(
        min_vol_stats["volatility"],
        min_vol_stats["expected_return"],
        marker="D",
        label="Min Vol",
        s=120,
    )
    plt.scatter(
        benchmark_vol,
        benchmark_ret,
        marker="X",
        label=f"{BENCHMARK} Benchmark",
        s=140,
    )
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(rolling.index, rolling["return"], label="Rolling Return")
    plt.plot(rolling.index, rolling["vol"], label="Rolling Volatility")
    plt.title("Rolling Portfolio Environment (5 Year Window)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
