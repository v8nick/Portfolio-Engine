# main.py

from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from frontier import efficient_frontier
from rollingfront import rolling_statistics
from black_litterman import black_litterman_posterior

from dashboard import (
    current_portfolio_performance,
    optimized_portfolio_summary,
    suggested_weight_changes,
    risk_contribution,
    monte_carlo_risk_report,
)
from config import (
    TICKERS,
    BENCHMARK,
    BASE_WEIGHTS,
    START_DATE,
    END_DATE,
    TRADING_DAYS,
    RISK_FREE_RATE,
    WEIGHT_BOUNDS,
    MC_YEARS,
    MC_N_SIMS,
    MC_INITIAL_VALUE,
    MC_SEED,
    USE_BLACK_LITTERMAN,
    BL_MARKET_WEIGHTS,
    BL_RISK_AVERSION,
    BL_TAU,
    BL_ABSOLUTE_VIEWS,
    BL_RELATIVE_VIEWS,
    FIXED_WEIGHTS,
    MIN_WEIGHTS,
)
from data import download_prices, compute_returns
from mpt import (
    annualize_mean_cov,
    optimize_max_sharpe,
    optimize_min_vol,
    portfolio_stats,
    weights_dict_to_array,
)
from report import (
    build_portfolio_return_series,
    cumulative_growth,
    summary_table,
)
from simulation import (
    simulate_portfolio_paths,
    terminal_value_stats,
    drawdown_stats,
)
from mpt import (
    annualize_mean_cov,
    optimize_max_sharpe,
    optimize_min_vol,
    optimize_max_sharpe_constrained,
    portfolio_stats,
    weights_dict_to_array,
)

def print_weights(title: str, tickers: list[str], weights) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for t, w in zip(tickers, weights):
        print(f"{t:>5}: {w:6.2%}")


def main() -> None:
    all_tickers = TICKERS + [BENCHMARK]

    prices = download_prices(all_tickers, START_DATE, END_DATE)
    returns = compute_returns(prices)

    asset_returns = returns[TICKERS]
    benchmark_returns = returns[BENCHMARK]

    hist_mu, cov = annualize_mean_cov(asset_returns, TRADING_DAYS)

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
    mu_compare = pd.DataFrame({
        "hist_mu": hist_mu,
        "pi": pi if USE_BLACK_LITTERMAN else hist_mu,
        "bl_mu": mu if USE_BLACK_LITTERMAN else hist_mu,
    })

    print("\nReturn Estimate Comparison")
    print(mu_compare.round(4))

    base_weights = weights_dict_to_array(BASE_WEIGHTS, TICKERS)

    base_stats = portfolio_stats(base_weights, mu, cov, RISK_FREE_RATE)
    max_sharpe_weights = optimize_max_sharpe_constrained(
        mu=mu,
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
    rolling = rolling_statistics(asset_returns, 252*5, TRADING_DAYS)
    print_weights("Base Portfolio Weights", TICKERS, base_weights)
    print("\nBase Portfolio Stats")
    print(base_stats)

    print_weights("Max Sharpe Weights", TICKERS, max_sharpe_weights)
    print("Max Sharpe Stats")
    print(portfolio_stats(max_sharpe_weights, mu, cov, RISK_FREE_RATE))

    print_weights("Min Vol Weights", TICKERS, min_vol_weights)
    print("Min Vol Stats")
    print(portfolio_stats(min_vol_weights, mu, cov, RISK_FREE_RATE))

    base_series = build_portfolio_return_series(asset_returns, base_weights, TICKERS)
    max_sharpe_series = build_portfolio_return_series(asset_returns, max_sharpe_weights, TICKERS)
    min_vol_series = build_portfolio_return_series(asset_returns, min_vol_weights, TICKERS)

    print("\nDetailed Summary: Base Portfolio")
    print(summary_table(base_series, benchmark_returns, RISK_FREE_RATE, TRADING_DAYS))
    mc_paths = simulate_portfolio_paths(
        weights=base_weights,
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
    print("\n" + "=" * 60)
    print("CURRENT PORTFOLIO PERFORMANCE")
    print("=" * 60)
    current_perf = current_portfolio_performance(
        portfolio_returns=base_series,
        benchmark_returns=benchmark_returns,
        rf=RISK_FREE_RATE,
        trading_days=TRADING_DAYS,
    )
    print(current_perf.round(4))

    print("\n" + "=" * 60)
    print("OPTIMIZED PORTFOLIO")
    print("=" * 60)
    opt_weights_df, opt_stats_df = optimized_portfolio_summary(
        tickers=TICKERS,
        weights=max_sharpe_weights,
        expected_return=max_sharpe_stats["expected_return"],
        volatility=max_sharpe_stats["volatility"],
        sharpe=max_sharpe_stats["sharpe"],
    )
    print(opt_stats_df.round(4))
    print(opt_weights_df.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("SUGGESTED WEIGHT CHANGES")
    print("=" * 60)
    changes_df = suggested_weight_changes(
        tickers=TICKERS,
        current_weights=base_weights,
        optimized_weights=max_sharpe_weights,
    )
    print(changes_df.round(4).to_string(index=False))

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
    for k, v in mc_terminal.items():
        print(f"{k}: {v:.4f}")

    print("\nMonte Carlo Drawdown Stats")
    for k, v in mc_drawdowns.items():
        print(f"{k}: {v:.4f}")
        
    growth_df = pd.DataFrame({
        "Base Portfolio": cumulative_growth(base_series),
        "Max Sharpe": cumulative_growth(max_sharpe_series),
        "Min Vol": cumulative_growth(min_vol_series),
        BENCHMARK: cumulative_growth(benchmark_returns),
    })
    frontier_vol, frontier_ret, _ = efficient_frontier(
    mu,
    cov,
    bounds=WEIGHT_BOUNDS
)

    base_ret = base_stats["expected_return"]
    base_vol = base_stats["volatility"]

    max_sharpe_stats = portfolio_stats(max_sharpe_weights, mu, cov, RISK_FREE_RATE)
    max_sharpe_ret = max_sharpe_stats["expected_return"]
    max_sharpe_vol = max_sharpe_stats["volatility"]

    min_vol_stats = portfolio_stats(min_vol_weights, mu, cov, RISK_FREE_RATE)
    min_vol_ret = min_vol_stats["expected_return"]
    min_vol_vol = min_vol_stats["volatility"]

    spy_ret = benchmark_returns.mean() * TRADING_DAYS
    spy_vol = benchmark_returns.std() * (TRADING_DAYS ** 0.5)

    qqq_ret = mu["QQQ"]
    qqq_vol = asset_returns["QQQ"].std() * (TRADING_DAYS ** 0.5)
    #cap mkt chart
    qqq_ret = float(mu["QQQ"])
    qqq_vol = asset_returns["QQQ"].std() * (TRADING_DAYS ** 0.5)

    spy_hist_ret = benchmark_returns.mean() * TRADING_DAYS
    spy_ret = 0.5 * spy_hist_ret + 0.5 * RISK_FREE_RATE
    spy_vol = benchmark_returns.std() * (TRADING_DAYS ** 0.5)

    max_sharpe_slope = (max_sharpe_ret - RISK_FREE_RATE) / max_sharpe_vol

    cml_x = pd.Series([
        0.0,
        max(frontier_vol.max(), spy_vol, qqq_vol) * 1.1
    ])

    cml_y = RISK_FREE_RATE + max_sharpe_slope * cml_x
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_vol, frontier_ret, label="Efficient Frontier")
    plt.plot(cml_x, cml_y, linestyle="--", label="Capital Market Line") 
    plt.scatter(base_vol, base_ret, marker="o", label="Base Portfolio", s=120)
    plt.scatter(max_sharpe_vol, max_sharpe_ret, marker="*", label="Max Sharpe", s=200)
    plt.scatter(min_vol_vol, min_vol_ret, marker="D", label="Min Vol", s=120)
    plt.scatter(spy_vol, spy_ret, marker="X", label="SPY", s=140)
    plt.scatter(qqq_vol, qqq_ret, marker="^", label="QQQ", s=140)

    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10,6))

    plt.plot(rolling.index, rolling["return"], label="Rolling Return")
    plt.plot(rolling.index, rolling["vol"], label="Rolling Volatility")

    plt.title("Rolling Portfolio Environment (5 Year Window)")
    plt.legend()
    plt.grid(True)

    plt.show() 
if __name__ == "__main__":
    main()