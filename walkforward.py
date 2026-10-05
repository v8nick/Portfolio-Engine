import pandas as pd
import numpy as np

from mpt import (
    annualize_mean_cov,
    optimize_max_sharpe_constrained,
)
from black_litterman import black_litterman_posterior, resolve_strategic_prior_weights
from data import get_risk_free_rate
from implementation import apply_implementation_layer


def get_rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """
    Get rebalance dates that actually exist in the return index.
    Monthly means last available trading day of each month.
    """
    if freq.upper() not in {"M", "ME"}:
        raise ValueError("Only monthly rebalancing ('ME') is currently supported.")

    s = pd.Series(index=index, data=index)
    dates = pd.DatetimeIndex(s.resample("ME").last().dropna().values)
    return pd.DatetimeIndex(dates)


def portfolio_return_series_from_weight_history(
    asset_returns: pd.DataFrame,
    weight_history: pd.DataFrame,
) -> pd.Series:
    """
    Decision-date weights become effective next observation.
    This is a constant-weight daily-rebalanced, gross-return baseline.
    """
    aligned_weights = weight_history.reindex(asset_returns.index).ffill().shift(1)
    aligned_weights = aligned_weights.dropna(how="all")

    aligned_returns = asset_returns.loc[aligned_weights.index]
    portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
    portfolio_returns.name = "walkforward_portfolio"
    return portfolio_returns


def rolling_black_litterman_backtest(
    asset_returns: pd.DataFrame,
    tickers: list[str],
    trading_days: int,
    risk_free_rate: float,
    weight_bounds: tuple[float, float],
    fixed_weights: dict[str, float],
    min_weights: dict[str, float],
    window_days: int = 252 * 3,
    rebalance_freq: str = "M",
    use_black_litterman: bool = True,
    bl_market_weights: dict[str, float] | None = None,
    bl_risk_aversion: float = 2.5,
    bl_tau: float = 0.05,
    bl_absolute_views: dict[str, tuple[float, float]] | None = None,
    use_cov_shrinkage: bool = False,
    cov_shrinkage_method: str = "ledoit_wolf",
    implementation_layer: bool = False,
    turnover_penalty_lambda: float = 0.0,
    starting_weights: np.ndarray | None = None,
    bl_relative_views: list[tuple[str, str, float, float]] | None = None,
    risk_free_yields: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward out-of-sample backtest:
    - estimate on trailing window
    - optimize on rebalance date
    - hold through next rebalance period

    Returns
    -------
    oos_returns : pd.Series
        Out-of-sample daily portfolio returns
    weight_history : pd.DataFrame
        Rebalance-date weights
    diagnostics : pd.DataFrame
        Rebalance-date diagnostics
    """
    if not asset_returns.index.is_unique or not asset_returns.index.is_monotonic_increasing:
        raise ValueError("Returns must have a unique, increasing date index.")
    if window_days < 2:
        raise ValueError("window_days must be at least two.")
    asset_returns = asset_returns[tickers].dropna()

    rebalance_dates = get_rebalance_dates(asset_returns.index, freq=rebalance_freq)

    eligible_dates = []
    for dt in rebalance_dates:
        loc = asset_returns.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        if window_days - 1 <= loc < len(asset_returns) - 1:
            eligible_dates.append(dt)

    eligible_dates = pd.DatetimeIndex(eligible_dates)

    weight_history = pd.DataFrame(index=eligible_dates, columns=tickers, dtype=float)
    diagnostics = []

    market_weights = resolve_strategic_prior_weights(tickers, bl_market_weights) if use_black_litterman else None

    current_weights = (
        starting_weights.copy()
        if starting_weights is not None
        else np.repeat(1 / len(tickers), len(tickers))
    )

    for dt in eligible_dates:
        end_loc = asset_returns.index.get_loc(dt)
        if isinstance(end_loc, slice):
            end_loc = end_loc.stop - 1

        window_returns = asset_returns.iloc[end_loc - window_days + 1 : end_loc + 1]
        rf = (get_risk_free_rate(as_of=dt, yields=risk_free_yields)
              if risk_free_yields is not None else risk_free_rate)

        hist_mu, cov = annualize_mean_cov(
            window_returns,
            trading_days=trading_days,
            use_shrinkage=use_cov_shrinkage,
            shrinkage_method=cov_shrinkage_method,
        )

        if use_black_litterman:
            mu, pi = black_litterman_posterior(
                cov=cov,
                market_weights=market_weights,
                risk_aversion=bl_risk_aversion,
                tau=bl_tau,
                absolute_views=bl_absolute_views,
                relative_views=bl_relative_views,
                risk_free_rate=rf,
            )
        else:
            mu = hist_mu
            pi = hist_mu.copy()

        mu_for_opt = apply_implementation_layer(
            mu=mu,
            current_weights=current_weights,
            tickers=tickers,
            implementation_layer=implementation_layer,
            turnover_penalty_lambda=turnover_penalty_lambda,
        )

        target_weights = optimize_max_sharpe_constrained(
            mu=mu_for_opt,
            cov=cov,
            rf=rf,
            tickers=tickers,
            bounds=weight_bounds,
            fixed_weights=fixed_weights,
            min_weights=min_weights,
        )

        weight_history.loc[dt] = target_weights

        exp_ret = float(target_weights @ mu.values)
        vol = float(np.sqrt(target_weights @ cov.values @ target_weights))
        sharpe = (exp_ret - rf) / vol if vol > 0 else np.nan
        turnover = float(np.sum(np.abs(target_weights - current_weights)))

        diagnostics.append(
            {
                "rebalance_date": dt,
                "risk_free_rate": rf,
                "estimation_end": window_returns.index[-1],
                "effective_date": asset_returns.index[end_loc + 1] if end_loc + 1 < len(asset_returns) else pd.NaT,
                "expected_return": exp_ret,
                "volatility": vol,
                "sharpe": sharpe,
                "turnover": turnover,
            }
        )

        current_weights = target_weights.copy()

    diagnostics = pd.DataFrame(diagnostics, columns=["rebalance_date", "risk_free_rate", "estimation_end", "effective_date", "expected_return", "volatility", "sharpe", "turnover"]).set_index("rebalance_date")

    oos_returns = portfolio_return_series_from_weight_history(asset_returns, weight_history)

    if len(weight_history.index) > 0:
        oos_returns = oos_returns.loc[weight_history.index.min():]

    return oos_returns, weight_history, diagnostics