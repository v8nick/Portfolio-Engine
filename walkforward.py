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
    aliases = {"M": "ME", "ME": "ME", "monthly": "ME", "Q": "QE", "QE": "QE", "quarterly": "QE"}
    freq = freq.lower() if freq.lower() in ("monthly", "quarterly") else freq.upper()
    if freq not in aliases:
        raise ValueError("Use monthly/ME or quarterly/QE rebalancing.")

    s = pd.Series(index=index, data=index)
    dates = pd.DatetimeIndex(s.resample(aliases[freq]).last().dropna().values)
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

# New research API. The legacy constant-weight baseline above remains available.
RESEARCH_STRATEGIES = ('current', 'strategic', 'equal_weight', 'spy', 'balanced_60_40',
                       'bl_max_sharpe', 'minimum_variance', 'risk_parity', 'cvar')


def research_price_calendar(prices, assets):
    """Observed portfolio sessions; exclude rows containing only unrelated macro quotes.

    Keep a session if any requested asset has a quote. Individual missing quotes
    remain missing and fail coverage checks; never forward-fill prices.
    """
    from regimes import _frame
    frame = _frame(prices)
    available = [asset for asset in dict.fromkeys(assets) if asset in frame]
    return frame.loc[frame[available].notna().any(axis=1)] if available else frame.iloc[:0]


def backtest_history_start(prices, assets, lookback=252):
    """First date with a complete trailing estimation window for this universe."""
    from data import compute_returns
    frame = research_price_calendar(prices, assets)
    if frame.empty or any(asset not in frame for asset in assets):
        return None
    returns = compute_returns(frame[list(dict.fromkeys(assets))])
    complete = (np.isfinite(returns) & (returns > -1)).all(axis=1)
    eligible = complete.rolling(int(lookback)).sum().eq(lookback)
    return eligible.index[eligible][0] if eligible.any() else None


def walk_forward_backtest(prices, yields, current_weights, strategic_allocation, *,
                          strategies=('strategic', 'bl_max_sharpe'), benchmark='spy',
                          start=None, end=None, lookback=756, rebalance_frequency='monthly',
                          return_model='bl', transaction_cost_bps=10., slippage_bps=5.,
                          costs_enabled=True, no_trade_threshold=.01, risk_free_rate=None,
                          max_satellite_allocation=1., max_individual_satellite_weight=1.,
                          view_history=(), regime_model_options=None):
    """Point-in-time decisions at close; trade/cost at next observation's beginning.

    Shares drift between trades. Complete OOS asset coverage is mandatory; no
    forward-filling prices or dropping missing trading observations. Fixed current,
    equal-weight and external benchmarks are explicitly unconstrained references.
    Optimized strategies and strategic policy enforce all supplied constraints.
    An optimizer failure retains holdings and is recorded, never replaced by a
    different optimizer. Today's holdings/policy are static hypothetical inputs,
    not a reconstruction of the user's historical account.
    """
    from config import shared
    from data import compute_returns, resolve_risk_free_rate
    from regimes import _frame, build_regime_history
    from portfolio_decision import resolve_allocation_policy
    from expected_returns import estimate_expected_returns, RETURN_MODELS
    from mpt import optimize_min_vol, optimize_risk_parity, optimize_min_cvar, validate_allocation_weights
    from implementation import compute_turnover, implementation_cost_rate
    from report import backtest_metrics, rolling_backtest_metrics, regime_backtest_metrics
    if return_model not in RETURN_MODELS or lookback < 2 or int(lookback) != lookback:
        raise ValueError('Choose a known return model and integer lookback >= 2.')
    if not strategies or any(s not in RESEARCH_STRATEGIES for s in (*strategies, benchmark)):
        raise ValueError('Choose known strategies and benchmark.')
    if not np.isfinite([transaction_cost_bps, slippage_bps, no_trade_threshold]).all() or min(transaction_cost_bps, slippage_bps, no_trade_threshold) < 0:
        raise ValueError('Costs and no-trade threshold must be finite and nonnegative.')
    if (transaction_cost_bps + slippage_bps) * 2 >= 10000:
        raise ValueError('Trading costs must be below portfolio value.')
    policy, tickers, strategic, bounds, caps = resolve_allocation_policy(
        strategic_allocation, current_weights, max_satellite_allocation, max_individual_satellite_weight)
    initial = np.array([current_weights.get(t, 0.) for t in tickers])
    validate_allocation_weights(initial, tickers, (0, 1))
    prices, yields = _frame(prices, end), _frame(yields, end)
    if prices.empty or start is not None and pd.Timestamp(start) >= prices.index[-1]:
        raise ValueError('Backtest dates have no out-of-sample interval.')
    if (prices <= 0).any().any():
        raise ValueError('Prices must be positive.')
    requested = tuple(dict.fromkeys([*strategies, benchmark]))
    calendar_assets = list(dict.fromkeys(asset for strategy in requested for asset in
        (['SPY'] if strategy == 'spy' else ['SPY', 'IEF'] if strategy == 'balanced_60_40' else tickers)))
    trading_prices = research_price_calendar(prices, calendar_assets)
    # Macro series can quote on exchange holidays. Filter before pct_change so
    # an unrelated quote cannot invalidate both a holiday and the next session.
    returns = compute_returns(trading_prices)
    if returns.empty:
        raise ValueError('No observed trading history for the requested assets.')
    history = build_regime_history(prices, yields, feature_options={'as_of': end}, **(regime_model_options or {}))
    # Use the prior portfolio session's causal classification.
    previous_regimes = history['primary_regime'].reindex(trading_prices.index).shift(1)
    calendar = returns.index
    decision_dates = get_rebalance_dates(calendar, rebalance_frequency)
    decision_dates = [d for d in decision_dates if (start is None or d >= pd.Timestamp(start))
                      and calendar.get_loc(d) >= lookback - 1 and d < calendar[-1]]
    if not decision_dates:
        raise ValueError('Insufficient history for an eligible rebalance plus a later observation.')
    first = decision_dates[0]
    oos_index = calendar[calendar > first]
    rf_series = pd.Series({dt: resolve_risk_free_rate(as_of=calendar[calendar.get_loc(dt)-1],
        yields=yields, override=risk_free_rate)['rate'] for dt in oos_index})
    outputs, failures = {}, {}
    for strategy in dict.fromkeys([*strategies, benchmark]):
        assets = ['SPY'] if strategy == 'spy' else ['SPY', 'IEF'] if strategy == 'balanced_60_40' else tickers
        if any(t not in returns for t in assets):
            failures[strategy] = 'Required assets unavailable: ' + ', '.join(t for t in assets if t not in returns)
            continue
        panel = returns[assets]
        observed = panel.loc[first:]
        invalid = ~np.isfinite(observed) | (observed <= -1)
        if invalid.any().any():
            affected = ', '.join(invalid.columns[invalid.any()])
            first_invalid = invalid.index[invalid.any(axis=1)][0].date()
            failures[strategy] = (f'Incomplete or invalid out-of-sample prices for {affected} '
                f'(first affected observation: {first_invalid}). Choose a period with complete history '
                'or refresh market data. No asset histories substituted.')
            continue
        fixed = {'current': initial, 'strategic': strategic, 'equal_weight': np.repeat(1/len(tickers), len(tickers)),
                 'spy': np.array([1.]), 'balanced_60_40': np.array([.6, .4])}.get(strategy)
        weights = fixed.copy() if fixed is not None else initial.copy()
        rows, decisions, daily, net_returns, gross_returns = [], {}, {}, {}, {}
        nav = 10000.
        pending = None
        for dt in calendar[calendar >= first]:
            if dt > first:
                cost, turnover = 0., 0.
                if pending is not None:
                    weights, record = pending
                    cost, turnover = record['cost_rate'], record['turnover']
                    record['cost_amount_per_10000_start'] = nav * cost
                    pending = None
                daily[dt] = weights.copy()
                values = panel.loc[dt].to_numpy()
                gross = float(weights @ values)
                net = (1 - cost) * (1 + gross) - 1
                net_returns[dt], gross_returns[dt] = net, gross
                nav *= 1 + net
                weights = weights * (1 + values) / (1 + gross)
            if dt not in decision_dates:
                continue
            sample = panel.loc[:dt].dropna().tail(lookback)
            record = {'decision_date': dt, 'effective_date': calendar[calendar.get_loc(dt)+1],
                      'estimation_end': dt, 'status': 'held', 'turnover': 0., 'cost_rate': 0.,
                      'cost_amount_per_10000_start': 0., 'regime': history.loc[:dt, 'primary_regime'].iloc[-1],
                      'expected_return_source': 'fixed allocation reference', 'error': ''}
            try:
                if len(sample) < lookback:
                    raise ValueError('Insufficient common trailing estimation history.')
                if fixed is not None:
                    target = fixed.copy()
                    record['constraints'] = 'supplied policy' if strategy == 'strategic' else 'unconstrained reference allocation'
                else:
                    rf_info = resolve_risk_free_rate(as_of=dt, yields=yields, override=risk_free_rate)
                    _, covariance = annualize_mean_cov(sample, shared.TRADING_DAYS,
                        shared.USE_COV_SHRINKAGE, shared.COV_SHRINKAGE_METHOD)
                    mu, source = estimate_expected_returns(sample, covariance, strategic, model=return_model,
                        risk_free_rate=rf_info['rate'], as_of=dt, prices=prices.loc[:dt], yields=yields.loc[:dt],
                        regime_history=history.loc[:dt], view_history=view_history, regime_model_options=regime_model_options)
                    record.update(expected_return_source=source['source'], estimation_metadata=source,
                                  risk_free=rf_info, covariance_method=shared.COV_SHRINKAGE_METHOD,
                                  constraints='supplied policy')
                    if strategy == 'bl_max_sharpe':
                        target = optimize_max_sharpe_constrained(mu, covariance, rf_info['rate'], assets,
                            bounds=bounds, max_group_weights=caps)
                    elif strategy == 'minimum_variance':
                        target = optimize_min_vol(mu, covariance, bounds=bounds, tickers=assets, max_group_weights=caps)
                    elif strategy == 'risk_parity':
                        target = optimize_risk_parity(covariance, bounds, strategic, caps)
                    else:
                        target = optimize_min_cvar(mu, sample, bounds, max_group_weights=caps)
                constrained = fixed is None or strategy == 'strategic'
                validate_allocation_weights(target, assets, bounds if constrained else (0, 1), caps if constrained else None)
                repair = False
                if constrained:
                    try:
                        validate_allocation_weights(weights, assets, bounds, caps)
                    except ValueError:
                        repair = True
                if repair or np.max(np.abs(target - weights)) > no_trade_threshold:
                    turnover = compute_turnover(weights, target)
                    cost = implementation_cost_rate(weights, target, transaction_cost_bps, slippage_bps) if costs_enabled else 0.
                    record.update(status='rebalanced', turnover=turnover, cost_rate=cost)
                    pending = (target.copy(), record)
                decisions[dt] = (target if pending is not None else weights).copy()
            except Exception as exc:
                # Retain the requested strategy identity and show every failure.
                record.update(status='optimizer/data failure; holdings retained', error=f'{type(exc).__name__}: {exc}')
                decisions[dt] = weights.copy()
            rows.append(record)
        diagnostics = pd.DataFrame(rows).set_index('decision_date')
        series = pd.Series(net_returns, name=strategy)
        outputs[strategy] = {'returns': series, 'gross_returns': pd.Series(gross_returns),
            'weight_history': pd.DataFrame.from_dict(decisions, orient='index', columns=assets),
            'daily_weights': pd.DataFrame.from_dict(daily, orient='index', columns=assets),
            'diagnostics': diagnostics, 'rolling': rolling_backtest_metrics(series, rf_series),
            'regime_performance': regime_backtest_metrics(series, previous_regimes, rf_series)}
    bench = outputs.get(benchmark, {}).get('returns')
    metrics = {name: backtest_metrics(item['returns'], bench, rf_series, item['diagnostics']) for name, item in outputs.items()}
    # Preserve the dated schema even when every strategy is unavailable, so the
    # caller can display coverage failures instead of crashing in resample().
    combined = pd.DataFrame({name: item['returns'] for name, item in outputs.items()}, index=oos_index)
    return {'strategies': outputs, 'returns': combined, 'metrics': pd.DataFrame(metrics).T,
            'failures': failures, 'benchmark': benchmark,
            'calendar_returns': ((1 + combined).resample('YE').prod() - 1),
            'metadata': {'expected_return_model': return_model, 'rebalance_frequency': rebalance_frequency,
                'lookback': lookback, 'first_decision': first.isoformat(), 'risk_free': 'Prior-observation Treasury or explicit fixed assumption',
                'execution': 'Decision after close; new weights and costs effective next observation. Holdings drift between trades.',
                'calendar': 'Observed sessions with at least one requested asset quote; macro-only dates excluded before returns. Individual asset gaps are not filled.',
                'excluded_macro_only_rows': len(prices) - len(trading_prices),
                'costs': 'Two-sided turnover × (transaction + slippage) bps; deducted multiplicatively on trade date; no taxes.',
                'metrics': '252 sessions/year; arithmetic excess Sharpe; CAGR geometric; tail risk daily; costs sum NAV rates and recorded dollar amounts.',
                'regimes': 'Previous-session causal regime; conditional drawdowns concatenate selected days and are descriptive, not a tradable history.',
                'limitations': 'Fixed present-day universe/policy; survivorship and adjusted-price/vintage limitations. No historical account reconstruction. No parameter tuning.'}}
