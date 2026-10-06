# report.py

from __future__ import annotations
from pathlib import Path

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
    peak = wealth.cummax().clip(lower=1.0)
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


def export_quantstats_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    output_path: str = "quantstats_report.html",
) -> None:
    import quantstats as qs

    port = portfolio_returns.dropna().copy()
    port.index = pd.to_datetime(port.index)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if benchmark_returns is not None:
        bench = benchmark_returns.dropna().copy()
        bench.index = pd.to_datetime(bench.index)
        qs.reports.html(
            port,
            benchmark=bench,
            output=str(output),
            title="Portfolio QuantStats Report",
        )
    else:
        qs.reports.html(
            port,
            output=str(output),
            title="Portfolio QuantStats Report",
        )


def empirical_var_cvar(returns: pd.Series, confidence: float = .95) -> dict:
    """Empirical daily losses, with exact fractional tail mass for expected shortfall.

    Positive=loss, negative=gain; no Gaussian assumption. VaR is the empirical
    inverse-CDF order statistic. CVaR averages the worst (1-confidence) fraction,
    including a fractional boundary observation when necessary.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('Tail confidence must be strictly between zero and one.')
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {'var': np.nan, 'cvar': np.nan, 'confidence': confidence, 'frequency': 'daily', 'unit': 'loss fraction'}
    losses = np.sort(-clean.to_numpy(dtype=float))
    mass = len(losses) * (1 - confidence)
    whole = int(np.floor(mass))
    fraction = mass - whole
    ordered = losses[::-1]
    cvar = (ordered[:whole].sum() + (fraction * ordered[whole] if fraction > 0 else 0)) / mass
    return {'var': float(losses[int(np.ceil(confidence * len(losses))) - 1]),
            'cvar': float(cvar), 'confidence': confidence, 'frequency': 'daily', 'unit': 'loss fraction'}


def historical_tail_risk(returns: pd.Series, confidence: float = .95,
                         trading_days: int = 252) -> dict:
    clean = returns.dropna()
    return {'max_drawdown': max_drawdown(clean) if len(clean) else np.nan,
            'annualized_volatility': annualized_volatility(clean, trading_days),
            'downside_deviation': float(np.sqrt(np.mean(np.minimum(clean, 0) ** 2)) * np.sqrt(trading_days)) if len(clean) else np.nan,
            'downside_target': 0., 'observations': len(clean),
            **empirical_var_cvar(clean, confidence)}


# Research analytics use arithmetic excess-return Sharpe; legacy reports remain intact.
def backtest_metrics(returns, benchmark=None, risk_free=0., diagnostics=None, trading_days=252):
    """Daily simple returns; CAGR geometric, vol/TE sqrt(252), alpha arithmetic.

    RF is annual decimal (scalar or point-in-time series), converted by /252.
    Sortino uses all observations with negative excess returns squared; daily
    empirical VaR/CVaR at 95%. Turnover is two-sided notional, costs are NAV rates.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return {}
    rf = pd.Series(risk_free, index=r.index) if np.isscalar(risk_free) else risk_free.reindex(r.index)
    if rf.isna().any() or not np.isfinite(r).all() or (r < -1).any():
        raise ValueError('Metrics require finite returns >= -100% and aligned risk-free data.')
    excess = r - rf / trading_days
    vol = r.std() * np.sqrt(trading_days)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(trading_days)
    growth = (1 + r).cumprod()
    drawdown = growth / growth.cummax().clip(lower=1) - 1
    cagr = annualized_return(r, trading_days)
    monthly = (1 + r).resample('ME').prod() - 1
    divide = lambda a, b: float(a / b) if np.isfinite(b) and abs(b) > 1e-14 else np.nan
    tail = empirical_var_cvar(r)
    output = {'cumulative_return': float(growth.iloc[-1] - 1), 'cagr': cagr,
              'annualized_volatility': float(vol), 'sharpe': divide(excess.mean() * trading_days, vol),
              'sortino': divide(excess.mean() * trading_days, downside),
              'max_drawdown': float(drawdown.min()), 'calmar': divide(cagr, abs(drawdown.min())),
              'var_95_daily': tail['var'], 'cvar_95_daily': tail['cvar'],
              'worst_month': float(monthly.min()), 'best_month': float(monthly.max()),
              'beta': np.nan, 'alpha': np.nan, 'tracking_error': np.nan, 'information_ratio': np.nan,
              'turnover': float(diagnostics['turnover'].sum()) if diagnostics is not None and len(diagnostics) else 0.,
              'transaction_costs': float(diagnostics['cost_rate'].sum()) if diagnostics is not None and len(diagnostics) else 0.}
    if benchmark is not None:
        pair = pd.concat([r.rename('p'), benchmark.rename('b'), rf.rename('rf')], axis=1).dropna()
        if len(pair) > 1:
            pe, be = pair.p - pair.rf / trading_days, pair.b - pair.rf / trading_days
            beta = divide(pe.cov(be), be.var())
            active = pair.p - pair.b
            te = active.std() * np.sqrt(trading_days)
            output.update(beta=beta, alpha=float((pe.mean() - beta * be.mean()) * trading_days),
                          tracking_error=float(te), information_ratio=divide(active.mean() * trading_days, te))
    return output


def rolling_backtest_metrics(returns, risk_free=0., trading_days=252):
    r = returns.astype(float)
    rf = pd.Series(risk_free, index=r.index) if np.isscalar(risk_free) else risk_free.reindex(r.index)
    excess = r - rf / trading_days
    result = pd.DataFrame(index=r.index)
    for years in (1, 3):
        n = years * trading_days
        result[f'return_{years}y'] = r.rolling(n).apply(lambda x: np.prod(1 + x) ** (1 / years) - 1, raw=True)
        vol = r.rolling(n).std() * np.sqrt(trading_days)
        result[f'sharpe_{years}y'] = (excess.rolling(n).mean() * trading_days / vol.replace(0, np.nan))
        if years == 1:
            result['volatility_1y'] = vol
    growth = (1 + r).cumprod()
    result['drawdown'] = growth / growth.cummax().clip(lower=1) - 1
    # Trailing-window MDD includes the initial wealth of each window.
    result['rolling_drawdown_1y'] = r.rolling(trading_days).apply(
        lambda x: np.min(np.cumprod(1 + x) / np.maximum.accumulate(np.r_[1., np.cumprod(1 + x)])[1:] - 1), raw=True)
    return result


def regime_backtest_metrics(returns, decision_regimes, risk_free=0.):
    """Labels known BEFORE each return; conditional days concatenated, not a tradable strategy."""
    from regimes import REGIMES
    records = []
    for regime in REGIMES:
        selected = returns.loc[decision_regimes.reindex(returns.index).eq(regime)]
        rf = risk_free if np.isscalar(risk_free) else risk_free.reindex(selected.index)
        metrics = backtest_metrics(selected, risk_free=rf)
        records.append({'regime': regime, 'observations': len(selected),
                        'annualized_return': metrics.get('cagr', np.nan),
                        'volatility': metrics.get('annualized_volatility', np.nan),
                        'sharpe': metrics.get('sharpe', np.nan) if len(selected) >= 20 else np.nan,
                        'max_drawdown': metrics.get('max_drawdown', np.nan),
                        'positive_frequency': float((selected > 0).mean()) if len(selected) else np.nan})
    return pd.DataFrame(records)


STRESS_PERIODS = {'Dot-com bust': ('2000-03-24', '2002-10-09'),
                  'Global financial crisis': ('2007-10-09', '2009-03-09'),
                  'COVID crash': ('2020-02-19', '2020-03-23'),
                  '2022 inflation/rate shock': ('2021-12-31', '2022-12-30')}


def historical_stress_tests(prices, weights, periods=None):
    """Buy-and-hold replay; require full start/end and internal coverage, no proxies."""
    records = []
    active = {t: w for t, w in weights.items() if w > 0}
    if not active or not np.isclose(sum(active.values()), 1) or any(w < 0 for w in weights.values()):
        raise ValueError('Stress weights must be nonnegative and sum to one.')
    for name, (start, end) in (STRESS_PERIODS if periods is None else periods).items():
        window = prices.reindex(columns=active).loc[start:end]
        missing = [t for t in active if window.empty or window[t].isna().any()
                   or window.index[0] > pd.Timestamp(start)
                   or window.index[-1] < pd.Timestamp(end)]
        row = {'event': name, 'start': start, 'end': end, 'status': 'incomplete coverage' if missing else 'complete',
               'missing_assets': ', '.join(missing), 'return': np.nan, 'max_drawdown': np.nan}
        if not missing:
            if (window <= 0).any().any():
                raise ValueError('Stress prices must be positive.')
            wealth = (window / window.iloc[0]).mul(pd.Series(active)).sum(axis=1)
            row.update({'return': float(wealth.iloc[-1] - 1),
                        'max_drawdown': float((wealth / wealth.cummax() - 1).min())})
        records.append(row)
    return pd.DataFrame(records)


def recommendation_diagnostics(result):
    """Conservative evidence classification; never a probability of outperformance."""
    rows = []
    confidence = result.get('recommended_portfolio', {}).get('confidence', {}).get('overall_score', 0.)
    for asset, policy in result.get('policy', {}).items():
        stability = result.get('stability', {}).get('per_asset', {}).get(asset, {})
        agreement = result.get('optimizer_consensus', {}).get('per_asset', {}).get(asset, {}).get('agreement_score', 0.)
        score = min(float(confidence or 0), float(stability.get('weight_stability_score', 0) or 0), float(agreement or 0))
        low, high = result.get('constraints', {}).get('asset_bounds', {}).get(asset, [None, None])
        rows.append({'Asset': asset, 'Central weight': result.get('recommended_portfolio', {}).get('central_weights', {}).get(asset),
                     'Bootstrap median': stability.get('median'), '10th percentile': stability.get('q10'),
                     '90th percentile': stability.get('q90'), 'Weight stability': stability.get('weight_stability_score'),
                     'Optimizer agreement': agreement, 'Strategic weight': policy.get('strategic_weight'),
                     'Policy low': low, 'Policy high': high,
                     'Annual regime view adjustment': result.get('expected_returns', {}).get('regime_bl_views', {}).get(asset, {}).get('adjustment'),
                     'Conviction': 'HIGH' if score >= .75 else 'MEDIUM' if score >= .4 else 'LOW',
                     'Reason codes': result.get('reason_codes', {}).get(asset, [])})
    return pd.DataFrame(rows)
