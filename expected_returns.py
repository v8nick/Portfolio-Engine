"""Return estimation only. No optimization, backtest scoring or parameter fitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import shared
from black_litterman import black_litterman_posterior, merge_manual_regime_views
from regimes import estimate_regime_expected_returns, get_current_regime

RETURN_MODELS = {'bl': 'Black–Litterman prior/posterior',
                 'historical_mean': 'Historical arithmetic mean (research comparison)',
                 'bl_regime': 'Black–Litterman with historical regime overlay'}


def estimate_expected_returns(returns, covariance, prior_weights, *, model='bl',
                              risk_free_rate=0., as_of=None, prices=None, yields=None,
                              regime_history=None, view_history=(), regime_model_options=None):
    """Annual arithmetic total returns. Only explicitly dated manual views are used.

    view_history records: {'as_of': date, 'absolute': {ticker: (return, confidence)},
    'relative': [(a,b,difference,confidence)]}. Latest eligible record replaces prior
    records. A historical run never imports today's configured manual opinions.
    Covariance is supplied by the separate covariance estimator.
    """
    if model not in RETURN_MODELS:
        raise ValueError('Unknown expected-return model.')
    if returns.empty:
        raise ValueError('Return estimation requires nonempty data.')
    date = pd.Timestamp(as_of if as_of is not None else returns.index[-1])
    if returns.index.max() > date:
        raise ValueError('Return estimation requires nonempty data available by as_of.')
    if list(covariance.index) != list(returns.columns) or list(covariance.columns) != list(returns.columns):
        raise ValueError('Covariance labels must match the return panel.')
    metadata = {'model': model, 'source': RETURN_MODELS[model], 'as_of': date.isoformat(),
                'estimation_start': returns.index[0].isoformat(),
                'observations': len(returns), 'annualization': 'arithmetic daily mean × 252',
                'manual_views': 'none; present-day configured opinions excluded'}
    historical = returns.mean() * shared.TRADING_DAYS
    if model == 'historical_mean':
        metadata['limitation'] = 'Research comparison, not evidence of superior forecasting.'
        return historical, metadata
    eligible = [v for v in view_history if pd.Timestamp(v['as_of']) <= date]
    selected = max(eligible, key=lambda v: pd.Timestamp(v['as_of'])) if eligible else {}
    absolute, relative = selected.get('absolute', {}), selected.get('relative', [])
    if selected:
        metadata['manual_views'] = str(selected['as_of'])
    _, prior = black_litterman_posterior(covariance, np.asarray(prior_weights),
        shared.BL_RISK_AVERSION, shared.BL_TAU, risk_free_rate=risk_free_rate)
    if model == 'bl_regime':
        if prices is None or yields is None or regime_history is None:
            raise ValueError('Regime overlay requires prices, yields and historical classifications.')
        from portfolio_decision import weight_regime_expected_returns, build_regime_bl_views
        cfg = shared.DECISION_SETTINGS
        regime = get_current_regime(prices.loc[:date], yields.loc[:date], as_of=date, **(regime_model_options or {}))
        estimates = estimate_regime_expected_returns(regime_history.loc[:date], prices.loc[:date],
            horizons=(cfg['regime_horizon'],), as_of=date, assets=tuple(returns.columns))
        weighted = weight_regime_expected_returns(estimates, regime['probabilities'], historical,
            horizon=cfg['regime_horizon'], minimum_samples=cfg['minimum_regime_samples'],
            sample_confidence_target=cfg['sample_confidence_target'],
            dispersion_scale=cfg['regime_return_dispersion_scale'])
        views = build_regime_bl_views(prior, weighted, regime, strength=cfg['regime_bl_strength'],
            max_adjustment=cfg['max_annual_regime_adjustment'], max_confidence=cfg['max_regime_view_confidence'])
        absolute, sources = merge_manual_regime_views(absolute, views)
        metadata.update(regime=regime['primary_regime'], view_sources=sources,
                        warnings=regime['data_quality'],
                        regime_outcomes='Only completed forward outcomes through decision date; descriptive estimates')
    mu, _ = black_litterman_posterior(covariance, np.asarray(prior_weights),
        shared.BL_RISK_AVERSION, shared.BL_TAU, absolute, relative, risk_free_rate)
    metadata['prior'] = 'Explicit fixed strategic modeling assumption, not inferred market capitalizations'
    return mu, metadata
