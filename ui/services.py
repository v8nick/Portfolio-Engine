"""Thin adapters to existing backend APIs. No portfolio recommendations in global cache."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
from data import download_prices
from market_intelligence import load_market_data, build_market_snapshot
from regimes import build_regime_history, analyze_regime_forward_returns
from portfolio_decision import build_portfolio_recommendation

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def market_data(start, tickers, refresh_token=0):
    # A refresh token invalidates Streamlit's entry and bypasses the backend CSV TTL.
    data = load_market_data(start, cache_dir=ROOT / 'outputs' / 'market_cache',
                            ttl_seconds=0 if refresh_token else 3600)
    prices = data['prices']
    for ticker in tickers:
        if ticker in prices and not prices[ticker].dropna().empty:
            continue
        try:
            extra = download_prices([ticker], start)
            if ticker not in extra or extra[ticker].dropna().empty:
                data['data_quality'].append(f'{ticker}: required price data unavailable.')
            else:
                prices = prices.drop(columns=[ticker], errors='ignore').join(extra[[ticker]], how='outer')
        except Exception as exc:
            data['data_quality'].append(f'{ticker}: price provider unavailable ({type(exc).__name__}).')
    data['prices'] = prices
    return data


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def research_outputs(prices, yields, as_of, assets, model_options):
    history = build_regime_history(prices, yields, feature_options={'as_of': as_of}, **model_options)
    outcomes = analyze_regime_forward_returns(history, prices, horizons=(21, 63, 126),
        non_overlapping=True, as_of=as_of, assets=assets)
    return history, outcomes


def snapshot(data, as_of):
    output = build_market_snapshot(data['prices'], data['yields'], as_of=as_of)
    output['data_quality'] = list(dict.fromkeys(output['data_quality'] + data.get('data_quality', [])))
    output['sources'] = data.get('sources', {})
    return output


def run_analysis(data, holdings, policy, options, risk_free, as_of):
    """Single engine call, followed by optional descriptive research; errors do not alter the engine."""
    result = build_portfolio_recommendation(data['prices'], data['yields'], current_weights=holdings,
        strategic_allocation=policy, as_of=as_of, risk_free_rate=risk_free, options=options)
    bundle = {'recommendation': result, 'snapshot': snapshot(data, result['as_of']),
              'research_data': {'prices': data['prices'].loc[:result['as_of']].copy(), 'yields': data['yields'].loc[:result['as_of']].copy()},
              'history': pd.DataFrame(), 'outcomes': pd.DataFrame(), 'warnings': list(data.get('data_quality', []))}
    try:
        bundle['history'], bundle['outcomes'] = research_outputs(data['prices'], data['yields'],
            result['as_of'], tuple(result['policy']), options.get('regime_model_options', {}))
    except Exception as exc:
        bundle['warnings'].append(f'Historical regime tables unavailable ({type(exc).__name__}); portfolio results remain available.')
    return bundle


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def backtest_research(data, recommendation, controls):
    from walkforward import walk_forward_backtest
    return walk_forward_backtest(data['prices'], data['yields'],
        recommendation['current_portfolio']['weights'], recommendation['policy'],
        max_satellite_allocation=recommendation['constraints']['max_satellite_allocation'],
        max_individual_satellite_weight=recommendation['constraints']['max_individual_satellite_weight'],
        **controls)


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def simulation_research(data, recommendation, portfolio, controls):
    from simulation import run_simulation
    from data import compute_returns
    from mpt import annualize_mean_cov
    from config import shared
    from report import historical_stress_tests
    r = recommendation
    choices = {'Current': r['current_portfolio']['weights'], 'Strategic': r['strategic_portfolio']['weights'],
               'Recommended': r['recommended_portfolio']['central_weights']}
    choices.update({key: value.get('weights') for key, value in r['candidate_allocations'].items()})
    weights = choices.get(portfolio)
    if not weights:
        raise ValueError('Selected portfolio is unavailable; no other allocation substituted.')
    assets = list(r['policy'])
    prices = data['prices'].loc[:r['as_of']]
    # Use the same trailing risk sample as the saved recommendation for simulation fit.
    returns = compute_returns(prices[assets]).dropna().loc[r['data_policy']['risk_sample_start']:]
    _, covariance = annualize_mean_cov(returns, shared.TRADING_DAYS, shared.USE_COV_SHRINKAGE, shared.COV_SHRINKAGE_METHOD)
    mu = pd.Series(r['expected_returns']['black_litterman_posterior']).reindex(assets)
    history = build_regime_history(prices, data['yields'].loc[:r['as_of']], feature_options={'as_of': r['as_of']})
    output = run_simulation([weights.get(t, 0.) for t in assets], mu, covariance,
        historical_returns=returns, regime_history=history, as_of=r['as_of'], **controls)
    output['metadata']['expected_return_source'] = 'Saved production Black–Litterman posterior for parametric methods; empirical returns for bootstrap methods'
    output['stress_tests'] = historical_stress_tests(prices, weights)
    return output
