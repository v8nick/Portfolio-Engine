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
              'history': pd.DataFrame(), 'outcomes': pd.DataFrame(), 'warnings': list(data.get('data_quality', []))}
    try:
        bundle['history'], bundle['outcomes'] = research_outputs(data['prices'], data['yields'],
            result['as_of'], tuple(result['policy']), options.get('regime_model_options', {}))
    except Exception as exc:
        bundle['warnings'].append(f'Historical regime tables unavailable ({type(exc).__name__}); portfolio results remain available.')
    return bundle
