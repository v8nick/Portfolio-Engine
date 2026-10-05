"""Input validation and read-only presentation transforms, without financial models."""
from __future__ import annotations

import math
import pandas as pd
from config import live
from portfolio_decision import resolve_allocation_policy

REASONS = {
    'positive_regime_return_delta': 'Current regime modestly favors this exposure.',
    'negative_regime_return_delta': 'Current regime modestly disfavors this exposure.',
    'no_regime_return_delta': 'No regime return adjustment is available.',
    'optimizer_consensus_positive': 'Portfolio methods favor a higher weight.',
    'optimizer_consensus_negative': 'Portfolio methods favor a lower weight.',
    'optimizer_disagreement': 'Portfolio methods disagree materially on this weight.',
    'unstable_optimizer_weight': 'Weight is sensitive to estimation changes.',
    'concentration_limit': 'Position exceeds portfolio concentration limits.',
    'strategic_band_limit': 'Policy band requires review or constrains this position.',
    'within_no_trade_band': 'Difference is below the rebalance threshold.',
    'within_recommended_range': 'Current weight is within the recommended range.',
    'low_recommendation_confidence': 'Confidence is too low for a discretionary change.',
    'transaction_cost_not_justified': 'Estimated benefit does not justify transaction costs.',
    'tax_cost_not_justified': 'Estimated benefit does not justify costs including taxes.',
    'within_tactical_band': 'Tactical adjustment remains within policy bands.',
    'black_litterman_support': 'Black-Litterman views support the direction of the tilt.',
    'tight_financial_conditions': 'Tight financial conditions dampen tactical tilts.',
    'elevated_stress': 'Elevated stress dampens tactical tilts.',
    'strategic_anchor': 'Allocation remains anchored to strategic policy.',
}


def reason_text(codes):
    return ' '.join(REASONS.get(c, c.replace('_', ' ').capitalize() + '.') for c in dict.fromkeys(codes or []))


def number(value, kind='number', signed=False):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 'N/A'
    sign = '+' if signed else ''
    if kind == 'percent':
        return format(value, f'{sign}.1%')
    if kind == 'yield':
        return f'{value:.2f}%'
    if kind == 'bp':
        return format(value, '+.1f') + ' bp'
    if kind == 'money':
        return f'${value:,.0f}'
    return format(value, f'{sign}.2f')


def initial_editor():
    rows = []
    for ticker in dict.fromkeys([*live.STRATEGIC_ALLOCATION, *live.CURRENT_WEIGHTS]):
        item = live.STRATEGIC_ALLOCATION.get(ticker, {
            'strategic_weight': 0., 'minimum_weight': 0.,
            'maximum_weight': live.MAX_INDIVIDUAL_SATELLITE_WEIGHT,
            'tactical_low': 0., 'tactical_high': live.MAX_INDIVIDUAL_SATELLITE_WEIGHT,
            'role': 'satellite', 'group': live.SATELLITE_GROUPS.get(ticker, 'satellite')})
        rows.append({'Ticker': ticker, 'Current %': live.CURRENT_WEIGHTS.get(ticker, 0.) * 100,
                     'Strategic %': item['strategic_weight'] * 100,
                     'Low %': item['tactical_low'] * 100, 'High %': item['tactical_high'] * 100,
                     'Minimum %': item['minimum_weight'] * 100, 'Maximum %': item['maximum_weight'] * 100,
                     'Fixed %': None if item.get('fixed_weight') is None else item['fixed_weight'] * 100,
                     'Role': item['role'], 'Group': item['group']})
    return pd.DataFrame(rows)


def validate_editor(frame, satellite_cap, individual_cap):
    """Convert displayed percentages to backend fractions; never silently normalize."""
    errors, holdings, policy = [], {}, {}
    for index, row in frame.iterrows():
        raw = row.get('Ticker')
        if raw is None or pd.isna(raw) or not str(raw).strip():
            errors.append(f'Row {index + 1}: enter a ticker or delete the row.')
            continue
        ticker = str(raw).strip().upper()
        if ticker in policy:
            errors.append(f'{ticker}: duplicate ticker.')
            continue
        try:
            values = {key: float(row[key]) / 100 for key in
                      ('Current %', 'Strategic %', 'Low %', 'High %', 'Minimum %', 'Maximum %')}
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in values.values()):
                raise ValueError('weights must be finite percentages from 0 to 100')
            item = {'strategic_weight': values['Strategic %'], 'tactical_low': values['Low %'],
                    'tactical_high': values['High %'], 'minimum_weight': values['Minimum %'],
                    'maximum_weight': values['Maximum %'], 'role': row['Role'], 'group': row['Group']}
            if item['role'] not in ('core', 'satellite') or pd.isna(item['group']) or not str(item['group']).strip():
                raise ValueError('choose core/satellite and enter an asset group')
            fixed = row.get('Fixed %')
            if pd.notna(fixed):
                item['fixed_weight'] = float(fixed) / 100
            if not item['tactical_low'] <= item['strategic_weight'] <= item['tactical_high']:
                raise ValueError('strategic target must lie within the tactical low/high band')
            if item['minimum_weight'] > item['maximum_weight']:
                raise ValueError('minimum must not exceed maximum')
            policy[ticker] = item
            if values['Current %'] > 0:
                holdings[ticker] = values['Current %']
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f'{ticker}: {exc}.')
    for name, total in [('Current', sum(holdings.values())),
                        ('Strategic', sum(p['strategic_weight'] for p in policy.values()))]:
        if not math.isclose(total, 1., abs_tol=1e-8, rel_tol=0):
            errors.append(f'{name} weights total {total * 100:.4f}%; they must total 100%.')
    if not errors:
        try:
            resolve_allocation_policy(policy, holdings, satellite_cap, individual_cap)
        except ValueError as exc:
            errors.append(f'Portfolio policy: {exc}')
    return holdings, policy, errors


def allocation_table(result):
    recommended = result.get('recommended_portfolio', {})
    rows = []
    for ticker in result.get('policy', {}):
        low, high = recommended.get('weight_ranges', {}).get(ticker, [None, None])
        rows.append({'Asset': ticker, 'Current': result.get('current_portfolio', {}).get('weights', {}).get(ticker),
                     'Strategic': result.get('strategic_portfolio', {}).get('weights', {}).get(ticker),
                     'Recommended': recommended.get('central_weights', {}).get(ticker), 'Low': low, 'High': high})
    return pd.DataFrame(rows, columns=['Asset', 'Current', 'Strategic', 'Recommended', 'Low', 'High'])


def metric_table(result):
    rows = []
    for name, key in [('Current', 'current_portfolio'), ('Strategic', 'strategic_portfolio'),
                      ('Recommended', 'recommended_portfolio')]:
        p = result.get(key, {})
        m, tail = p.get('metrics', {}), p.get('historical_tail_risk', {})
        rows.append({'Portfolio': name, 'Expected return': m.get('expected_return'),
                     'Volatility': m.get('volatility'), 'Sharpe': m.get('sharpe'),
                     'Historical max drawdown': tail.get('max_drawdown'),
                     'Downside deviation': tail.get('downside_deviation'), 'Daily VaR': tail.get('var'),
                     'Daily CVaR': tail.get('cvar'), 'HHI': m.get('hhi'), 'Effective holdings': m.get('effective_holdings')})
    return pd.DataFrame(rows)


def action_table(result):
    rows = []
    for ticker, item in result.get('rebalance', {}).get('actions', {}).items():
        low, high = item.get('recommended_range', [None, None])
        current, target = item.get('current_weight'), item.get('central_weight')
        rows.append({'Asset': ticker, 'Current': current, 'Target': target, 'Low': low, 'High': high,
                     'Difference': None if current is None or target is None else target - current,
                     'Action': item.get('action', 'N/A'), 'Required?': item.get('required', False),
                     'Estimated cost': 'N/A — aggregate estimate only',
                     'Reason': reason_text(result.get('reason_codes', {}).get(ticker, item.get('reason_codes', [])))})
    return pd.DataFrame(rows)
