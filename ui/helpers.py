"""Input validation and read-only presentation transforms, without financial models."""
from __future__ import annotations

import math
import csv
import io
import re
import pandas as pd
from config import live
from portfolio_decision import resolve_allocation_policy

REASONS = {
    'positive_regime_return_delta': 'Historical market-pattern adjustment favors this exposure.',
    'negative_regime_return_delta': 'Historical market-pattern adjustment disfavors this exposure.',
    'no_regime_return_delta': 'No historical market-pattern adjustment is available.',
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


# These names match the engine's saved input features; no scores are recomputed here.
MACRO_INPUTS = (
    ('Growth', 'growth_iwm_spy', 'Small caps − large caps (IWM − SPY)', 'percent', 'Higher supports the growth proxy'),
    ('Growth', 'growth_rsp_spy', 'Equal weight − cap weight (RSP − SPY)', 'percent', 'Higher supports the growth proxy'),
    ('Growth', 'growth_xly_xlp', 'Discretionary − staples (XLY − XLP)', 'percent', 'Higher supports the growth proxy'),
    ('Growth', 'growth_hyg_lqd', 'High yield − investment grade (HYG − LQD)', 'percent', 'Higher supports the growth proxy'),
    ('Growth', 'growth_spy', 'US equity return (SPY)', 'percent', 'Higher supports the growth proxy'),
    ('Inflation / rate pressure', 'inflation_10y_change_bp', '10-year Treasury yield change', 'bp', 'Higher raises the pressure proxy'),
    ('Inflation / rate pressure', 'inflation_tip_ief', 'Inflation protected − Treasuries (TIP − IEF)', 'percent', 'Higher raises the pressure proxy'),
    ('Inflation / rate pressure', 'inflation_dbc', 'Commodity return (DBC)', 'percent', 'Higher raises the pressure proxy'),
    ('Inflation / rate pressure', 'inflation_gld', 'Gold return (GLD)', 'percent', 'Higher raises the pressure proxy; half weight'),
    ('Financial conditions', 'conditions_2y_change_bp', '2-year Treasury yield change', 'bp', 'Higher supports the tightening proxy'),
    ('Financial conditions', 'conditions_dxy', 'Dollar index change (DXY)', 'percent', 'Higher supports the tightening proxy'),
    ('Financial conditions', 'conditions_hyg', 'High-yield bond return (HYG)', 'percent', 'Lower supports the tightening proxy'),
    ('Financial conditions', 'conditions_vix', 'VIX change', 'percent', 'Higher supports the tightening proxy'),
    ('Stress', 'stress_vix_zscore', 'VIX standardized level', 'number', 'Higher supports the stress proxy'),
    ('Stress', 'stress_spy_realized_vol_21d', 'SPY realized volatility (21 sessions, annualized)', 'percent', 'Higher supports the stress proxy'),
    ('Stress', 'stress_credit_weakness', 'Credit weakness (LQD − HYG)', 'percent', 'Higher supports the stress proxy'),
    ('Stress', 'stress_breadth_weakness', 'Breadth weakness (SPY − RSP)', 'percent', 'Higher supports the stress proxy'),
)


def macro_input_table(regime, groups=None):
    """Format the actual saved raw features, retaining missing values as N/A."""
    features = regime.get('features', {})
    rows = []
    for group, key, label, kind, meaning in MACRO_INPUTS:
        if groups is not None and group not in groups:
            continue
        single = key in ('stress_vix_zscore', 'stress_spy_realized_vol_21d')
        rows.append({'Category': group, 'Metric': label,
                     '21 sessions (~1 month)': '—' if single else number(features.get(f'{key}_21D'), kind, signed=True),
                     '63 sessions (~3 months)': '—' if single else number(features.get(f'{key}_63D'), kind, signed=True),
                     'Latest level': number(features.get(key), kind) if single else '—',
                     'Model direction': meaning})
    return pd.DataFrame(rows)


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


def import_portfolio_csv(content, existing, replace=False):
    """Import tickers/percentage columns atomically without guessing allocations."""
    try:
        text = content.decode('utf-8-sig')
        rows = [row for row in csv.reader(io.StringIO(text), strict=True) if any(cell.strip() for cell in row)]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError('Use a UTF-8, comma-separated CSV file.') from exc
    if not rows:
        raise ValueError('The CSV is empty.')
    columns = list(initial_editor().columns)
    normalize = lambda name: re.sub(r'[^a-z0-9]', '', name.strip().lower())
    aliases = {normalize(name): name for name in columns}
    aliases.update(symbol='Ticker', tickers='Ticker', current='Current %', weight='Current %',
                   currentweight='Current %', strategic='Strategic %', target='Strategic %',
                   strategicweight='Strategic %')
    header = [normalize(cell) for cell in rows[0]]
    if any(aliases.get(cell) == 'Ticker' for cell in header):
        unknown = [cell for cell in rows[0] if normalize(cell) not in aliases]
        if unknown:
            raise ValueError('Unsupported CSV columns: ' + ', '.join(unknown) + '. Use the downloadable template.')
        names = [aliases[cell] for cell in header]
        if len(names) != len(set(names)):
            raise ValueError('The CSV contains duplicate columns.')
        rows = rows[1:]
    else:
        if any(len(row) != 1 for row in rows):
            raise ValueError('Include a Ticker or Symbol header for files with allocation columns.')
        names = ['Ticker']
    if not rows:
        raise ValueError('The CSV contains no tickers.')
    if len(rows) > 1000:
        raise ValueError('Import at most 1,000 tickers at a time.')
    base = existing.copy(deep=True).reset_index(drop=True)
    records = base.to_dict('records')
    lookup = {str(row['Ticker']).strip().upper(): row for row in records if pd.notna(row.get('Ticker'))}
    imported, seen = [], set()
    for index, values in enumerate(rows, 1):
        if len(values) != len(names):
            raise ValueError(f'CSV row {index}: column count does not match the header.')
        supplied = dict(zip(names, (cell.strip() for cell in values)))
        ticker = supplied['Ticker'].upper()
        if not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=\-]{0,24}', ticker):
            raise ValueError(f'CSV row {index}: enter a valid ticker symbol.')
        if ticker in seen:
            raise ValueError(f'{ticker}: duplicate ticker in CSV.')
        seen.add(ticker)
        row = dict(lookup.get(ticker, {'Ticker': ticker, 'Current %': 0., 'Strategic %': 0.,
            'Low %': 0., 'High %': 100., 'Minimum %': 0., 'Maximum %': 100.,
            'Fixed %': None, 'Role': 'satellite', 'Group': 'satellite'}))
        row['Ticker'] = ticker
        for name, value in supplied.items():
            if name == 'Ticker':
                continue
            if name == 'Fixed %' and not value:
                row[name] = None
            elif not value:
                continue  # An omitted value preserves an existing setting.
            elif name == 'Role':
                if value.lower() not in ('core', 'satellite'):
                    raise ValueError(f'{ticker}: Role must be core or satellite.')
                row[name] = value.lower()
            elif name == 'Group':
                row[name] = value
            else:
                try:
                    parsed = float(value.removesuffix('%').strip())
                except ValueError as exc:
                    raise ValueError(f'{ticker}: {name} must be a percentage from 0 to 100.') from exc
                if not math.isfinite(parsed) or not 0 <= parsed <= 100:
                    raise ValueError(f'{ticker}: {name} must be a finite percentage from 0 to 100.')
                row[name] = parsed
        imported.append(row)
    if replace:
        records = imported
    else:
        for row in imported:
            if row['Ticker'] in lookup:
                lookup[row['Ticker']].update(row)
            else:
                records.append(row)
    return pd.DataFrame(records, columns=columns)


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
