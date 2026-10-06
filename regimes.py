"""Trailing market-proxy quadrants, heuristic probabilities, descriptive outcomes.

No portfolio integration or parameter fitting. Data availability is modeled using
observation dates plus a configurable Treasury lag, not actual release vintages.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from config.shared import (
    REGIME_HORIZONS, REGIME_ZSCORE_WINDOW, REGIME_ZSCORE_MIN_PERIODS,
    REGIME_ZSCORE_CLIP, REGIME_SMOOTHING_SPAN, REGIME_TEMPERATURE,
    REGIME_MIN_COMPONENTS, REGIME_MAX_AGE_DAYS, REGIME_YIELD_LAG_DAYS,
    REGIME_CONDITIONS_THRESHOLD, REGIME_STRESS_THRESHOLDS,
    REGIME_SHRINKAGE_STRENGTH, MARKET_ZSCORE_WINDOW,
)
from market_intelligence import (
    HORIZONS, relative_returns, rolling_zscore, build_market_snapshot,
    get_regime_features, load_market_data,
)

REGIMES = ('goldilocks', 'reflation', 'slowdown', 'stagflation')
FORWARD_ASSETS = ('SPY', 'QQQ', 'RSP', 'IWM', 'SMH', 'TLT', 'IEF', 'SGOV',
                  'BIL', 'LQD', 'HYG', 'TIP', 'GLD', 'DBC', 'VNQ')
# (Phase 1 feature prefix, sign, family weight). Average horizons within each
# family first, so missing horizons cannot double a proxy's economic influence.
COMPONENTS = {
    'growth': [('growth_iwm_spy', 1, 1), ('growth_rsp_spy', 1, 1),
               ('growth_xly_xlp', 1, 1), ('growth_hyg_lqd', 1, 1), ('growth_spy', 1, 1)],
    'inflation': [('inflation_10y_change_bp', 1, 1), ('inflation_tip_ief', 1, 1),
                  ('inflation_dbc', 1, 1), ('inflation_gld', 1, .5)],
    'financial_conditions': [('conditions_2y_change_bp', 1, 1), ('conditions_dxy', 1, 1),
                             ('conditions_hyg', -1, 1), ('conditions_vix', 1, 1)],
    'stress': [('stress_vix_zscore', 1, 1), ('stress_spy_realized_vol_21d', 1, 1),
               ('stress_credit_weakness', 1, 1), ('stress_breadth_weakness', 1, 1)],
}


def _frame(frame: pd.DataFrame, as_of=None) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame.sort_index()
    if not frame.index.is_unique:
        raise ValueError('Input dates must be unique after normalization.')
    if as_of is not None:
        frame = frame.loc[:pd.Timestamp(as_of).tz_localize(None).normalize()]
    return frame.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _asof(series: pd.Series, dates: pd.DatetimeIndex, max_age_days: int,
          lag_days: int = 0) -> pd.Series:
    """Carry only already available observations, bounded by original source age."""
    series = series.dropna().sort_index()
    if series.empty:
        return pd.Series(np.nan, index=dates, dtype=float)
    available_dates = series.index + pd.Timedelta(days=lag_days)
    values = pd.Series(series.to_numpy(), index=available_dates).reindex(dates, method='ffill')
    source_dates = pd.Series(series.index, index=available_dates).reindex(dates, method='ffill')
    ages = pd.Series(dates, index=dates) - source_dates
    return values.where(ages <= pd.Timedelta(days=max_age_days))


def build_regime_feature_history(
    prices: pd.DataFrame, yields: pd.DataFrame, horizons: tuple[int, ...] = REGIME_HORIZONS,
    max_age_days: int = REGIME_MAX_AGE_DAYS, yield_lag_days: int = REGIME_YIELD_LAG_DAYS,
    vix_window: int = MARKET_ZSCORE_WINDOW, as_of=None,
) -> pd.DataFrame:
    """Vectorized historical equivalents of Phase 1 get_regime_features().

    Horizons count valid source/joint observations as in Phase 1. History rows
    use price sessions (yield dates if no prices); no artificial holiday rows.
    No backfill. A yield feature becomes usable observation-date + lag_days.
    """
    if not horizons or len(set(horizons)) != len(horizons) or any(n not in HORIZONS.values() for n in horizons):
        raise ValueError('Horizons must be distinct Phase 1 observation horizons.')
    if max_age_days < 0 or yield_lag_days < 0:
        raise ValueError('Data age and yield lag must be nonnegative.')
    prices, yields = _frame(prices, as_of), _frame(yields, as_of)
    prices = prices.dropna(how='all')
    yields = yields.dropna(how='all')
    dates = prices.index if len(prices) else yields.index
    features = pd.DataFrame(index=dates)
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    def momentum(asset, n):
        s = prices.get(asset, empty).dropna()
        return s.pct_change(n, fill_method=None).replace([np.inf, -np.inf], np.nan)
    def add(name, series, lag=0):
        features[name] = _asof(series, dates, max_age_days, lag)
    for n in horizons:
        suffix = f'{n}D'
        for pair in ('IWM_SPY', 'RSP_SPY', 'XLY_XLP', 'HYG_LQD'):
            a, b = pair.split('_')
            add(f'growth_{pair.lower()}_{suffix}', relative_returns(prices, a, b, n))
        add(f'growth_spy_{suffix}', momentum('SPY', n))
        add(f'inflation_tip_ief_{suffix}', relative_returns(prices, 'TIP', 'IEF', n))
        for asset in ('DBC', 'GLD'):
            add(f'inflation_{asset.lower()}_{suffix}', momentum(asset, n))
        for maturity, group in (('10Y', 'inflation'), ('2Y', 'conditions')):
            s = yields.get(maturity, empty).dropna()
            add(f'{group}_{maturity.lower()}_change_bp_{suffix}', s.diff(n) * 100, yield_lag_days)
        for asset in ('DXY', 'HYG', 'VIX'):
            add(f'conditions_{asset.lower()}_{suffix}', momentum(asset, n))
        for pair, name in (('HYG_LQD', 'credit'), ('RSP_SPY', 'breadth')):
            a, b = pair.split('_')
            add(f'stress_{name}_weakness_{suffix}', -relative_returns(prices, a, b, n))
    vix = prices.get('VIX', empty).dropna()
    add('stress_vix_zscore', rolling_zscore(vix, vix_window))
    # Keep the same 21 consecutive price-session requirement as the snapshot.
    spy_returns = prices.get('SPY', empty).pct_change(fill_method=None)
    realized = spy_returns.rolling(21, min_periods=21).std() * np.sqrt(252)
    add('stress_spy_realized_vol_21d', realized)
    features.attrs.update(horizons=tuple(horizons), yield_lag_days=yield_lag_days,
                          units='decimal returns/differences; yield changes in bp')
    return features


def standardize_regime_features(features: pd.DataFrame, window: int = REGIME_ZSCORE_WINDOW,
                                min_periods: int = REGIME_ZSCORE_MIN_PERIODS,
                                clip: float = REGIME_ZSCORE_CLIP) -> pd.DataFrame:
    """Rolling trailing z-scores including t; no full-sample estimation/backfill."""
    if not 2 <= min_periods <= window or not np.isfinite(clip) or clip <= 0:
        raise ValueError('Require 2 <= min_periods <= window and positive finite clip.')
    features = _frame(features)
    rolling = features.rolling(window, min_periods=min_periods)
    std = rolling.std(ddof=1)
    return ((features - rolling.mean()) / std.replace(0, np.nan)).clip(-clip, clip)


def composite_regime_scores(standardized: pd.DataFrame,
                            min_components: int = REGIME_MIN_COMPONENTS) -> pd.DataFrame:
    """Positive conditions=tighter; positive stress=more stress.

    Positive inflation=market-implied inflation/long-rate pressure, not measured
    inflation. Gold has half the weight of each other inflation proxy.
    Missing families are renormalized; require at least min_components families.
    """
    if not 1 <= min_components <= 4:
        raise ValueError('min_components must be between one and four.')
    result = pd.DataFrame(index=standardized.index)
    for group, definitions in COMPONENTS.items():
        families = {}
        weights = {}
        for prefix, sign, weight in definitions:
            columns = [c for c in standardized if c == prefix or c.startswith(prefix + '_')]
            families[prefix] = standardized[columns].mean(axis=1) * sign if columns else pd.Series(np.nan, index=standardized.index)
            weights[prefix] = weight
        family_frame = pd.DataFrame(families)
        weight_series = pd.Series(weights)
        denominator = family_frame.notna().mul(weight_series).sum(axis=1)
        count = family_frame.notna().sum(axis=1)
        result[f'{group}_score'] = (family_frame.mul(weight_series).sum(axis=1) / denominator.replace(0, np.nan)).where(count >= min_components)
        result[f'{group}_component_count'] = count
        result[f'{group}_coverage'] = denominator / weight_series.sum()
    result['feature_count'] = standardized.notna().sum(axis=1)
    return result


def classify_quadrant(growth: float, inflation: float) -> str:
    if not np.isfinite([growth, inflation]).all():
        return 'unavailable'
    if growth > 0:
        return 'reflation' if inflation > 0 else 'goldilocks'
    return 'stagflation' if inflation > 0 else 'slowdown'


def regime_probabilities(growth: float, inflation: float,
                         temperature: float = REGIME_TEMPERATURE) -> dict[str, float]:
    """Normalized softmax scores, NOT calibrated probabilities of the economy."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive and finite.')
    if not np.isfinite([growth, inflation]).all():
        return dict.fromkeys(REGIMES, .25)
    logits = np.array([growth - inflation, growth + inflation,
                       -growth - inflation, -growth + inflation], dtype=float)
    # Subtract maximum before dividing to keep extreme/small-temperature inputs safe.
    with np.errstate(over='ignore', under='ignore'):
        mass = np.exp((logits - logits.max()) / temperature)
    return dict(zip(REGIMES, (mass / mass.sum()).tolist()))


def classify_overlays(conditions: float, stress: float,
                      conditions_threshold: float = REGIME_CONDITIONS_THRESHOLD,
                      stress_thresholds: tuple[float, float, float] = REGIME_STRESS_THRESHOLDS) -> dict[str, str]:
    if conditions_threshold <= 0 or not np.isfinite(conditions_threshold):
        raise ValueError('Conditions threshold must be positive and finite.')
    if len(stress_thresholds) != 3 or not np.isfinite(stress_thresholds).all() or not stress_thresholds[0] < stress_thresholds[1] < stress_thresholds[2]:
        raise ValueError('Stress thresholds must be three finite increasing values.')
    financial = ('unavailable' if not np.isfinite(conditions) else 'tight' if conditions >= conditions_threshold else 'easy' if conditions <= -conditions_threshold else 'neutral')
    stress_label = 'unavailable' if not np.isfinite(stress) else ('low', 'normal', 'elevated', 'severe')[int(np.searchsorted(stress_thresholds, stress, side='right'))]
    return {'financial_conditions': financial, 'stress': stress_label}


def classify_regime_history(features: pd.DataFrame, *, window: int = REGIME_ZSCORE_WINDOW,
                            min_periods: int = REGIME_ZSCORE_MIN_PERIODS,
                            clip: float = REGIME_ZSCORE_CLIP,
                            smoothing_span: int = REGIME_SMOOTHING_SPAN,
                            temperature: float = REGIME_TEMPERATURE,
                            min_components: int = REGIME_MIN_COMPONENTS) -> pd.DataFrame:
    """Public feature-to-state interface. Both raw and smoothed regimes are exposed."""
    if smoothing_span < 1:
        raise ValueError('Smoothing span must be at least one.')
    standardized = standardize_regime_features(features, window, min_periods, clip)
    result = composite_regime_scores(standardized, min_components)
    groups = tuple(COMPONENTS)
    for group in groups:
        score = result[f'{group}_score']
        # Missing current scores stay missing; EWMA cannot silently fill outages.
        result[f'smoothed_{group}_score'] = score.ewm(span=smoothing_span, adjust=False, ignore_na=True).mean().where(score.notna())
    result['raw_regime'] = [classify_quadrant(g, i) for g, i in zip(result['growth_score'], result['inflation_score'])]
    result['smoothed_regime'] = [classify_quadrant(g, i) for g, i in zip(result['smoothed_growth_score'], result['smoothed_inflation_score'])]
    result['primary_regime'] = result['smoothed_regime']
    probabilities = [regime_probabilities(g, i, temperature) for g, i in zip(result['smoothed_growth_score'], result['smoothed_inflation_score'])]
    for regime in REGIMES:
        result[f'probability_{regime}'] = [p[regime] for p in probabilities]
    coverage = result[['growth_coverage', 'inflation_coverage']].min(axis=1)
    result['confidence'] = [sorted(p.values(), reverse=True)[0] - sorted(p.values(), reverse=True)[1] for p in probabilities]
    result['confidence'] *= coverage
    overlays = [classify_overlays(c, s) for c, s in zip(result['smoothed_financial_conditions_score'], result['smoothed_stress_score'])]
    for group in ('financial_conditions', 'stress'):
        result[f'{group}_overlay'] = [o[group] for o in overlays]
    result.attrs['probability_kind'] = 'heuristic normalized quadrant scores; uncalibrated'
    return result


def build_regime_history(prices: pd.DataFrame, yields: pd.DataFrame, *,
                         feature_options: dict | None = None, **model_options) -> pd.DataFrame:
    return classify_regime_history(build_regime_feature_history(prices, yields, **(feature_options or {})), **model_options)


def _finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def get_current_regime(prices: pd.DataFrame, yields: pd.DataFrame, *, as_of=None,
                       feature_options: dict | None = None, **model_options) -> dict:
    """JSON-safe current state; no downloads. Defaults to last observed data date.

    Use as_of explicitly to evaluate current freshness. Inferred-date historical
    calls are reproducible. A single Phase 1 snapshot supplies current raw features
    and quality flags; no repeated snapshots are used for historical construction.
    """
    prices, yields = _frame(prices, as_of), _frame(yields, as_of)
    dates = [f.index.max() for f in (prices, yields) if len(f)]
    date = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else (max(dates) if dates else None)
    options = dict(feature_options or {})
    if 'as_of' in options:
        raise ValueError('Pass the current evaluation date via as_of, not feature_options.')
    features = build_regime_feature_history(prices, yields, as_of=date, **options)
    observation_date = features.index[-1] if len(features) else date
    yield_cutoff = observation_date - pd.Timedelta(days=options.get('yield_lag_days', REGIME_YIELD_LAG_DAYS)) if observation_date is not None else None
    eligible_yields = yields.loc[:yield_cutoff] if yield_cutoff is not None else yields
    snapshot = build_market_snapshot(prices, eligible_yields, as_of=date,
                                     zscore_window=options.get('vix_window', MARKET_ZSCORE_WINDOW),
                                     max_age_days=options.get('max_age_days', REGIME_MAX_AGE_DAYS))
    current_features = {}
    for n in options.get('horizons', REGIME_HORIZONS):
        current_features.update(get_regime_features(snapshot, f'{n}D'))
    if len(features):
        # At an explicit later evaluation date, remove now-stale raw inputs before
        # recomputing the latest composites/smoothing. Historical rows stay intact.
        features.loc[features.index[-1], [c for c in features if c not in current_features]] = np.nan
    history = classify_regime_history(features, **model_options)
    quality = list(snapshot['data_quality'])
    if not len(history):
        return {'as_of': date.isoformat() if date is not None else None, 'observation_as_of': None,
                'primary_regime': 'unavailable', 'raw_regime': 'unavailable', 'smoothed_regime': 'unavailable',
                'scores': dict.fromkeys(COMPONENTS), 'probabilities': dict.fromkeys(REGIMES, .25),
                'probability_kind': 'heuristic normalized quadrant scores; uncalibrated',
                'confidence': 0., 'overlays': dict.fromkeys(('financial_conditions', 'stress'), 'unavailable'),
                'feature_count': 0, 'coverage': dict.fromkeys(COMPONENTS, 0.),
                'data_quality': quality + ['No usable historical observations.'], 'features': current_features}
    row = history.iloc[-1]
    for group in COMPONENTS:
        if pd.isna(row[f'{group}_score']):
            quality.append(f'{group}: insufficient standardized component history or current data.')
        elif row[f'{group}_coverage'] < 1:
            quality.append(f'{group}: partial component coverage; available weights renormalized.')
    return {'as_of': date.isoformat(), 'observation_as_of': observation_date.isoformat(),
            'primary_regime': row['primary_regime'], 'raw_regime': row['raw_regime'],
            'smoothed_regime': row['smoothed_regime'],
            'scores': {g: _finite(row[f'smoothed_{g}_score']) for g in COMPONENTS},
            'raw_scores': {g: _finite(row[f'{g}_score']) for g in COMPONENTS},
            'probabilities': {r: float(row[f'probability_{r}']) for r in REGIMES},
            'probability_kind': history.attrs['probability_kind'], 'confidence': float(row['confidence']),
            'overlays': {g: row[f'{g}_overlay'] for g in ('financial_conditions', 'stress')},
            'feature_count': int(row['feature_count']),
            'coverage': {g: float(row[f'{g}_coverage']) for g in COMPONENTS},
            'data_quality': quality, 'features': current_features}


def regime_diagnostics(history: pd.DataFrame, regime_column: str = 'smoothed_regime') -> dict:
    labels = history.get(regime_column, pd.Series(dtype=str))
    valid = labels.isin(REGIMES)
    counts = labels[valid].value_counts().reindex(REGIMES, fill_value=0)
    run_ids = labels.ne(labels.shift()).cumsum()
    runs = pd.DataFrame({'regime': labels, 'run': run_ids}).loc[valid].groupby(['regime', 'run']).size()
    durations = runs.groupby(level=0).mean().reindex(REGIMES) if len(runs) else pd.Series(np.nan, index=REGIMES)
    transitions = valid & valid.shift(fill_value=False) & labels.ne(labels.shift())
    return {'regime_frequency': {r: float(counts[r] / valid.sum()) if valid.sum() else 0. for r in REGIMES},
            'sample_size_by_regime': {r: int(counts[r]) for r in REGIMES},
            'average_duration_observations': {r: _finite(durations[r]) for r in REGIMES},
            'number_of_transitions': int(transitions.sum()), 'unavailable_observations': int((~valid).sum()),
            'latest_regime': labels.iloc[-1] if len(labels) else 'unavailable',
            'notes': 'Duration includes censored first/last runs; unavailable rows break runs. Frequencies exclude unavailable rows.'}


def _forward_samples(regime_history: pd.DataFrame, asset_prices: pd.DataFrame,
                     horizon: int, regime_column: str, non_overlapping: bool, assets=None):
    prices = _frame(asset_prices).dropna(how='all').where(lambda x: x > 0)
    labels = regime_history[regime_column].reindex(prices.index)
    forward = prices.shift(-horizon) / prices - 1
    forward = forward.replace([np.inf, -np.inf], np.nan)
    samples = {}
    for asset in (FORWARD_ASSETS if assets is None else assets):
        returns = forward.get(asset, pd.Series(np.nan, index=prices.index))
        eligible = returns.notna() & labels.isin(REGIMES)
        if non_overlapping:
            # Global chronological selection per asset/horizon, not independently
            # within each regime. Adjacent selected intervals share no returns.
            selected = np.zeros(len(prices), dtype=bool)
            next_position = 0
            for pos in np.flatnonzero(eligible.to_numpy()):
                if pos >= next_position:
                    selected[pos] = True
                    next_position = pos + horizon
            eligible &= selected
        samples[asset] = pd.DataFrame({'regime': labels[eligible], 'return': returns[eligible]})
    return samples, forward


def analyze_regime_forward_returns(regime_history: pd.DataFrame, asset_prices: pd.DataFrame,
                                   horizons: tuple[int, ...] = (21, 63, 126), *,
                                   regime_column: str = 'smoothed_regime',
                                   non_overlapping: bool = False, as_of=None, assets=None) -> pd.DataFrame:
    """Descriptive outcomes AFTER t: P[t+h]/P[t]-1, with no unfinished outcomes.

    Endpoint horizons use the shared observed price-session calendar. Missing
    asset endpoints are excluded, never filled. Daily samples overlap heavily
    and are not statistically independent. as_of truncates outcomes as well as
    labels before shifting; this is required when used for historical decisions.
    assets optionally extends the default ETF universe (e.g. VEA/VWO/satellites).
    """
    if not horizons or any(not isinstance(h, (int, np.integer)) or h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError('Forward horizons must be distinct positive integers.')
    if regime_column not in regime_history:
        raise ValueError(f'Regime history must contain {regime_column}.')
    prices = _frame(asset_prices, as_of)
    labels = regime_history.loc[:as_of] if as_of is not None else regime_history
    records = []
    for h in horizons:
        samples, _ = _forward_samples(labels, prices, h, regime_column, non_overlapping, assets)
        for asset, frame in samples.items():
            for regime in REGIMES:
                values = frame.loc[frame['regime'] == regime, 'return']
                records.append({'regime': regime, 'asset': asset, 'horizon': h, 'n_observations': len(values),
                                'mean': values.mean(), 'median': values.median(), 'std': values.std(ddof=1),
                                'positive_frequency': (values > 0).mean() if len(values) else np.nan,
                                'q25': values.quantile(.25), 'q75': values.quantile(.75)})
    result = pd.DataFrame(records)
    result.attrs.update(return_unit='decimal cumulative forward return; not annualized',
                        sampling='non-overlapping global chronological intervals' if non_overlapping else 'daily overlapping intervals; observations are NOT independent',
                        interpretation='descriptive historical conditional outcomes, not forecasts',
                        as_of=pd.Timestamp(as_of).isoformat() if as_of is not None else None)
    return result


def estimate_regime_expected_returns(regime_history: pd.DataFrame, asset_prices: pd.DataFrame,
                                     horizons: tuple[int, ...] = (21, 63, 126), *, as_of=None,
                                     shrinkage_strength: float = REGIME_SHRINKAGE_STRENGTH,
                                     lambda_override: float | None = None,
                                     non_overlapping: bool = True,
                                     regime_column: str = 'smoothed_regime', assets=None) -> pd.DataFrame:
    """Shrink completed conditional outcomes toward same-horizon long-run means.

    lambda=n/(n+strength); default n counts non-overlapping outcomes. No regime
    samples => lambda=0 and unconditional fallback. Not connected to BL.
    Pass as_of for any historical decision to exclude future/unfinished returns.
    """
    if not np.isfinite(shrinkage_strength) or shrinkage_strength <= 0:
        raise ValueError('Shrinkage strength must be positive and finite.')
    if lambda_override is not None and (not np.isfinite(lambda_override) or not 0 <= lambda_override <= 1):
        raise ValueError('lambda_override must be in [0,1].')
    prices = _frame(asset_prices, as_of)
    analysis = analyze_regime_forward_returns(regime_history, prices, horizons, as_of=as_of,
                                             regime_column=regime_column, non_overlapping=non_overlapping, assets=assets)
    records = []
    for h in horizons:
        _, forward = _forward_samples(regime_history, prices, h, regime_column, False, assets)
        for asset in (FORWARD_ASSETS if assets is None else assets):
            values = forward.get(asset, pd.Series(dtype=float)).dropna()
            if non_overlapping:
                # Select by calendar position (not every h-th nonmissing return).
                positions = forward.index.get_indexer(values.index)
                selected, next_position = [], 0
                for value, pos in zip(values.to_numpy(), positions):
                    if pos >= next_position:
                        selected.append(value)
                        next_position = pos + h
                values = pd.Series(selected, dtype=float)
            unconditional = values.mean()
            for _, row in analysis.loc[(analysis['horizon'] == h) & (analysis['asset'] == asset)].iterrows():
                n = int(row['n_observations'])
                weight = (n / (n + shrinkage_strength) if lambda_override is None else lambda_override) if n else 0.
                shrunk = weight * row['mean'] + (1 - weight) * unconditional if n else unconditional
                records.append({'regime': row['regime'], 'asset': asset, 'horizon': h,
                                'raw_regime_estimate': row['mean'], 'unconditional_estimate': unconditional,
                                'shrunk_estimate': shrunk, 'sample_count': n,
                                'unconditional_count': len(values), 'lambda': weight})
    result = pd.DataFrame(records)
    result.attrs.update(analysis.attrs)
    result.attrs['interpretation'] = 'shrunk descriptive forward-return estimates; not calibrated forecasts'
    return result


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(frame.notna(), None).to_dict(orient='records')


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 2 market-proxy regimes and descriptive forward returns')
    parser.add_argument('--start', default='2007-01-01', help='Long history for trailing standardization')
    parser.add_argument('--end', default=None)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--non-overlapping', action='store_true')
    args = parser.parse_args()
    data = load_market_data(args.start, args.end)
    date = args.end or pd.Timestamp.now().normalize()
    current = get_current_regime(data['prices'], data['yields'], as_of=date)
    current['data_quality'].extend(data['data_quality'])
    history = build_regime_history(data['prices'], data['yields'], feature_options={'as_of': date})
    diagnostics = regime_diagnostics(history)
    analysis = analyze_regime_forward_returns(history, data['prices'], non_overlapping=args.non_overlapping, as_of=date)
    if args.json:
        print(json.dumps({'current': current, 'diagnostics': diagnostics,
                          'forward_returns': _records(analysis), 'analysis_notes': analysis.attrs}, indent=2, allow_nan=False))
        return
    print('\nCURRENT MARKET REGIME')
    print(f"As of: {current['as_of']} (observations through {current['observation_as_of']})")
    print(f"Primary: {current['primary_regime'].title()} | Raw: {current['raw_regime'].title()}")
    print(f"Model confidence: {current['confidence']:.0%} — uncalibrated score gap, adjusted for coverage")
    for group, value in current['scores'].items():
        overlay = current['overlays'].get(group)
        formatted = f'{value:+.2f}' if value is not None else 'unavailable'
        label = {'inflation': 'market-implied inflation / long-rate pressure',
                 'financial_conditions': 'financial conditions (positive=tighter)'}.get(group, group)
        print(f"{label}: {formatted}" + (f' ({overlay})' if overlay else ''))
    print('\nProbabilities — heuristic normalized scores:')
    for regime, value in current['probabilities'].items():
        print(f'{regime.title():12} {value:.0%}')
    selected = analysis.loc[(analysis['regime'] == current['primary_regime']) & (analysis['horizon'] == 63), ['asset', 'n_observations', 'median']]
    print('\nHistorical 3-month median forward returns — descriptive, not forecasts:')
    selected = selected.copy()
    selected['median'] = selected['median'].map(lambda value: f'{value:.2%}' if pd.notna(value) else 'unavailable')
    print(selected.to_string(index=False) if len(selected) else 'Unavailable for the current regime.')
    print(analysis.attrs['sampling'])
    print('\nDiagnostics: ' + json.dumps(diagnostics))
    if current['data_quality']:
        print('\nData quality: ' + '; '.join(current['data_quality']))


if __name__ == '__main__':
    main()


SCENARIOS = ('Baseline', 'Growth boom', 'Persistent slowdown', 'Persistent inflation',
             'Stagflation stress', 'Severe financial stress')


def estimate_transition_matrix(history, as_of=None, prior_strength=4.):
    """Daily-session transition counts including self transitions; unavailable labels break runs.

    Each row shrinks toward observed unconditional occupancy. Empty history uses
    an explicit uniform assumption. Both source and destination must be <= as_of.
    """
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError('Transition prior strength must be positive.')
    labels = history['primary_regime'].loc[:as_of] if as_of is not None else history['primary_regime']
    counts = pd.DataFrame(0., index=REGIMES, columns=REGIMES)
    for previous, current in zip(labels.iloc[:-1], labels.iloc[1:]):
        if previous in REGIMES and current in REGIMES:
            counts.loc[previous, current] += 1
    frequency = labels[labels.isin(REGIMES)].value_counts().reindex(REGIMES, fill_value=0).astype(float)
    # Small occupancy prior prevents inaccessible states without inventing data.
    prior = (frequency + 1) / (frequency.sum() + len(REGIMES))
    matrix = counts.add(prior * prior_strength, axis=1).div(counts.sum(axis=1) + prior_strength, axis=0)
    return matrix, {'transition_counts': counts.to_dict(), 'observed_occupancy': frequency.to_dict(),
                    'initial_distribution': prior.to_dict(), 'prior_strength': prior_strength,
                    'fallback': 'Uniform prior: no classified observations' if not frequency.sum() else 'Rows shrunk toward smoothed historical occupancy',
                    'frequency': 'one transition per trading observation', 'as_of': str(as_of)}


def scenario_transition_matrix(matrix, scenario='Baseline'):
    """Transparent hypothetical transition tilts; never applies fixed asset losses."""
    if scenario not in SCENARIOS:
        raise ValueError('Unknown regime scenario.')
    matrix = matrix.reindex(index=REGIMES, columns=REGIMES).astype(float)
    if not np.isfinite(matrix).all().all() or (matrix < 0).any().any() or not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError('Transition matrix must be row-stochastic.')
    # Destination multipliers and additional self-persistence are scenario assumptions.
    factors = {'Baseline': ([1, 1, 1, 1], []), 'Growth boom': ([2, 2, .5, .5], [0, 1]),
               'Persistent slowdown': ([.5, .5, 3, 1], [2]),
               'Persistent inflation': ([.5, 2, .5, 2], [1, 3]),
               'Stagflation stress': ([.5, .5, 1, 4], [3]),
               'Severe financial stress': ([.25, .25, 3, 3], [2, 3])}
    multipliers, persistence = factors[scenario]
    changed = matrix.to_numpy() * np.array(multipliers)
    for index in persistence:
        changed[index, index] *= 2
    changed /= changed.sum(axis=1, keepdims=True)
    return pd.DataFrame(changed, index=REGIMES, columns=REGIMES), {
        'scenario': scenario, 'destination_multipliers': dict(zip(REGIMES, multipliers)),
        'self_transition_multiplier': 2 if persistence else 1,
        'persistence_regimes': [REGIMES[i] for i in persistence],
        'limitation': 'Hypothetical transition tilt, not a calibrated crisis model. No arbitrary asset return shocks.'}
