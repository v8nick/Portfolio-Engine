"""Phase 3 financial API; no UI, execution, or performance-fitted parameters.

build_portfolio_recommendation takes adjusted prices, percent Treasury yields,
and explicit policy/holdings. Inputs are truncated before all estimation.
Forward simple-return means are scaled linearly by 252/horizon to the existing
engine's annual arithmetic-return convention; this is not a CAGR forecast.
Default core policy is illustrative and independent of the legacy holdings.
"""
from __future__ import annotations

import copy
import json
import numpy as np
import pandas as pd
from pypfopt.exceptions import OptimizationError
from cvxpy.error import SolverError

from config import live, shared
from black_litterman import (implied_equilibrium_returns, black_litterman_posterior,
                            merge_manual_regime_views)
from data import compute_returns, resolve_risk_free_rate
from dashboard import risk_contribution
from mpt import (annualize_mean_cov, portfolio_stats, optimize_min_vol,
                 optimize_max_sharpe_constrained, optimize_risk_parity, optimize_min_cvar,
                 project_allocation, validate_allocation_weights)
from regimes import (REGIMES, get_current_regime, build_regime_history,
                     estimate_regime_expected_returns, _frame)
from implementation import compute_turnover, net_expected_return_after_all_costs
from report import build_portfolio_return_series, historical_tail_risk


def _settings(options=None):
    settings = copy.deepcopy(shared.DECISION_SETTINGS)
    settings.update(max_satellite_allocation=live.MAX_SATELLITE_ALLOCATION,
                    max_individual_satellite_weight=live.MAX_INDIVIDUAL_SATELLITE_WEIGHT)
    if set(options or {}) - set(settings):
        raise ValueError('Unknown decision setting: ' + ', '.join(sorted(set(options) - set(settings))))
    settings.update(options or {})
    for key, value in settings.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f'{key} must be finite.')
    for key in ('regime_bl_strength', 'max_regime_view_confidence', 'tactical_blend',
                'minimum_trade_confidence', 'max_satellite_allocation', 'max_individual_satellite_weight',
                'bootstrap_mean_scale', 'tight_conditions_tilt_scale', 'elevated_stress_tilt_scale', 'severe_stress_tilt_scale'):
        if not 0 <= settings[key] <= 1:
            raise ValueError(f'{key} must be in [0,1].')
    if settings['max_regime_view_confidence'] >= 1:
        raise ValueError('Tactical BL confidence must be below one.')
    for key in ('regime_horizon', 'minimum_regime_samples', 'sample_confidence_target',
                'minimum_history', 'covariance_lookback', 'bootstrap_block_days'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if settings['minimum_history'] < 2 or settings['covariance_lookback'] < settings['minimum_history']:
        raise ValueError('Require covariance_lookback >= minimum_history >= 2.')
    repetitions = settings['robustness_repetitions']
    if not isinstance(repetitions, int) or not 0 <= repetitions <= 300:
        raise ValueError('Robustness repetitions must be between 0 and 300.')
    for key in ('max_annual_regime_adjustment', 'regime_return_dispersion_scale',
                'cost_benefit_horizon_years', 'agreement_dispersion_scale'):
        if settings[key] <= 0:
            raise ValueError(f'{key} must be positive.')
    if settings['rebalance_threshold'] < 0 or not 0 < settings['tail_confidence'] < 1:
        raise ValueError('Invalid rebalance threshold or tail confidence.')
    for prefix in ('agreement', 'confidence'):
        if not 0 <= settings[f'{prefix}_moderate'] <= settings[f'{prefix}_high'] <= 1:
            raise ValueError(f'Invalid {prefix} label thresholds.')
    return settings


def resolve_allocation_policy(strategic_allocation: dict, current_weights: dict,
                              max_satellite_allocation: float, max_individual_satellite_weight: float):
    """Never infer strategic weights from holdings. Unknown holdings become zero-prior satellites."""
    policy = copy.deepcopy(strategic_allocation)
    if not policy:
        raise ValueError('An explicit strategic allocation is required.')
    for asset in current_weights:
        if asset not in policy:
            policy[asset] = {'strategic_weight': 0., 'minimum_weight': 0.,
                             'maximum_weight': max_individual_satellite_weight,
                             'tactical_low': 0., 'tactical_high': max_individual_satellite_weight,
                             'role': 'satellite', 'group': live.SATELLITE_GROUPS.get(asset, 'satellite')}
    tickers, strategic, bounds = list(policy), [], []
    satellites = []
    for asset, item in policy.items():
        item.setdefault('role', 'core')
        item.setdefault('group', asset)
        if item['role'] not in ('core', 'satellite'):
            raise ValueError('Allocation roles must be core or satellite.')
        if asset in getattr(live, 'INDIVIDUAL_STOCK_TICKERS', ()) and item['role'] != 'satellite':
            raise ValueError(f'Individual stock {asset} must be configured as a satellite.')
        values = [item[k] for k in ('strategic_weight', 'minimum_weight', 'maximum_weight', 'tactical_low', 'tactical_high')]
        if not np.isfinite(values).all() or any(v < 0 or v > 1 for v in values):
            raise ValueError(f'{asset}: policy weights must be finite fractions in [0,1].')
        if item['minimum_weight'] > item['maximum_weight'] or item['tactical_low'] > item['tactical_high']:
            raise ValueError(f'{asset}: reversed policy bounds.')
        low = max(item['minimum_weight'], item['tactical_low'])
        high = min(item['maximum_weight'], item['tactical_high'])
        if item['role'] == 'satellite':
            satellites.append(asset)
            high = min(high, max_individual_satellite_weight)
        fixed = item.get('fixed_weight')
        if fixed is not None:
            if not np.isfinite(fixed) or not low <= fixed <= high or not np.isclose(fixed, item['strategic_weight']):
                raise ValueError(f'{asset}: fixed weight must match strategy and lie within policy bands.')
            low = high = float(fixed)
        if low > high or not low <= item['strategic_weight'] <= high:
            raise ValueError(f'{asset}: strategic weight outside feasible limits.')
        strategic.append(item['strategic_weight'])
        bounds.append((low, high))
    caps = {'satellites': (satellites, max_satellite_allocation)}
    strategic = np.array(strategic, dtype=float)
    validate_allocation_weights(strategic, tickers, bounds, caps)
    return policy, tickers, strategic, bounds, caps


def weight_regime_expected_returns(estimates: pd.DataFrame, probabilities: dict,
                                  fallback_annual: pd.Series, *, horizon=63,
                                  minimum_samples=5, sample_confidence_target=12,
                                  dispersion_scale=.10) -> dict:
    """Mix all four SHRUNK estimates; sparse estimates shrink further to unconditional.

    Annual arithmetic rate = horizon simple-return mean * trading_days/horizon.
    Raw conditional means are deliberately unused. Missing outcomes use the
    unconditional mean, or the trailing annual mean when no long-run mean exists.
    """
    if set(probabilities) != set(REGIMES) or not np.isfinite(list(probabilities.values())).all() or any(p < 0 for p in probabilities.values()) or not np.isclose(sum(probabilities.values()), 1, atol=1e-8, rtol=0):
        raise ValueError('Four finite nonnegative regime probabilities summing to one are required.')
    if horizon < 1 or minimum_samples < 1 or sample_confidence_target <= 0 or dispersion_scale <= 0:
        raise ValueError('Invalid expected-return settings.')
    estimates = estimates.reindex(columns=['asset', 'horizon', 'regime', 'unconditional_estimate', 'shrunk_estimate', 'sample_count', 'lambda'])
    scale, result = shared.TRADING_DAYS / horizon, {}
    for asset, fallback in fallback_annual.items():
        rows = estimates.loc[(estimates['asset'] == asset) & (estimates['horizon'] == horizon)]
        unconditional_values = rows['unconditional_estimate'].dropna()
        unconditional = float(unconditional_values.iloc[0] * scale) if len(unconditional_values) else float(fallback)
        components, quality = {}, []
        sample_quality = 0.
        for regime in REGIMES:
            selected = rows.loc[rows['regime'] == regime]
            n, annual, shrinkage = 0, unconditional, 0.
            if len(selected):
                row = selected.iloc[0]
                if pd.notna(row['shrunk_estimate']) and np.isfinite(row['shrunk_estimate']):
                    n = int(row['sample_count'])
                    annual = float(row['shrunk_estimate'] * scale)
                    shrinkage = float(row['lambda'])
            trust = min(1., n / minimum_samples)
            annual = unconditional + trust * (annual - unconditional)
            if n < minimum_samples:
                quality.append(f'{regime}: sparse/missing outcomes; unconditional fallback or additional shrinkage')
            components[regime] = {'shrunk_annual_return': annual, 'sample_count': n,
                                  'phase2_lambda': shrinkage, 'sparse_sample_retention': trust}
            sample_quality += probabilities[regime] * min(1., n / sample_confidence_target)
        weighted = sum(probabilities[r] * components[r]['shrunk_annual_return'] for r in REGIMES)
        dispersion = np.sqrt(sum(probabilities[r] * (components[r]['shrunk_annual_return'] - weighted) ** 2 for r in REGIMES))
        result[asset] = {'unconditional': unconditional, 'weighted_regime': weighted, 'delta': weighted - unconditional,
                         'regime_components': components, 'probabilities': dict(probabilities),
                         'sample_quality': sample_quality, 'estimate_stability': 1 / (1 + dispersion / dispersion_scale),
                         'annualization': f'linear arithmetic scaling: {shared.TRADING_DAYS}/{horizon}; not CAGR',
                         'unconditional_source': 'completed non-overlapping outcomes' if len(unconditional_values) else 'trailing annual mean fallback',
                         'data_quality': quality, 'horizon': horizon}
    return result


def build_regime_bl_views(prior: pd.Series, regime_returns: dict, regime: dict, *,
                          strength=.5, max_adjustment=.025, max_confidence=.25) -> dict:
    """Modest prior-relative annual views; quality and dispersion discount evidence."""
    if not 0 <= strength <= 1 or not 0 <= max_confidence < 1 or max_adjustment <= 0:
        raise ValueError('Invalid tactical view controls.')
    result = {}
    coverage = min(regime.get('coverage', {}).get('growth', 0), regime.get('coverage', {}).get('inflation', 0))
    for asset, value in prior.items():
        source = regime_returns[asset]
        evidence = float(regime['confidence']) * coverage * source['sample_quality'] * source['estimate_stability']
        raw_adjustment = strength * evidence * source['delta']
        adjustment = float(np.clip(raw_adjustment, -max_adjustment, max_adjustment))
        result[asset] = {'prior_expected_return': float(value), 'regime_weighted_expected_return': source['weighted_regime'],
                         'raw_regime_delta': source['delta'], 'adjustment': adjustment,
                         'adjusted_bl_view': float(value + adjustment), 'confidence': min(max_confidence, evidence),
                         'effective_regime_evidence': evidence, 'cap_applied': abs(raw_adjustment) > max_adjustment,
                         'source': 'regime', 'sample_quality': source['sample_quality'],
                         'estimate_stability': source['estimate_stability'], 'horizon': source['horizon']}
    return result


def optimizer_consensus(candidates: dict[str, np.ndarray], tickers: list[str],
                        strategic: np.ndarray, dispersion_scale=.025,
                        high_threshold=.75, moderate_threshold=.40) -> dict:
    """Agreement is a dispersion heuristic; optimizers are not independent forecasts."""
    if not candidates or dispersion_scale <= 0:
        raise ValueError('Consensus requires candidates and positive dispersion scale.')
    matrix = np.stack(list(candidates.values()))
    per_asset = {}
    for i, asset in enumerate(tickers):
        values = matrix[:, i]
        sd = float(values.std(ddof=0))
        per_asset[asset] = {'strategic_weight': float(strategic[i]),
                           **{method: float(weights[i]) for method, weights in candidates.items()},
                           'mean': float(values.mean()), 'median': float(np.median(values)),
                           'minimum': float(values.min()), 'maximum': float(values.max()),
                           'standard_deviation': sd, 'agreement_score': float(np.exp(-sd / dispersion_scale))}
    score = float(np.mean([item['agreement_score'] for item in per_asset.values()]))
    if len(candidates) < 2:
        score = 0.  # One remaining method supplies no agreement evidence.
    return {'per_asset': per_asset, 'agreement_score': score,
            'agreement_label': 'high' if score >= high_threshold else 'medium' if score >= moderate_threshold else 'low',
            'methods_used': list(candidates), 'interpretation': 'dispersion heuristic, not independent statistical forecasts'}


def resampled_weight_stability(returns: pd.DataFrame, mu: pd.Series, strategic: np.ndarray,
                               bounds, caps, settings: dict) -> tuple[pd.DataFrame, dict]:
    """Moving-block bootstrap sensitivity of BL Sharpe, not regime refitting or forecast calibration."""
    rng = np.random.default_rng(settings['seed'])
    tickers = list(mu.index)
    baseline_mean = returns.mean() * shared.TRADING_DAYS
    block = min(settings['bootstrap_block_days'], len(returns))
    samples, failures = [], []
    for _ in range(settings['robustness_repetitions']):
        starts = rng.integers(0, len(returns) - block + 1, size=int(np.ceil(len(returns) / block)))
        positions = np.concatenate([np.arange(start, start + block) for start in starts])[:len(returns)]
        draw = returns.iloc[positions].reset_index(drop=True)
        try:
            mean, covariance = annualize_mean_cov(draw, shared.TRADING_DAYS, shared.USE_COV_SHRINKAGE, shared.COV_SHRINKAGE_METHOD)
            perturbed_mu = mu + settings['bootstrap_mean_scale'] * (mean - baseline_mean)
            weights = optimize_max_sharpe_constrained(perturbed_mu, covariance, settings['resolved_rf'], tickers,
                                                     bounds=bounds, max_group_weights=caps)
            validate_allocation_weights(weights, tickers, bounds, caps)
            samples.append(weights)
        except (ValueError, RuntimeError, OptimizationError, SolverError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
    frame = pd.DataFrame(samples, columns=tickers, dtype=float)
    per_asset = {}
    for i, asset in enumerate(tickers):
        values = frame[asset]
        sd = float(values.std(ddof=0)) if len(values) else np.nan
        width = bounds[i][1] - bounds[i][0]
        score = float(np.exp(-sd / max(width, 1e-6))) if len(values) else 0.
        per_asset[asset] = {'median': values.median(), 'q25': values.quantile(.25), 'q75': values.quantile(.75),
                            'q10': values.quantile(.10), 'q90': values.quantile(.90),
                            'standard_deviation': sd, 'weight_stability_score': score}
    moving = [per_asset[t]['weight_stability_score'] for i, t in enumerate(tickers) if bounds[i][1] > bounds[i][0]]
    success_rate = len(samples) / settings['robustness_repetitions'] if settings['robustness_repetitions'] else 0.
    overall = float(np.mean(moving) * success_rate) if moving else success_rate
    return frame, {'per_asset': per_asset, 'overall_score': overall, 'requested_repetitions': settings['robustness_repetitions'],
                   'successful_repetitions': len(samples), 'failed_repetitions': len(failures),
                   'failure_examples': list(dict.fromkeys(failures))[:3],
                   'method': 'moving-block return bootstrap; perturbed annual mean and re-estimated covariance; BL Sharpe only'}


def _portfolio_analysis(weights, tickers, mu, cov, returns, rf, policy, tail_confidence):
    metrics = portfolio_stats(weights, mu, cov, rf)
    hhi = float(weights @ weights)
    metrics.update(hhi=hhi, effective_holdings=1 / hhi)
    if metrics['volatility'] > 0:
        contributions = risk_contribution(tickers, weights, cov).set_index('Ticker')
        risk = {asset: {'marginal': float(contributions.loc[asset, 'Marginal Risk Contribution']),
                        'component': float(contributions.loc[asset, 'Component Risk Contribution']),
                        'percentage': float(contributions.loc[asset, 'Percent Risk Contribution'])} for asset in tickers}
    else:
        risk = {asset: {'marginal': None, 'component': 0., 'percentage': None} for asset in tickers}
    groups = {}
    for i, asset in enumerate(tickers):
        group = policy[asset]['group']
        item = groups.setdefault(group, {'weight': 0., 'risk_contribution': 0., 'percentage_risk_contribution': 0.})
        item['weight'] += float(weights[i])
        item['risk_contribution'] += risk[asset]['component']
        item['percentage_risk_contribution'] += risk[asset]['percentage'] or 0.
    series = build_portfolio_return_series(returns, weights, tickers)
    return {'weights': dict(zip(tickers, weights)), 'metrics': metrics, 'risk_contributions': risk,
            'groups': groups, 'group_hhi': sum(v['weight'] ** 2 for v in groups.values()),
            'role_weights': {role: sum(float(weights[i]) for i, t in enumerate(tickers) if policy[t]['role'] == role) for role in ('core', 'satellite')},
            'historical_tail_risk': historical_tail_risk(series, tail_confidence, shared.TRADING_DAYS)}


def build_tactical_allocation(strategic, tickers, bounds, caps, consensus, stability,
                              confidence, regime, settings):
    """Convex strategy anchor plus confidence-discounted feasible consensus tilt."""
    median = np.array([consensus['per_asset'][t]['median'] for t in tickers])
    feasible_median = project_allocation(median, tickers, bounds, strategic, caps)
    blend = settings['tactical_blend'] * confidence * min(consensus['agreement_score'], stability['overall_score'])
    if regime['overlays']['financial_conditions'] == 'tight':
        blend *= settings['tight_conditions_tilt_scale']
    if regime['overlays']['stress'] in ('elevated', 'severe'):
        blend *= settings['severe_stress_tilt_scale'] if regime['overlays']['stress'] == 'severe' else settings['elevated_stress_tilt_scale']
    central = strategic + blend * (feasible_median - strategic)
    validate_allocation_weights(central, tickers, bounds, caps)
    ranges = {}
    for i, asset in enumerate(tickers):
        item = stability['per_asset'][asset]
        q25 = item['q25'] if pd.notna(item['q25']) else central[i]
        q75 = item['q75'] if pd.notna(item['q75']) else central[i]
        ranges[asset] = [float(max(bounds[i][0], min(strategic[i], central[i], q25))),
                         float(min(bounds[i][1], max(strategic[i], central[i], q75)))]
    return central, ranges, blend


def implementation_rebalance_decision(current, target, tickers, bounds, caps, ranges,
                                      mu, cov, confidence, settings, gain_rates=None) -> dict:
    """Decision labels only, not executable trade quantities. Costs assume a full rebalance.

    Discretionary changes require model CE benefit over the configured horizon
    to exceed one-time costs/taxes. Hard mandate violations require review even
    if confidence is low. No lot accounting or guaranteed cost-benefit inference.
    """
    gross = float(target @ mu.values)
    estimated = net_expected_return_after_all_costs(gross, current, target,
        shared.TRANSACTION_COST_BPS, shared.SLIPPAGE_BPS, gain_rates,
        shared.SHORT_TERM_TAX_RATE, shared.LONG_TERM_TAX_RATE, shared.LONG_TERM_FRACTION)
    annual_benefit = float((target - current) @ mu.values - shared.BL_RISK_AVERSION / 2 *
                           (target @ cov.values @ target - current @ cov.values @ current))
    benefit = max(0., annual_benefit) * settings['cost_benefit_horizon_years'] * confidence
    violations = [not bounds[i][0] - 1e-6 <= current[i] <= bounds[i][1] + 1e-6 for i in range(len(tickers))]
    group_violation = any(sum(current[tickers.index(t)] for t in names) > cap + 1e-6 for names, cap in caps.values())
    mandatory = any(violations) or group_violation
    actions = {}
    for i, asset in enumerate(tickers):
        delta = target[i] - current[i]
        low, high = ranges[asset]
        reasons = []
        required = violations[i] or (group_violation and delta < 0 and asset in caps['satellites'][0])
        if required:
            action = 'ADD' if delta > 0 else 'TRIM' if delta < 0 else 'HOLD'
            if violations[i]:
                reasons.append('strategic_band_limit')
            if asset in caps['satellites'][0] and (group_violation or current[i] > bounds[i][1]):
                reasons.append('concentration_limit')
        elif abs(delta) < settings['rebalance_threshold']:
            action = 'HOLD'
            reasons.append('within_no_trade_band')
        elif low - 1e-6 <= current[i] <= high + 1e-6:
            action = 'HOLD'
            reasons.append('within_recommended_range')
        elif confidence < settings['minimum_trade_confidence']:
            action = 'HOLD'
            reasons.append('low_recommendation_confidence')
        elif not mandatory and benefit <= estimated['implementation_cost'] + estimated['tax_drag']:
            action = 'HOLD'
            reasons.append('transaction_cost_not_justified' if benefit <= estimated['implementation_cost'] else 'tax_cost_not_justified')
        else:
            action = 'ADD' if delta > 0 else 'TRIM'
        actions[asset] = {'action': action, 'required': bool(required), 'current_weight': float(current[i]),
                          'central_weight': float(target[i]), 'recommended_range': ranges[asset], 'reason_codes': reasons}
    any_change = any(item['action'] != 'HOLD' for item in actions.values())
    return {'actions': actions, 'portfolio_action': 'REBALANCE' if any_change else 'HOLD',
            'constraint_repair_required': bool(mandatory),
            'estimated_turnover': estimated['turnover'] if any_change else 0.,
            'estimated_transaction_cost': estimated['implementation_cost'] if any_change else 0.,
            'estimated_tax_drag': estimated['tax_drag'] if any_change else 0.,
            'prospective_full_rebalance': estimated, 'model_annual_ce_improvement': annual_benefit,
            'confidence_discounted_horizon_benefit': benefit,
            'notes': 'Approximate one-time costs/taxes; two-sided traded notional. Action labels are not a self-financing order list; prospective estimates assume the full central portfolio.'}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat() if pd.notna(value) else None
    return value


def build_portfolio_recommendation(prices: pd.DataFrame, yields: pd.DataFrame,
                                   current_weights: dict | None = None,
                                   strategic_allocation: dict | None = None, *, as_of=None,
                                   manual_views: dict | None = None, relative_views: list | None = None,
                                   risk_free_rate: float | None = None, options: dict | None = None) -> dict:
    """Primary Phase 4 API. Pure financial computation; no network or UI calls.

    Required price columns are all policy and held tickers. Covariance/tail risk
    use one complete-case trailing panel; missing required assets fail explicitly.
    Regime history/outcomes use the full supplied history up to the same cutoff.
    Default policy is config.live. All amounts are fractions, annual returns are
    arithmetic model rates, tails are empirical daily losses. Detailed floats
    retain budget consistency; display_weights_percent avoids presentation precision.
    """
    settings = _settings(options)
    notes = ['Heuristic regime probabilities/confidence; conditional outcomes are descriptive, not calibrated forecasts.',
             'Treasury lag uses observation dates, not release timestamps or vintage data.',
             'Manual opinions are supplied as-of assumptions, not a dated historical view archive.',
             'Historical portfolio risk assumes constant daily weights, excluding realized costs/taxes.']
    holdings = dict(live.CURRENT_WEIGHTS if current_weights is None else current_weights)
    if not holdings or not np.isfinite(list(holdings.values())).all() or any(w < 0 for w in holdings.values()) or not np.isclose(sum(holdings.values()), 1, atol=1e-8, rtol=0):
        raise ValueError('Current long-only weights must be finite and sum to one; cash must be explicit.')
    if strategic_allocation is None:
        allocation = copy.deepcopy(live.STRATEGIC_ALLOCATION)
        notes.append('Default core strategy is illustrative configuration, not inferred from holdings; customize the mandate.')
    else:
        allocation = strategic_allocation
    policy, tickers, strategic, bounds, caps = resolve_allocation_policy(allocation, holdings,
        settings['max_satellite_allocation'], settings['max_individual_satellite_weight'])
    prices, yields = _frame(prices, as_of), _frame(yields, as_of)
    missing = [t for t in tickers if t not in prices or prices[t].dropna().empty]
    if missing:
        raise ValueError('Missing required portfolio price series: ' + ', '.join(missing))
    if (prices[tickers] <= 0).any().any():
        raise ValueError('Portfolio prices must be positive.')
    returns = compute_returns(prices[tickers]).dropna().tail(settings['covariance_lookback'])
    if len(returns) < settings['minimum_history']:
        raise ValueError('Insufficient common portfolio history; no assets are silently dropped.')
    date = returns.index[-1]
    query_date = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else prices.index.max()
    if (query_date - date).days > shared.REGIME_MAX_AGE_DAYS:
        raise ValueError('Common portfolio observations are stale; recommendation cannot be evaluated reliably.')
    hist_mu, cov = annualize_mean_cov(returns, shared.TRADING_DAYS, shared.USE_COV_SHRINKAGE, shared.COV_SHRINKAGE_METHOD)
    if not np.isfinite(cov.to_numpy()).all() or np.trace(cov) <= 0:
        raise ValueError('Portfolio covariance must contain finite positive risk.')
    rf_info = resolve_risk_free_rate(as_of=date, yields=yields, override=risk_free_rate)
    rf = rf_info['rate']
    settings['resolved_rf'] = rf
    regime = get_current_regime(prices, yields, as_of=date, **settings['regime_model_options'])
    history = build_regime_history(prices, yields, feature_options={'as_of': date}, **settings['regime_model_options'])
    estimates = estimate_regime_expected_returns(history, prices, horizons=(settings['regime_horizon'],),
                                                 as_of=date, assets=tuple(tickers))
    weighted = weight_regime_expected_returns(estimates, regime['probabilities'], hist_mu,
        horizon=settings['regime_horizon'], minimum_samples=settings['minimum_regime_samples'],
        sample_confidence_target=settings['sample_confidence_target'], dispersion_scale=settings['regime_return_dispersion_scale'])
    prior = implied_equilibrium_returns(cov, strategic, shared.BL_RISK_AVERSION, rf)
    tactical_views = build_regime_bl_views(prior, weighted, regime, strength=settings['regime_bl_strength'],
                                         max_adjustment=settings['max_annual_regime_adjustment'],
                                         max_confidence=settings['max_regime_view_confidence'])
    manual = {t: v for t, v in live.BL_ABSOLUTE_VIEWS.items() if t in tickers} if manual_views is None else manual_views
    relative = [v for v in live.BL_RELATIVE_VIEWS if v[0] in tickers and v[1] in tickers] if relative_views is None else relative_views
    if any(t not in tickers for t in manual) or any(a not in tickers or b not in tickers for a, b, _, _ in relative):
        raise ValueError('Manual views must reference the decision universe.')
    omitted_manual = {t: v for t, v in live.BL_ABSOLUTE_VIEWS.items() if t not in tickers} if manual_views is None else {}
    omitted_relative = [v for v in live.BL_RELATIVE_VIEWS if v[0] not in tickers or v[1] not in tickers] if relative_views is None else []
    if omitted_manual or omitted_relative:
        notes.append('Configured manual views outside the decision universe are retained as excluded metadata, not applied.')
    merged, view_sources = merge_manual_regime_views(manual, tactical_views)
    mu, _ = black_litterman_posterior(cov, strategic, shared.BL_RISK_AVERSION, shared.BL_TAU,
                                      merged, relative, rf)
    current = np.array([holdings.get(t, 0.) for t in tickers])
    candidates = {'strategic': strategic}
    status = {'strategic': 'configured'}
    methods = {'minimum_variance': lambda: optimize_min_vol(mu, cov, bounds=bounds, tickers=tickers, max_group_weights=caps),
               'bl_max_sharpe': lambda: optimize_max_sharpe_constrained(mu, cov, rf, tickers, bounds=bounds, max_group_weights=caps),
               'risk_parity': lambda: optimize_risk_parity(cov, bounds, strategic, caps)}
    if settings['enable_cvar']:
        methods['cvar'] = lambda: optimize_min_cvar(mu, returns, bounds, settings['tail_confidence'], caps)
    for name, optimizer in methods.items():
        try:
            weights = optimizer()
            validate_allocation_weights(weights, tickers, bounds, caps)
            candidates[name] = weights
            status[name] = 'optimized'
        except (ValueError, RuntimeError, OptimizationError, SolverError, np.linalg.LinAlgError) as exc:
            status[name] = 'unavailable'
            notes.append(f'{name} unavailable: {exc}')
    consensus = optimizer_consensus(candidates, tickers, strategic, settings['agreement_dispersion_scale'],
                                    settings['agreement_high'], settings['agreement_moderate'])
    _, stability = resampled_weight_stability(returns, mu, strategic, bounds, caps, settings)
    if stability['failed_repetitions'] or not stability['successful_repetitions']:
        notes.append('Robustness has failed/disabled runs; confidence is discounted. ' + '; '.join(stability['failure_examples']))
    regime_quality = min(regime['coverage']['growth'], regime['coverage']['inflation'])
    sample_quality = float(np.mean([v['sample_quality'] for v in weighted.values()]))
    components = {'regime': regime['confidence'], 'data_quality': regime_quality, 'conditional_sample_quality': sample_quality,
                  'optimizer_agreement': consensus['agreement_score'], 'weight_stability': stability['overall_score'],
                  'optimizer_success': (len(candidates) - 1) / len(methods)}
    # Equal static component weights; candidate failure discount prevents apparent
    # agreement among a reduced set from being treated as robust evidence.
    overall = float(np.mean(list(components.values())) * components['optimizer_success'])
    central, ranges, blend = build_tactical_allocation(strategic, tickers, bounds, caps, consensus, stability, overall, regime, settings)
    widths = np.array([hi - lo for lo, hi in bounds])
    movable = widths > 0
    anchor_distance = float(np.mean(np.abs(central[movable] - strategic[movable]) / widths[movable])) if movable.any() else 0.
    components['strategic_anchor'] = 1 - min(1., anchor_distance)
    overall *= components['strategic_anchor']
    # Recompute once using the distance-discounted confidence (always moves closer
    # to the same strategic anchor and cannot increase distance).
    central, ranges, blend = build_tactical_allocation(strategic, tickers, bounds, caps, consensus, stability, overall, regime, settings)
    confidence = {'overall_score': overall, 'aggregation': 'mean(regime,data_quality,conditional_sample_quality,optimizer_agreement,weight_stability,optimizer_success) * optimizer_success * provisional strategic_anchor', 'label': 'high' if overall >= settings['confidence_high'] else 'moderate' if overall >= settings['confidence_moderate'] else 'low',
                  'components': components, 'interpretation': 'heuristic evidence/robustness score, not statistical certainty'}
    gains = np.full(len(tickers), shared.DEFAULT_UNREALIZED_GAIN_RATE) if shared.APPLY_TAX_EFFECTS else None
    rebalance = implementation_rebalance_decision(current, central, tickers, bounds, caps, ranges, mu, cov, overall, settings, gains)
    reasons = {}
    for i, asset in enumerate(tickers):
        codes = list(rebalance['actions'][asset]['reason_codes'])
        delta, tilt = weighted[asset]['delta'], central[i] - strategic[i]
        if abs(tilt) > 1e-8:
            codes += ['positive_regime_return_delta' if delta > 0 else 'negative_regime_return_delta' if delta < 0 else 'no_regime_return_delta',
                      'optimizer_consensus_positive' if consensus['per_asset'][asset]['median'] > strategic[i] else 'optimizer_consensus_negative',
                      'within_tactical_band']
            if (mu[asset] - prior[asset]) * tilt > 0:
                codes.append('black_litterman_support')
        if consensus['per_asset'][asset]['agreement_score'] < settings['agreement_moderate']:
            codes.append('optimizer_disagreement')
        if stability['per_asset'][asset]['weight_stability_score'] < settings['confidence_moderate']:
            codes.append('unstable_optimizer_weight')
        if regime['overlays']['financial_conditions'] == 'tight':
            codes.append('tight_financial_conditions')
        if regime['overlays']['stress'] in ('elevated', 'severe'):
            codes.append('elevated_stress')
        reasons[asset] = list(dict.fromkeys(codes or ['strategic_anchor']))
    analysis = lambda w: _portfolio_analysis(w, tickers, mu, cov, returns, rf, policy, settings['tail_confidence'])
    recommended = analysis(central)
    recommended.update(central_weights=recommended['weights'], weight_ranges=ranges, confidence=confidence,
                        tactical_blend_applied=blend, tactical_adjustments=dict(zip(tickers, central - strategic)),
                        display_weights_percent={t: round(float(w) * 100, 1) for t, w in zip(tickers, central)},
                        display_rounding_note='Display percentages are rounded independently and may not sum to 100; central_weights retain exact budget consistency.')
    notes.extend(regime['data_quality'])
    if any(v['data_quality'] for v in weighted.values()):
        notes.append('Some conditional outcomes are sparse; unconditional fallbacks/additional shrinkage applied per asset.')
    if shared.APPLY_TAX_EFFECTS:
        notes.append('Tax drag is approximate: assumed gains as a fraction of market value and holding periods; no tax lots.')
    result = {'as_of': date.isoformat(), 'requested_as_of': query_date.isoformat(), 'regime': regime,
              'risk_free_rate': rf_info, 'current_portfolio': analysis(current), 'strategic_portfolio': analysis(strategic),
              'expected_returns': {'source': 'Black–Litterman posterior with configured manual views and existing regime overlay', 'prior': prior.to_dict(), 'regime_weighted': {t: v['weighted_regime'] for t, v in weighted.items()},
                                   'regime_delta': {t: v['delta'] for t, v in weighted.items()}, 'per_asset': weighted,
                                   'black_litterman_posterior': mu.to_dict(), 'regime_bl_views': tactical_views,
                                   'merged_absolute_views': merged, 'view_sources': view_sources, 'manual_relative_views': relative,
                                   'excluded_manual_views': omitted_manual, 'excluded_relative_views': omitted_relative},
              'candidate_allocations': {name: dict(status=state, **analysis(candidates[name])) if name in candidates else {'status': state, 'weights': None} for name, state in status.items()},
              'optimizer_consensus': consensus, 'stability': stability, 'recommended_portfolio': recommended,
              'rebalance': rebalance, 'reason_codes': reasons, 'policy': policy,
              'asset_correlation': {'method': 'Pearson correlation of daily simple returns',
                                    'observations': len(returns),
                                    'sample_start': returns.index[0].isoformat(),
                                    'sample_end': returns.index[-1].isoformat(),
                                    'matrix': returns.corr(min_periods=2).to_dict()},
              'constraints': {'asset_bounds': dict(zip(tickers, bounds)), 'max_satellite_allocation': settings['max_satellite_allocation'],
                              'max_individual_satellite_weight': settings['max_individual_satellite_weight']},
              'data_policy': {'covariance_method': shared.COV_SHRINKAGE_METHOD if shared.USE_COV_SHRINKAGE else 'sample',
                              'common_observations': len(returns), 'risk_sample_start': returns.index[0].isoformat(),
                              'missing_data': 'complete-case common risk panel; no silent asset exclusion',
                              'regime_outcomes': 'completed non-overlapping intervals up to as_of'},
              'warnings': list(dict.fromkeys(notes))}
    return _json_safe(result)


def main():
    """Optional CLI: python main_live.py --decision, or python portfolio_decision.py."""
    import argparse
    from market_intelligence import load_market_data
    from data import download_prices
    parser = argparse.ArgumentParser(description='Phase 3 portfolio decision JSON (no execution)')
    parser.add_argument('--decision', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--start', default='2007-01-01')
    parser.add_argument('--as-of', default=None)
    parser.add_argument('--robustness-repetitions', type=int, default=shared.DECISION_SETTINGS['robustness_repetitions'])
    args = parser.parse_args()
    data = load_market_data(args.start)
    needed = list(dict.fromkeys(list(live.STRATEGIC_ALLOCATION) + list(live.CURRENT_WEIGHTS)))
    extra = [t for t in needed if t not in data['prices']]
    prices = data['prices']
    if extra:
        prices = prices.join(download_prices(extra, args.start), how='outer')
    recommendation = build_portfolio_recommendation(prices, data['yields'], as_of=args.as_of,
        options={'robustness_repetitions': args.robustness_repetitions})
    recommendation['warnings'].extend(data['data_quality'])
    print(json.dumps(recommendation, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
