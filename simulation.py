"""Forward path simulation and compact research summaries; no backtest calibration."""


from __future__ import annotations
import numpy as np
import pandas as pd


def terminal_value_stats(paths: pd.DataFrame) -> dict:
    """
    Summary statistics for ending portfolio values.
    """
    terminal = paths.iloc[-1]
    initial = paths.attrs.get("initial_value", 1.0)

    return {
        "mean_terminal_value": float(terminal.mean()),
        "median_terminal_value": float(terminal.median()),
        "p5_terminal_value": float(terminal.quantile(0.05)),
        "p25_terminal_value": float(terminal.quantile(0.25)),
        "p75_terminal_value": float(terminal.quantile(0.75)),
        "p95_terminal_value": float(terminal.quantile(0.95)),
        "prob_loss": float((terminal < initial).mean()),
        "prob_double": float((terminal >= 2 * initial).mean()),
        "prob_5x": float((terminal >= 5 * initial).mean()),
    }


def path_max_drawdown(path: pd.Series, initial_value: float = 1.0) -> float:
    """
    Max drawdown for one simulated portfolio path.
    """
    peak = path.cummax().clip(lower=initial_value)
    drawdown = path / peak - 1.0
    return float(drawdown.min())


def drawdown_stats(paths: pd.DataFrame) -> dict:
    """
    Summary stats for max drawdowns across all simulations.
    """
    drawdowns = paths.apply(path_max_drawdown, axis=0, initial_value=paths.attrs.get("initial_value", 1.0))

    return {
        "mean_max_drawdown": float(drawdowns.mean()),
        "median_max_drawdown": float(drawdowns.median()),
        "p95_worst_drawdown": float(drawdowns.quantile(0.05)),
        "prob_drawdown_30": float((drawdowns <= -0.30).mean()),
        "prob_drawdown_50": float((drawdowns <= -0.50).mean()),
        "prob_drawdown_60": float((drawdowns <= -0.60).mean()),
    }

SIMULATION_METHODS = ('gaussian', 'student_t', 'block_bootstrap', 'regime_switching')


def synchronized_block_indices(length, horizon, paths, block_length, rng):
    """One source row per path/time shared by ALL assets; no independent asset sampling."""
    if min(length, horizon, paths, block_length) < 1 or block_length > length:
        raise ValueError('Positive dimensions and block_length <= history length required.')
    starts = rng.integers(0, length - block_length + 1, size=(int(np.ceil(horizon / block_length)), paths))
    return (starts[:, None, :] + np.arange(block_length)[None, :, None]).reshape(-1, paths)[:horizon]


def simulation_statistics(terminal, drawdowns, initial_value, target_value=None):
    terminal, dd = np.asarray(terminal), np.asarray(drawdowns)
    return {'mean_terminal_value': float(np.mean(terminal)), 'median_terminal_value': float(np.median(terminal)),
            **{f'p{p}_terminal_value': float(np.percentile(terminal, p)) for p in (5, 25, 75, 95)},
            'prob_loss': float(np.mean(terminal < initial_value)), 'prob_double': float(np.mean(terminal >= 2 * initial_value)),
            'prob_goal': float(np.mean(terminal >= target_value)) if target_value is not None else None,
            'mean_max_drawdown': float(np.mean(dd)), 'median_max_drawdown': float(np.median(dd)),
            'p95_worst_drawdown': float(np.quantile(dd, .05)),
            'prob_drawdown_30': float(np.mean(dd <= -.30)), 'prob_drawdown_50': float(np.mean(dd <= -.50))}


def run_simulation(weights, mu, cov, *, method='gaussian', historical_returns=None,
                   regime_history=None, as_of=None, starting_regime='Current', scenario='Baseline',
                   years=10, n_sims=2000, trading_days=252, initial_value=10000., target_value=None,
                   annual_contribution=0., annual_withdrawal=0., contribution_growth=0., inflation=0.,
                   rebalance_frequency='monthly', transaction_cost_bps=0., degrees_of_freedom=5.,
                   block_length=21, include_estimation_uncertainty=False, seed=42,
                   min_regime_samples=30, return_paths=False):
    """Common long-only simulation interface with streaming percentile summaries.

    Target weights drift between rebalances. Cash flows occur AFTER the final
    session of each model year: contribution (grown annually) then withdrawal
    (inflation-indexed). Flows are allocated/removed proportionately, not used to
    rebalance for free. Costs are charged on actual two-sided turnover. Inflation
    deflates nominal values for an additional real fan chart. Drawdown is measured
    on unitized investment performance (costs included; cash flows excluded).
    Negative simple-return draws below -100% are floored at -100% and counted.
    Portfolio values cannot become negative. Underfunded withdrawals are reported.
    """
    from regimes import REGIMES, estimate_transition_matrix, scenario_transition_matrix
    from mpt import annualize_mean_cov
    if method not in SIMULATION_METHODS:
        raise ValueError('Unknown simulation method.')
    scalars = [years, n_sims, trading_days, initial_value, annual_contribution, annual_withdrawal,
               contribution_growth, inflation, transaction_cost_bps, degrees_of_freedom, block_length]
    if not np.isfinite(scalars).all() or years <= 0 or n_sims < 1 or int(n_sims) != n_sims or trading_days < 1 or int(trading_days) != trading_days:
        raise ValueError('Finite positive horizon and integer path/session counts required.')
    if initial_value <= 0 or min(annual_contribution, annual_withdrawal, transaction_cost_bps) < 0 or min(inflation, contribution_growth) <= -1:
        raise ValueError('Invalid initial value, cash flows, growth, inflation or costs.')
    if degrees_of_freedom <= 2 or block_length < 1 or int(block_length) != block_length or min_regime_samples < 1:
        raise ValueError('Student-t df must exceed 2; block and minimum sample sizes must be positive.')
    if transaction_cost_bps * 2 >= 10000 or target_value is not None and (not np.isfinite(target_value) or target_value < 0):
        raise ValueError('Invalid transaction cost or goal.')
    if rebalance_frequency not in ('daily', 'monthly', 'quarterly', 'annual', 'none'):
        raise ValueError('Unknown simulation rebalance frequency.')
    n_sims, trading_days, block_length = int(n_sims), int(trading_days), int(block_length)
    steps = int(round(years * trading_days))
    if steps < 1 or steps > 252 * 100 or n_sims > 100000:
        raise ValueError('Horizon must be 1–25,200 observations; at most 100,000 paths.')
    weights = np.asarray(weights, dtype=float)
    assets = list(cov.index)
    covariance, mean = cov.to_numpy(dtype=float), np.asarray(mu, dtype=float)
    if list(cov.columns) != assets or covariance.shape != (len(weights), len(weights)) or mean.shape != weights.shape:
        raise ValueError('Moment dimensions and covariance labels must match weights.')
    if isinstance(mu, pd.Series) and not isinstance(mu.index, pd.RangeIndex) and list(mu.index) != assets:
        raise ValueError('Expected-return labels must match covariance order.')
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1) or not np.isfinite(mean).all():
        raise ValueError('Long-only finite weights must sum to one; means must be finite.')
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError('Covariance must be finite, symmetric and positive semidefinite.')
    history = None
    if historical_returns is not None:
        if not historical_returns.index.is_unique or not historical_returns.index.is_monotonic_increasing:
            raise ValueError('History must have unique increasing dates.')
        if any(t not in historical_returns for t in assets):
            raise ValueError('Historical returns must contain all assets; no substitutions.')
        history = historical_returns.loc[:as_of, assets] if as_of is not None else historical_returns[assets]
        history = history.dropna()
        if len(history) < 2 or not np.isfinite(history).all().all() or (history <= -1).any().any():
            raise ValueError('At least two finite synchronized historical returns above -100% required.')
    if method in ('block_bootstrap', 'regime_switching') or include_estimation_uncertainty:
        if history is None:
            raise ValueError('This method requires historical returns.')
        if block_length > len(history):
            raise ValueError('Block length exceeds available history.')
    rng = np.random.default_rng(seed)
    metadata = {'method': method, 'seed': seed, 'steps': steps, 'paths': n_sims,
        'as_of': str(as_of), 'degrees_of_freedom': degrees_of_freedom if method == 'student_t' else None,
        'block_length': block_length, 'rebalance_frequency': rebalance_frequency,
        'cash_flow_timing': 'End of each model year, contribution then withdrawal; contributions grow, withdrawals inflation-indexed; flows proportional to holdings.',
        'annualization': f'{trading_days} observations/year; monthly={max(1, trading_days//12)}, quarterly={max(1, trading_days//4)} observations.',
        'drawdown_basis': 'Unitized investment performance including costs, excluding external flows; insolvent investment paths retain -100% drawdown.',
        'probabilities': 'Model-dependent fractions of simulated paths, not calibrated forecasts or guarantees; nominal terminal goals include cash flows.',
        'taxes': 'Not modeled; no tax-lot information.', 'include_estimation_uncertainty': include_estimation_uncertainty,
        'historical_rows': len(history) if history is not None else 0,
        'assumptions': 'Long-only simple returns; parametric draws below -100% floored and reported; no borrowing.'}
    # Modest number of parameter populations, reused by groups of paths.
    groups = min(32, n_sims) if include_estimation_uncertainty else 1
    means, roots, pools, pool_labels = [], [], [], []
    labels = None
    transition = None
    if method == 'regime_switching':
        if regime_history is None:
            raise ValueError('Regime history is required.')
        cutoff = history.index[-1] if as_of is None else pd.Timestamp(as_of)
        regime_history = regime_history.loc[:cutoff]
        base, transition_info = estimate_transition_matrix(regime_history, cutoff)
        transition, scenario_info = scenario_transition_matrix(base, scenario)
        # Regime known at prior close determines the return distribution for t.
        labels = regime_history['primary_regime'].shift(1).reindex(history.index)
        initial_distribution = np.array([transition_info['initial_distribution'][r] for r in REGIMES])
        if starting_regime in ('Current', 'current'):
            latest = regime_history['primary_regime'].iloc[-1] if len(regime_history) else 'unavailable'
            if latest not in REGIMES:
                raise ValueError('Current regime unavailable; select Random historical or an explicit starting regime.')
            state = np.full(n_sims, REGIMES.index(latest))
        elif starting_regime in ('Random historical', 'random'):
            state = rng.choice(4, n_sims, p=initial_distribution)
        elif starting_regime.lower() in REGIMES:
            state = np.full(n_sims, REGIMES.index(starting_regime.lower()))
        else:
            raise ValueError('Unknown starting regime.')
        counts = {r: int(labels.eq(r).sum()) for r in REGIMES}
        metadata.update(transition_estimation=transition_info, scenario=scenario_info,
            regime_samples=counts, starting_regime=starting_regime,
            sparse_fallback={r: {'conditional_probability': min(1., counts[r] / min_regime_samples),
                                 'unconditional_probability': 1 - min(1., counts[r] / min_regime_samples)} for r in REGIMES},
            regime_sampling='Synchronized historical daily rows, conditioned on PRIOR-session regime; daily Markov transitions, not within-regime block dynamics.')
        occupancy = np.zeros((steps + 1, 4))
        occupancy[0] = np.bincount(state, minlength=4) / n_sims
    base_array = history.to_numpy() if history is not None else None
    for group in range(groups):
        if include_estimation_uncertainty:
            indices = synchronized_block_indices(len(history), len(history), 1, block_length, rng)[:, 0]
            sample = history.iloc[indices]
            estimated_mean, estimated_cov = annualize_mean_cov(sample, trading_days, use_shrinkage=True)
            group_mean = mean + estimated_mean.to_numpy() - history.mean().to_numpy() * trading_days
            group_cov = estimated_cov.to_numpy()
            pools.append(base_array[indices])
            pool_labels.append(labels.iloc[indices].to_numpy() if labels is not None else None)
        else:
            group_mean, group_cov = mean, covariance
            pools.append(base_array)
            pool_labels.append(labels.to_numpy() if labels is not None else None)
        eigenvalues, eigenvectors = np.linalg.eigh(group_cov / trading_days)
        roots.append(eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0))))
        means.append(group_mean / trading_days)
    metadata['parameter_populations'] = groups
    metadata['uncertainty_method'] = 'Moving-block resampled parameter populations; mean perturbations around supplied model mean, shrinkage covariance; nonparametric methods resample empirical populations.' if include_estimation_uncertainty else 'Fixed supplied moments / fixed empirical population'
    masks = [np.arange(g, n_sims, groups) for g in range(groups)]
    starts = np.zeros(n_sims, dtype=int)
    holdings = np.tile(weights, (n_sims, 1)) * initial_value
    unit = np.ones(n_sims)
    peak = unit.copy()
    worst = np.zeros(n_sims)
    fan = np.zeros((steps + 1, 5)); fan[0] = initial_value
    paths = np.empty((steps, n_sims)) if return_paths else None
    clipped = 0
    unmet = np.zeros(n_sims)
    cumulative_cost = np.zeros(n_sims)
    rebalance_steps = {'daily': 1, 'monthly': max(1, trading_days//12), 'quarterly': max(1, trading_days//4), 'annual': trading_days, 'none': steps + 1}[rebalance_frequency]
    for step in range(1, steps + 1):
        draws = np.empty((n_sims, len(weights)))
        if transition is not None:
            probabilities = transition.to_numpy()[state]
            state = (rng.random(n_sims)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1).clip(max=3)
            occupancy[step] = np.bincount(state, minlength=4) / n_sims
        for group, mask in enumerate(masks):
            if method in ('gaussian', 'student_t'):
                noise = rng.standard_normal((len(mask), len(weights))) @ roots[group].T
                if method == 'student_t':
                    noise *= np.sqrt((degrees_of_freedom - 2) / rng.chisquare(degrees_of_freedom, len(mask)))[:, None]
                draws[mask] = means[group] + noise
            elif method == 'block_bootstrap':
                offset = (step - 1) % block_length
                if offset == 0:
                    starts[mask] = rng.integers(0, len(pools[group]) - block_length + 1, len(mask))
                draws[mask] = pools[group][starts[mask] + offset]
            else:
                for regime_index, regime in enumerate(REGIMES):
                    chosen = mask[state[mask] == regime_index]
                    candidates = np.flatnonzero(pool_labels[group] == regime)
                    conditional = rng.random(len(chosen)) < min(1., len(candidates) / min_regime_samples)
                    row_indices = rng.integers(0, len(pools[group]), len(chosen))
                    if len(candidates):
                        row_indices[conditional] = rng.choice(candidates, int(conditional.sum()))
                    draws[chosen] = pools[group][row_indices]
        clipped += int(np.count_nonzero(draws < -1))
        draws = np.maximum(draws, -1)
        before = holdings.sum(axis=1)
        holdings *= 1 + draws
        after_market = holdings.sum(axis=1)
        # Rebalance after market movement at the specified session close.
        if step % rebalance_steps == 0:
            target = after_market[:, None] * weights
            cost = np.abs(target - holdings).sum(axis=1) * transaction_cost_bps / 10000
            holdings = (after_market - cost)[:, None] * weights
            cumulative_cost += cost
        after = holdings.sum(axis=1)
        factor = np.divide(after, before, out=np.ones(n_sims), where=before > 0)
        unit *= factor
        peak = np.maximum(peak, unit)
        worst = np.minimum(worst, np.divide(unit, peak, out=np.zeros(n_sims), where=peak > 0) - 1)
        if step % trading_days == 0:
            year = step // trading_days - 1
            contribution = annual_contribution * (1 + contribution_growth) ** year
            withdrawal = annual_withdrawal * (1 + inflation) ** year
            fractions = np.divide(holdings, after[:, None], out=np.tile(weights, (n_sims, 1)), where=after[:, None] > 0)
            available = after + contribution
            unmet += np.maximum(withdrawal - available, 0)
            holdings = np.maximum(available - withdrawal, 0)[:, None] * fractions
        values = holdings.sum(axis=1)
        if not np.isfinite(values).all():
            raise ValueError('Simulation overflow; reduce horizon, returns or cash-flow growth.')
        fan[step] = np.percentile(values, [5, 25, 50, 75, 95])
        if paths is not None:
            paths[step-1] = values
    terminal = holdings.sum(axis=1)
    columns = ['p5', 'p25', 'median', 'p75', 'p95']
    fan = pd.DataFrame(fan, columns=columns, index=pd.RangeIndex(steps + 1, name='step'))
    stats = simulation_statistics(terminal, worst, initial_value, target_value)
    stats.update(prob_unmet_withdrawal=float(np.mean(unmet > 0)), mean_unmet_withdrawal=float(unmet.mean()),
                 mean_transaction_cost=float(cumulative_cost.mean()))
    metadata['floored_asset_draws'] = clipped
    result = {'statistics': stats, 'percentile_paths': fan,
              'real_percentile_paths': fan.div((1 + inflation) ** (fan.index.to_numpy() / trading_days), axis=0),
              'terminal_values': terminal, 'metadata': metadata,
              'transition_matrix': transition,
              'regime_occupancy': pd.DataFrame(occupancy, columns=REGIMES) if transition is not None else pd.DataFrame()}
    if paths is not None:
        result['paths'] = pd.DataFrame(paths, columns=[f'sim_{i+1}' for i in range(n_sims)], index=pd.RangeIndex(1, steps+1, name='step'))
        result['paths'].attrs.update(initial_value=initial_value, method=method, assumptions=metadata)
    return result


# Preserve the historical return type and positional signature for console callers.
def simulate_portfolio_paths(weights, mu, cov, years=10, n_sims=10000, trading_days=252,
                             initial_value=1., seed=42, method='gaussian', **kwargs):
    kwargs.setdefault('rebalance_frequency', 'daily')
    return run_simulation(weights, mu, cov, years=years, n_sims=n_sims, trading_days=trading_days,
        initial_value=initial_value, seed=seed, method=method, return_paths=True, **kwargs)['paths']
