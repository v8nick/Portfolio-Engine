"""Economic integration tests: bounds, information timing, evidence, and costs."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from black_litterman import merge_manual_regime_views
from config import shared
from market_intelligence import MARKET_SYMBOLS
from mpt import optimize_risk_parity, validate_allocation_weights
from portfolio_decision import (build_portfolio_recommendation, weight_regime_expected_returns,
    build_regime_bl_views, resolve_allocation_policy, optimizer_consensus,
    resampled_weight_stability, build_tactical_allocation,
    implementation_rebalance_decision, _settings)
from regimes import REGIMES
from report import empirical_var_cvar


def policy(weights, low=0., high=1.):
    return {t: {'strategic_weight': w, 'minimum_weight': low, 'maximum_weight': high,
                'tactical_low': low, 'tactical_high': high, 'group': t, 'role': 'core'}
            for t, w in weights.items()}


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = dict(zip(REGIMES, [.1, .2, .3, .4]))
        self.estimates = pd.DataFrame([{'asset': 'SPY', 'regime': r, 'horizon': 63,
            'unconditional_estimate': .02, 'shrunk_estimate': v, 'sample_count': 20, 'lambda': .5}
            for r, v in zip(REGIMES, [.01, .02, .03, .04])])
        self.regime = {'confidence': .8, 'coverage': {'growth': 1., 'inflation': 1.},
                       'overlays': {'financial_conditions': 'neutral', 'stress': 'normal'}}
        rng = np.random.default_rng(2026)
        dates = pd.bdate_range('2016-01-01', periods=650)
        tickers = list(MARKET_SYMBOLS) + ['VEA', 'VWO', 'MSFT']
        self.prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(.0003, .01, (650, len(tickers))), axis=0), index=dates, columns=tickers)
        self.yields = pd.DataFrame({'2Y': 3 + rng.normal(0, .2, 650), '10Y': 4 + rng.normal(0, .2, 650), '3MO': 3.}, index=dates)
        self.strategy = policy({'SPY': .5, 'IEF': .3, 'GLD': .2})
        for item in self.strategy.values():
            item['tactical_low'] = item['strategic_weight'] - .05
            item['tactical_high'] = item['strategic_weight'] + .05
        self.options = {'robustness_repetitions': 6, 'minimum_history': 100}

    def recommendation(self, **kwargs):
        defaults = dict(current_weights={'SPY': .5, 'IEF': .3, 'GLD': .2},
                        strategic_allocation=self.strategy, manual_views={}, relative_views=[], options=self.options)
        defaults.update(kwargs)
        return build_portfolio_recommendation(self.prices, self.yields, **defaults)

    def test_probability_weighted_returns_use_all_regimes(self):
        result = weight_regime_expected_returns(self.estimates, self.probabilities, pd.Series({'SPY': .1}))['SPY']
        self.assertAlmostEqual(result['weighted_regime'], .12)
        self.assertAlmostEqual(result['unconditional'], .08)
        self.assertAlmostEqual(result['delta'], .04)
        self.assertEqual(len(result['regime_components']), 4)

    def test_missing_and_sparse_estimates_fall_back(self):
        result = weight_regime_expected_returns(self.estimates.iloc[:3], self.probabilities, pd.Series({'SPY': .1}))['SPY']
        self.assertAlmostEqual(result['regime_components']['stagflation']['shrunk_annual_return'], .08)
        missing = weight_regime_expected_returns(pd.DataFrame(), self.probabilities, pd.Series({'SPY': .1}))['SPY']
        self.assertAlmostEqual(missing['weighted_regime'], .1)
        self.assertEqual(missing['sample_quality'], 0.)
        sparse = self.estimates.copy()
        sparse['sample_count'] = 1
        reduced = weight_regime_expected_returns(sparse, self.probabilities, pd.Series({'SPY': .1}))['SPY']
        full = weight_regime_expected_returns(self.estimates, self.probabilities, pd.Series({'SPY': .1}))['SPY']
        self.assertLess(abs(reduced['delta']), abs(full['delta']))
        self.assertLess(reduced['sample_quality'], result['sample_quality'])

    def test_bl_adjustments_are_capped_and_quality_discounted(self):
        inputs = {'SPY': {'weighted_regime': 2., 'delta': 2., 'sample_quality': 1., 'estimate_stability': 1., 'horizon': 63}}
        positive = build_regime_bl_views(pd.Series({'SPY': .08}), inputs, self.regime, max_adjustment=.025)['SPY']
        self.assertAlmostEqual(positive['adjustment'], .025)
        self.assertTrue(positive['cap_applied'])
        self.assertLessEqual(positive['confidence'], .25)
        inputs['SPY']['delta'] = -2.
        negative = build_regime_bl_views(pd.Series({'SPY': .08}), inputs, self.regime)['SPY']
        self.assertAlmostEqual(negative['adjustment'], -.025)
        inputs['SPY']['sample_quality'] = 0.
        absent = build_regime_bl_views(pd.Series({'SPY': .08}), inputs, self.regime)['SPY']
        self.assertEqual(absent['adjustment'], 0.)
        self.assertEqual(absent['confidence'], 0.)

    def test_manual_views_preserved_with_transparent_merge(self):
        manual = {'SPY': (.1, .75)}
        tactical = {'SPY': {'confidence': .25, 'adjustment': .02, 'adjusted_bl_view': .08, 'prior_expected_return': .06}}
        merged, sources = merge_manual_regime_views(manual, tactical)
        self.assertEqual(manual, {'SPY': (.1, .75)})
        self.assertAlmostEqual(merged['SPY'][0], .102)
        self.assertEqual(merged['SPY'][1], .75)
        self.assertEqual(sources['SPY']['source'], 'combined')
        unchanged, _ = merge_manual_regime_views({'SPY': (.1, 1.)}, tactical)
        self.assertEqual(unchanged['SPY'], (.1, 1.))
        tactical['SPY']['adjustment'] = 0.
        unchanged, source = merge_manual_regime_views(manual, tactical)
        self.assertEqual(unchanged, manual)
        self.assertEqual(source['SPY']['source'], 'manual')

    def test_candidates_and_final_respect_budget_bands_and_risk_reconciliation(self):
        result = self.recommendation()
        self.assertEqual(set(result['candidate_allocations']), {'strategic', 'minimum_variance', 'bl_max_sharpe', 'risk_parity', 'cvar'})
        for item in list(result['candidate_allocations'].values()) + [result['recommended_portfolio']]:
            self.assertNotEqual(item.get('status'), 'unavailable')
            self.assertAlmostEqual(sum(item['weights'].values()), 1., places=6)
            for asset, weight in item['weights'].items():
                limits = result['constraints']['asset_bounds'][asset]
                self.assertGreaterEqual(weight + 1e-6, limits[0])
                self.assertLessEqual(weight - 1e-6, limits[1])
            self.assertAlmostEqual(sum(v['percentage'] for v in item['risk_contributions'].values()), 1., places=6)
            self.assertAlmostEqual(sum(v['percentage_risk_contribution'] for v in item['groups'].values()), 1., places=6)
        for asset, central in result['recommended_portfolio']['central_weights'].items():
            low, high = result['recommended_portfolio']['weight_ranges'][asset]
            self.assertLessEqual(low, central)
            self.assertGreaterEqual(high, central)
        self.assertIn('historical_tail_risk', result['recommended_portfolio'])
        self.assertLessEqual(result['candidate_allocations']['cvar']['historical_tail_risk']['cvar'],
                             result['candidate_allocations']['strategic']['historical_tail_risk']['cvar'] + 1e-6)
        json.dumps(result, allow_nan=False)

    def test_risk_parity_equal_contributions_synthetic(self):
        cov = pd.DataFrame(np.diag([.01, .04, .09]), index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
        weights = optimize_risk_parity(cov, [(0, 1)] * 3, np.repeat(1/3, 3))
        contributions = weights * (cov.to_numpy() @ weights)
        percentages = contributions / contributions.sum()
        np.testing.assert_allclose(percentages, np.repeat(1/3, 3), atol=1e-5)
        np.testing.assert_allclose(weights, np.array([1., .5, 1/3]) / (1 + .5 + 1/3), atol=1e-5)

    def test_consensus_distinguishes_disagreement(self):
        high = optimizer_consensus({'a': np.array([.5, .5]), 'b': np.array([.51, .49])}, ['A', 'B'], np.array([.5, .5]))
        low = optimizer_consensus({'a': np.array([.1, .9]), 'b': np.array([.9, .1])}, ['A', 'B'], np.array([.5, .5]))
        self.assertEqual(high['agreement_label'], 'high')
        self.assertEqual(low['agreement_label'], 'low')
        self.assertGreater(high['agreement_score'], low['agreement_score'])

    def test_resampled_distributions_are_valid(self):
        returns = self.prices[['SPY', 'IEF', 'GLD']].pct_change(fill_method=None).dropna()
        settings = _settings(self.options)
        settings['resolved_rf'] = .03
        mu = pd.Series([.1, .06, .07], index=returns.columns)
        strategic = np.array([.5, .3, .2])
        bounds = [(.45, .55), (.25, .35), (.15, .25)]
        caps = {'satellites': ([], .15)}
        samples, stability = resampled_weight_stability(returns, mu, strategic, bounds, caps, settings)
        self.assertEqual(len(samples), 6)
        for _, row in samples.iterrows():
            validate_allocation_weights(row.values, list(returns), bounds, caps)
        self.assertTrue(0 <= stability['overall_score'] <= 1)
        for item in stability['per_asset'].values():
            self.assertLessEqual(item['q25'], item['median'])
            self.assertLessEqual(item['median'], item['q75'])

    def test_tactical_projection_cannot_breach_bands(self):
        strategic = np.array([.5, .5])
        bounds = [(.45, .55), (.45, .55)]
        consensus = optimizer_consensus({'a': np.array([1., 0.]), 'b': np.array([1., 0.])}, ['A', 'B'], strategic)
        stability = {'overall_score': 1., 'per_asset': {t: {'q25': 0., 'q75': 1.} for t in ['A', 'B']}}
        settings = _settings({'tactical_blend': 1.})
        central, ranges, _ = build_tactical_allocation(strategic, ['A', 'B'], bounds, {'satellites': ([], .15)}, consensus, stability, 1., self.regime, settings)
        np.testing.assert_allclose(central, [.55, .45], atol=1e-6)
        for i, t in enumerate(['A', 'B']):
            self.assertLessEqual(ranges[t][0], central[i])
            self.assertGreaterEqual(ranges[t][1], central[i])
            self.assertGreaterEqual(ranges[t][0], bounds[i][0])
            self.assertLessEqual(ranges[t][1], bounds[i][1])

    def test_no_trade_threshold_and_required_repair(self):
        tickers = ['A', 'B']
        mu = pd.Series([.1, .05], index=tickers)
        cov = pd.DataFrame(np.diag([.04, .01]), index=tickers, columns=tickers)
        args = dict(tickers=tickers, bounds=[(.3, .7)] * 2, caps={'satellites': ([], .15)},
                    ranges={'A': [.49, .51], 'B': [.49, .51]}, mu=mu, cov=cov, confidence=.9,
                    settings=_settings())
        held = implementation_rebalance_decision(np.array([.501, .499]), np.array([.5, .5]), **args)
        self.assertEqual(held['portfolio_action'], 'HOLD')
        self.assertEqual(held['estimated_turnover'], 0.)
        self.assertIn('within_no_trade_band', held['actions']['A']['reason_codes'])
        args['confidence'] = .1
        required = implementation_rebalance_decision(np.array([.9, .1]), np.array([.5, .5]), **args)
        self.assertTrue(required['constraint_repair_required'])
        self.assertEqual(required['actions']['A']['action'], 'TRIM')
        self.assertTrue(required['actions']['A']['required'])

    def test_current_weights_never_become_strategic_and_satellites_capped(self):
        result = self.recommendation(current_weights={'SPY': .6, 'IEF': .15, 'GLD': .15, 'MSFT': .1})
        self.assertEqual(result['strategic_portfolio']['weights']['SPY'], .5)
        self.assertEqual(result['strategic_portfolio']['weights']['MSFT'], 0.)
        self.assertEqual(result['policy']['MSFT']['role'], 'satellite')
        for item in result['candidate_allocations'].values():
            self.assertLessEqual(item['weights']['MSFT'], .05 + 1e-6)
        self.assertTrue(result['rebalance']['actions']['MSFT']['required'])
        self.assertIn('concentration_limit', result['reason_codes']['MSFT'])
        with self.assertRaises(ValueError):
            resolve_allocation_policy({}, {'SPY': 1.}, .15, .05)

    def test_future_observations_do_not_change_recommendation(self):
        cutoff = self.prices.index[500]
        earlier = self.recommendation(as_of=cutoff)
        prices, yields = self.prices.copy(), self.yields.copy()
        prices.loc[prices.index > cutoff] *= 10
        yields.loc[yields.index > cutoff] += 10
        changed = build_portfolio_recommendation(prices, yields, {'SPY': .5, 'IEF': .3, 'GLD': .2}, self.strategy,
            as_of=cutoff, manual_views={}, relative_views=[], options=self.options)
        self.assertEqual(earlier, changed)

    def test_sample_quality_reduces_regime_view_confidence(self):
        strong = weight_regime_expected_returns(self.estimates, self.probabilities, pd.Series({'SPY': .1}))
        low_counts = self.estimates.copy()
        low_counts['sample_count'] = 1
        weak = weight_regime_expected_returns(low_counts, self.probabilities, pd.Series({'SPY': .1}))
        prior = pd.Series({'SPY': .05})
        strong_view = build_regime_bl_views(prior, strong, self.regime)['SPY']
        weak_view = build_regime_bl_views(prior, weak, self.regime)['SPY']
        self.assertGreater(strong_view['confidence'], weak_view['confidence'])
        self.assertGreater(abs(strong_view['adjustment']), abs(weak_view['adjustment']))

    def test_empirical_cvar_known_sample_and_fractional_tail(self):
        returns = pd.Series([-.3, -.2, .1, .2])
        risk = empirical_var_cvar(returns, .75)
        self.assertAlmostEqual(risk['var'], .2)
        self.assertAlmostEqual(risk['cvar'], .3)
        self.assertAlmostEqual(empirical_var_cvar(returns, .875)['cvar'], .3)
        self.assertEqual(risk['frequency'], 'daily')

    def test_fixed_weights_and_invalid_strategy_fail_closed(self):
        strategy = policy({'SPY': .5, 'IEF': .3, 'GLD': .2})
        strategy['SPY']['fixed_weight'] = .5
        result = self.recommendation(strategic_allocation=strategy)
        for candidate in result['candidate_allocations'].values():
            self.assertAlmostEqual(candidate['weights']['SPY'], .5, places=6)
        strategy['SPY']['strategic_weight'] = .7
        with self.assertRaises(ValueError):
            self.recommendation(strategic_allocation=strategy)
        with self.assertRaises(ValueError):
            build_portfolio_recommendation(self.prices.drop(columns='IEF'), self.yields, {'SPY': .5, 'IEF': .3, 'GLD': .2}, self.strategy, options=self.options)

    def test_default_policy_budget_and_total_satellite_limit(self):
        from config import live
        policy_data, tickers, strategic, bounds, caps = resolve_allocation_policy(
            live.STRATEGIC_ALLOCATION, live.CURRENT_WEIGHTS,
            live.MAX_SATELLITE_ALLOCATION, live.MAX_INDIVIDUAL_SATELLITE_WEIGHT)
        self.assertAlmostEqual(strategic.sum(), 1.)
        self.assertTrue(all(strategic[tickers.index(t)] == 0 for t in caps['satellites'][0]))
        prices = self.prices.copy()
        for i, t in enumerate(tickers):
            if t not in prices:
                prices[t] = self.prices['MSFT'] * (1 + .001 * np.sin(np.arange(len(prices)) + i))
        result = build_portfolio_recommendation(prices, self.yields, options=self.options)
        for candidate in result['candidate_allocations'].values():
            self.assertNotEqual(candidate['status'], 'unavailable')
            self.assertAlmostEqual(sum(candidate['weights'].values()), 1., places=6)
            satellite_total = sum(candidate['weights'][t] for t in caps['satellites'][0])
            self.assertLessEqual(satellite_total, live.MAX_SATELLITE_ALLOCATION + 1e-6)
        json.dumps(result, allow_nan=False)

    def test_cost_and_tax_drag_can_suppress_discretionary_changes(self):
        tickers = ['A', 'B']
        args = dict(current=np.array([.6, .4]), target=np.array([.4, .6]), tickers=tickers,
                    bounds=[(.3, .7)] * 2, caps={'satellites': ([], .15)},
                    ranges={'A': [.39, .41], 'B': [.59, .61]}, mu=pd.Series([.1, .09], index=tickers),
                    cov=pd.DataFrame(np.diag([.04, .01]), index=tickers, columns=tickers),
                    confidence=.9, settings=_settings())
        no_tax = implementation_rebalance_decision(**args)
        self.assertEqual(no_tax['portfolio_action'], 'REBALANCE')
        high_tax = implementation_rebalance_decision(**args, gain_rates=np.array([1., 1.]))
        self.assertEqual(high_tax['portfolio_action'], 'HOLD')
        self.assertIn('tax_cost_not_justified', high_tax['actions']['A']['reason_codes'])
        with patch('config.shared.TRANSACTION_COST_BPS', 1000):
            high_cost = implementation_rebalance_decision(**args)
        self.assertEqual(high_cost['portfolio_action'], 'HOLD')
        self.assertIn('transaction_cost_not_justified', high_cost['actions']['A']['reason_codes'])

    def test_sample_quality_affects_overall_confidence(self):
        from portfolio_decision import weight_regime_expected_returns as original
        def controlled(quality):
            def estimate(*args, **kwargs):
                result = original(*args, **kwargs)
                for item in result.values():
                    item['sample_quality'] = quality
                return result
            return estimate
        options = dict(self.options, regime_bl_strength=0.)
        with patch('portfolio_decision.weight_regime_expected_returns', side_effect=controlled(.1)):
            weak = self.recommendation(options=options)
        with patch('portfolio_decision.weight_regime_expected_returns', side_effect=controlled(.9)):
            strong = self.recommendation(options=options)
        self.assertEqual(weak['expected_returns']['black_litterman_posterior'], strong['expected_returns']['black_litterman_posterior'])
        self.assertGreater(strong['recommended_portfolio']['confidence']['overall_score'], weak['recommended_portfolio']['confidence']['overall_score'])

    def test_solver_failure_is_visible_and_excluded_from_consensus(self):
        from pypfopt.exceptions import OptimizationError
        with patch('portfolio_decision.optimize_min_cvar', side_effect=OptimizationError('synthetic failure')):
            result = self.recommendation()
        self.assertEqual(result['candidate_allocations']['cvar']['status'], 'unavailable')
        self.assertNotIn('cvar', result['optimizer_consensus']['methods_used'])
        self.assertTrue(any('cvar unavailable' in w for w in result['warnings']))
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
