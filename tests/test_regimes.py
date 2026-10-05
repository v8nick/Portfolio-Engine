"""Phase 2: trailing information, economic signs, and completed outcomes."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import regimes
from market_intelligence import build_market_snapshot, get_regime_features, MARKET_SYMBOLS


class RegimeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.dates = pd.bdate_range('2020-01-01', periods=420)
        names = list(MARKET_SYMBOLS)
        self.prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(.0003, .01, (420, len(names))), axis=0), index=self.dates, columns=names)
        self.yields = pd.DataFrame({'2Y': 3 + np.cumsum(rng.normal(0, .02, 420)),
                                    '10Y': 4 + np.cumsum(rng.normal(0, .02, 420))}, index=self.dates)
        self.model_options = {'window': 126, 'min_periods': 63}

    def test_history_matches_phase1_features(self):
        history = regimes.build_regime_feature_history(self.prices, self.yields, yield_lag_days=0)
        snapshot = build_market_snapshot(self.prices, self.yields)
        for horizon in ('21D', '63D'):
            for name, value in get_regime_features(snapshot, horizon).items():
                self.assertIn(name, history)
                self.assertAlmostEqual(history[name].iloc[-1], value, places=10)

    def test_features_standardization_and_classification_ignore_future(self):
        features = regimes.build_regime_feature_history(self.prices, self.yields)
        changed_prices, changed_yields = self.prices.copy(), self.yields.copy()
        changed_prices.iloc[301:] *= 10
        changed_yields.iloc[301:] += 20
        changed = regimes.build_regime_feature_history(changed_prices, changed_yields)
        pd.testing.assert_frame_equal(features.iloc[:301], changed.iloc[:301])
        standardized = regimes.standardize_regime_features(features, **self.model_options)
        future_standardized = regimes.standardize_regime_features(changed, **self.model_options)
        pd.testing.assert_frame_equal(standardized.iloc[:301], future_standardized.iloc[:301])
        full = regimes.classify_regime_history(features, **self.model_options)
        truncated = regimes.classify_regime_history(features.iloc[:301], **self.model_options)
        pd.testing.assert_frame_equal(full.iloc[:301], truncated)
        altered = regimes.classify_regime_history(changed, **self.model_options)
        pd.testing.assert_frame_equal(full.iloc[:301], altered.iloc[:301])
        current = regimes.get_current_regime(self.prices, self.yields, as_of=self.dates[300], **self.model_options)
        changed_current = regimes.get_current_regime(changed_prices, changed_yields, as_of=self.dates[300], **self.model_options)
        self.assertEqual(current, changed_current)

    def test_rolling_standardization_and_clipping(self):
        features = pd.DataFrame({'x': [1., 2., 3., 4., 1000.]}, index=self.dates[:5])
        standardized = regimes.standardize_regime_features(features, window=3, min_periods=3, clip=.75)
        self.assertTrue(standardized.iloc[:2].isna().all().all())
        self.assertAlmostEqual(standardized['x'].iloc[2], .75)
        self.assertLessEqual(standardized.abs().max().max(), .75)
        self.assertTrue(regimes.standardize_regime_features(features * 0, window=3, min_periods=3).isna().all().all())
        with self.assertRaises(ValueError):
            regimes.standardize_regime_features(features, window=2, min_periods=3)

    def test_yield_lag_holidays_and_staleness(self):
        dates = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-08'])
        prices = pd.DataFrame({'SPY': [100., 101., 102., 103.]}, index=dates)
        yields = pd.DataFrame({'10Y': [4., 4.1]}, index=pd.to_datetime(['2024-01-01', '2024-01-02']))
        features = regimes.build_regime_feature_history(prices, yields, horizons=(1,), max_age_days=3, yield_lag_days=1)
        self.assertTrue(pd.isna(features['inflation_10y_change_bp_1D'].iloc[0]))
        self.assertAlmostEqual(features['inflation_10y_change_bp_1D'].iloc[1], 10.)
        self.assertAlmostEqual(features['inflation_10y_change_bp_1D'].iloc[2], 10.)
        self.assertTrue(pd.isna(features['inflation_10y_change_bp_1D'].iloc[3]))
        self.assertEqual(list(features.index), list(dates))

    def test_quadrants_and_probability_directions(self):
        cases = [(1, -1, 'goldilocks'), (1, 1, 'reflation'), (-1, -1, 'slowdown'), (-1, 1, 'stagflation'), (0, 0, 'slowdown')]
        for g, i, label in cases:
            self.assertEqual(regimes.classify_quadrant(g, i), label)
            p = regimes.regime_probabilities(g, i)
            self.assertAlmostEqual(sum(p.values()), 1.)
            self.assertTrue(all(0 <= x <= 1 for x in p.values()))
        base = regimes.regime_probabilities(0, 0)
        growth = regimes.regime_probabilities(1, 0)
        inflation = regimes.regime_probabilities(0, 1)
        self.assertGreater(growth['goldilocks'] + growth['reflation'], base['goldilocks'] + base['reflation'])
        self.assertGreater(inflation['reflation'] + inflation['stagflation'], base['reflation'] + base['stagflation'])
        self.assertAlmostEqual(sum(regimes.regime_probabilities(1000, -1000, .0001).values()), 1.)
        with self.assertRaises(ValueError):
            regimes.regime_probabilities(1, 1, 0)

    def test_composite_signs_and_gold_weight(self):
        inputs = {'conditions_2y_change_bp_21D': 1., 'conditions_dxy_21D': 1.,
                  'conditions_hyg_21D': -1., 'conditions_vix_21D': 1.,
                  'stress_credit_weakness_21D': 1., 'stress_breadth_weakness_21D': 1.,
                  'stress_vix_zscore': 1., 'stress_spy_realized_vol_21d': 1.,
                  'growth_iwm_spy_21D': 1., 'growth_rsp_spy_21D': 1.,
                  'growth_hyg_lqd_21D': 1., 'growth_xly_xlp_21D': 1.,
                  'inflation_10y_change_bp_21D': 0., 'inflation_tip_ief_21D': 0.,
                  'inflation_dbc_21D': 0., 'inflation_gld_21D': 2.}
        positive = pd.DataFrame([inputs], index=self.dates[:1])
        scores = regimes.composite_regime_scores(positive)
        self.assertAlmostEqual(scores['financial_conditions_score'].iloc[0], 1.)
        self.assertAlmostEqual(scores['stress_score'].iloc[0], 1.)
        self.assertAlmostEqual(scores['growth_score'].iloc[0], 1.)
        self.assertAlmostEqual(scores['inflation_score'].iloc[0], 1 / 3.5)
        opposite = regimes.composite_regime_scores(-positive)
        for group in regimes.COMPONENTS:
            self.assertAlmostEqual(opposite[f'{group}_score'].iloc[0], -scores[f'{group}_score'].iloc[0])

    def test_overlays_and_smoothing_reduce_boundary_flips(self):
        self.assertEqual(regimes.classify_overlays(-1, -1), {'financial_conditions': 'easy', 'stress': 'low'})
        self.assertEqual(regimes.classify_overlays(0, 0), {'financial_conditions': 'neutral', 'stress': 'normal'})
        self.assertEqual(regimes.classify_overlays(1, 1), {'financial_conditions': 'tight', 'stress': 'elevated'})
        self.assertEqual(regimes.classify_overlays(1, 2)['stress'], 'severe')
        growth = [1., 1., 1., -.1, 1., -.1]
        standardized = pd.DataFrame({'growth_iwm_spy_21D': growth, 'growth_rsp_spy_21D': growth,
                                     'inflation_dbc_21D': -1., 'inflation_tip_ief_21D': -1.}, index=self.dates[:6])
        with patch('regimes.standardize_regime_features', return_value=standardized):
            history = regimes.classify_regime_history(standardized, smoothing_span=5)
        self.assertEqual(history['raw_regime'].iloc[3], 'slowdown')
        self.assertEqual(history['smoothed_regime'].iloc[3], 'goldilocks')
        self.assertTrue(history['smoothed_regime'].eq('goldilocks').all())
        self.assertTrue(history['confidence'].between(0, 1).all())

    def test_missing_features_and_current_json(self):
        current = regimes.get_current_regime(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(current['primary_regime'], 'unavailable')
        self.assertEqual(current['feature_count'], 0)
        self.assertEqual(current['confidence'], 0)
        self.assertAlmostEqual(sum(current['probabilities'].values()), 1.)
        json.dumps(current, allow_nan=False)
        short = regimes.get_current_regime(self.prices.iloc[:30], self.yields.iloc[:30])
        self.assertEqual(short['primary_regime'], 'unavailable')
        partial = regimes.get_current_regime(self.prices.drop(columns=['GLD']), self.yields, **self.model_options)
        self.assertNotEqual(partial['primary_regime'], 'unavailable')
        self.assertLess(partial['coverage']['inflation'], 1.)
        self.assertTrue(partial['data_quality'])
        json.dumps(partial, allow_nan=False)
        stale = regimes.get_current_regime(self.prices, self.yields, as_of=self.dates[-1] + pd.Timedelta(days=20), **self.model_options)
        self.assertEqual(stale['primary_regime'], 'unavailable')
        self.assertEqual(stale['feature_count'], 0)

    def test_forward_returns_begin_after_regime_date(self):
        prices = pd.DataFrame({'SPY': [1., 10., 11., 12.]}, index=self.dates[:4])
        labels = pd.DataFrame({'smoothed_regime': ['unavailable', 'goldilocks', 'slowdown', 'slowdown']}, index=prices.index)
        analysis = regimes.analyze_regime_forward_returns(labels, prices, horizons=(1,))
        row = analysis.loc[(analysis['regime'] == 'goldilocks') & (analysis['asset'] == 'SPY')].iloc[0]
        self.assertEqual(row['n_observations'], 1)
        self.assertAlmostEqual(row['mean'], .1)  # 11/10-1, not the 900% jump on decision date.
        self.assertEqual(row['positive_frequency'], 1.)
        empty = analysis.loc[analysis['asset'] == 'QQQ']
        self.assertTrue(empty['n_observations'].eq(0).all())
        self.assertTrue(empty['mean'].isna().all())

    def test_nonoverlapping_sampling_and_completed_cutoff(self):
        prices = pd.DataFrame({'SPY': np.arange(1., 11.)}, index=self.dates[:10])
        labels = pd.DataFrame({'smoothed_regime': ['goldilocks', 'slowdown'] * 5}, index=prices.index)
        samples, _ = regimes._forward_samples(labels, prices, 3, 'smoothed_regime', True)
        self.assertEqual(list(samples['SPY'].index), list(prices.index[[0, 3, 6]]))
        table = regimes.analyze_regime_forward_returns(labels, prices, horizons=(3,), non_overlapping=True)
        self.assertEqual(table.loc[table.asset == 'SPY', 'n_observations'].sum(), 3)
        table = regimes.analyze_regime_forward_returns(labels, prices, horizons=(3,), as_of=prices.index[4])
        self.assertEqual(table.loc[table.asset == 'SPY', 'n_observations'].sum(), 2)
        altered = prices.copy()
        altered.iloc[5:] *= 100
        changed = regimes.analyze_regime_forward_returns(labels, altered, horizons=(3,), as_of=prices.index[4])
        pd.testing.assert_frame_equal(table, changed)

    def test_shrinkage_toward_unconditional_and_no_future_outcomes(self):
        prices = pd.DataFrame({'SPY': [100., 110., 121., 120., 118., 116., 114., 113.]}, index=self.dates[:8])
        labels = pd.DataFrame({'smoothed_regime': ['goldilocks'] * 2 + ['slowdown'] * 6}, index=prices.index)
        estimates = regimes.estimate_regime_expected_returns(labels, prices, horizons=(1,), shrinkage_strength=2.)
        row = estimates.loc[(estimates.asset == 'SPY') & (estimates.regime == 'goldilocks')].iloc[0]
        self.assertEqual(row['sample_count'], 2)
        self.assertAlmostEqual(row['lambda'], .5)
        self.assertAlmostEqual(row['shrunk_estimate'], .5 * row['raw_regime_estimate'] + .5 * row['unconditional_estimate'])
        self.assertLess(abs(row['shrunk_estimate'] - row['unconditional_estimate']), abs(row['raw_regime_estimate'] - row['unconditional_estimate']))
        missing = estimates.loc[(estimates.asset == 'SPY') & (estimates.regime == 'stagflation')].iloc[0]
        self.assertEqual(missing['lambda'], 0.)
        self.assertAlmostEqual(missing['shrunk_estimate'], missing['unconditional_estimate'])
        early = regimes.estimate_regime_expected_returns(labels, prices, horizons=(2,), as_of=prices.index[4])
        altered = prices.copy()
        altered.iloc[5:] *= 1000
        later = regimes.estimate_regime_expected_returns(labels, altered, horizons=(2,), as_of=prices.index[4])
        pd.testing.assert_frame_equal(early, later)

    def test_diagnostics_break_at_missing_regimes(self):
        history = pd.DataFrame({'smoothed_regime': ['goldilocks', 'goldilocks', 'slowdown', 'unavailable', 'slowdown', 'slowdown']}, index=self.dates[:6])
        diagnostics = regimes.regime_diagnostics(history)
        self.assertEqual(diagnostics['number_of_transitions'], 1)
        self.assertAlmostEqual(diagnostics['average_duration_observations']['slowdown'], 1.5)
        self.assertAlmostEqual(diagnostics['regime_frequency']['goldilocks'], 2 / 5)
        self.assertEqual(diagnostics['latest_regime'], 'slowdown')
        self.assertEqual(diagnostics['unavailable_observations'], 1)

    def test_cli_json_and_console_offline(self):
        data = {'prices': self.prices, 'yields': self.yields, 'data_quality': []}
        for mode in (['--json'], []):
            output = io.StringIO()
            with patch('regimes.load_market_data', return_value=data), patch('sys.argv', ['regimes.py', '--end', self.dates[-1].date().isoformat()] + mode), contextlib.redirect_stdout(output):
                regimes.main()
            if mode:
                result = json.loads(output.getvalue())
                self.assertAlmostEqual(sum(result['current']['probabilities'].values()), 1.)
                self.assertEqual(len(result['forward_returns']), len(regimes.FORWARD_ASSETS) * 4 * 3)
            else:
                self.assertIn('CURRENT MARKET REGIME', output.getvalue())
                self.assertIn('descriptive, not forecasts', output.getvalue())


if __name__ == '__main__':
    unittest.main()
