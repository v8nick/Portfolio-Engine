"""Offline economic invariants for research timing, simulation and UI integration."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from pathlib import Path
from data import compute_returns
from expected_returns import estimate_expected_returns
from mpt import annualize_mean_cov
from regimes import REGIMES, SCENARIOS, build_regime_history, estimate_transition_matrix, scenario_transition_matrix
from report import (backtest_metrics, rolling_backtest_metrics, regime_backtest_metrics,
                    historical_stress_tests, recommendation_diagnostics)
from simulation import run_simulation, synchronized_block_indices, simulation_statistics, simulate_portfolio_paths
from walkforward import walk_forward_backtest, get_rebalance_dates, backtest_history_start
from ui import services


def make_data(n=420):
    rng = np.random.default_rng(8)
    dates = pd.bdate_range('2018-01-01', periods=n)
    tickers = ['SPY', 'IEF', 'GLD', 'IWM', 'RSP', 'XLY', 'XLP', 'HYG', 'LQD', 'TIP', 'DBC', 'DXY', 'VIX']
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(.0003, .01, (n, len(tickers))), axis=0), index=dates, columns=tickers)
    yields = pd.DataFrame({'3MO': 3., '2Y': 3. + rng.normal(0, .1, n), '10Y': 4. + rng.normal(0, .1, n)}, index=dates)
    weights = {'SPY': .6, 'IEF': .4}
    policy = {t: {'strategic_weight': w, 'minimum_weight': 0., 'maximum_weight': 1.,
                  'tactical_low': 0., 'tactical_high': 1., 'role': 'core', 'group': t} for t, w in weights.items()}
    return prices, yields, weights, policy


class WalkForwardResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices, cls.yields, cls.weights, cls.policy = make_data(180)

    def run_test(self, **kwargs):
        defaults = dict(strategies=('bl_max_sharpe',), benchmark='spy', lookback=20,
                        return_model='historical_mean', costs_enabled=True, no_trade_threshold=0)
        defaults.update(kwargs)
        return walk_forward_backtest(self.prices, self.yields, self.weights, self.policy, **defaults)

    def test_next_observation_cost_timing_drift_and_benchmark_alignment(self):
        with patch('walkforward.optimize_max_sharpe_constrained', return_value=np.array([.2, .8])):
            output = self.run_test(transaction_cost_bps=100., slippage_bps=0.)
        item = output['strategies']['bl_max_sharpe']
        raw = compute_returns(self.prices[['SPY', 'IEF']])
        first = item['diagnostics'].iloc[0]
        self.assertEqual(first['turnover'], .8)
        self.assertGreater(first['effective_date'], item['diagnostics'].index[0])
        self.assertNotIn(item['diagnostics'].index[0], item['returns'].index)
        dt = first['effective_date']
        expected_gross = raw.loc[dt].to_numpy() @ [.2, .8]
        self.assertAlmostEqual(item['returns'].loc[dt], (1 - .008) * (1 + expected_gross) - 1)
        next_dt = item['returns'].index[1]
        drift = np.array([.2, .8]) * (1 + raw.loc[dt].to_numpy()) / (1 + expected_gross)
        np.testing.assert_allclose(item['daily_weights'].loc[next_dt], drift)
        self.assertAlmostEqual(item['returns'].loc[next_dt], raw.loc[next_dt].to_numpy() @ drift)
        effective_costs = item['diagnostics'].set_index('effective_date')['cost_rate'].reindex(item['returns'].index).fillna(0)
        np.testing.assert_allclose(item['returns'], (1 - effective_costs) * (1 + item['gross_returns']) - 1)
        self.assertTrue(item['returns'].index.equals(output['strategies']['spy']['returns'].index))
        np.testing.assert_allclose(output['strategies']['spy']['returns'], raw.SPY.loc[item['returns'].index])

    def test_no_future_prices_or_yields_used(self):
        def optimizer(mu, **kwargs):
            return np.array([.8, .2]) if mu.iloc[0] > mu.iloc[1] else np.array([.2, .8])
        with patch('walkforward.optimize_max_sharpe_constrained', side_effect=optimizer):
            first = self.run_test()
            cutoff = first['strategies']['bl_max_sharpe']['weight_history'].index[2]
            prices, yields = self.prices.copy(), self.yields.copy()
            prices.loc[prices.index > cutoff] *= 5
            yields.loc[yields.index > cutoff] = 99
            changed = walk_forward_backtest(prices, yields, self.weights, self.policy,
                strategies=('bl_max_sharpe',), benchmark='spy', lookback=20, return_model='historical_mean', no_trade_threshold=0)
        a, b = first['strategies']['bl_max_sharpe'], changed['strategies']['bl_max_sharpe']
        pd.testing.assert_frame_equal(a['weight_history'].loc[:cutoff], b['weight_history'].loc[:cutoff])
        pd.testing.assert_series_equal(a['returns'].loc[:cutoff], b['returns'].loc[:cutoff])

    def test_failures_do_not_switch_optimizer_or_hide_missing_benchmark(self):
        with patch('walkforward.optimize_max_sharpe_constrained', side_effect=ValueError('test solver failure')):
            result = self.run_test()
        diag = result['strategies']['bl_max_sharpe']['diagnostics']
        self.assertTrue(diag.error.str.contains('test solver failure').all())
        self.assertTrue(diag.turnover.eq(0).all())
        result = walk_forward_backtest(self.prices.drop(columns='SPY'), self.yields,
            {'IEF': 1.}, {'IEF': dict(self.policy['IEF'], strategic_weight=1.)}, strategies=('current',), benchmark='spy', lookback=20)
        self.assertIn('spy', result['failures'])
        self.assertTrue(np.isnan(result['metrics'].loc['current', 'beta']))

    def test_quarterly_no_trade_and_costs_off(self):
        dates = get_rebalance_dates(self.prices.index, 'quarterly')
        self.assertTrue(all(dt.month in (3, 6, 9, 12) for dt in dates[:-1]))
        result = self.run_test(strategies=('current',), rebalance_frequency='quarterly',
                               no_trade_threshold=1., costs_enabled=False)
        self.assertTrue(result['strategies']['current']['diagnostics'].turnover.eq(0).all())
        self.assertTrue(result['strategies']['current']['diagnostics'].cost_rate.eq(0).all())

    def test_all_strategies_unavailable_preserves_dates_and_coverage_errors(self):
        prices = self.prices.copy()
        prices.loc[prices.index[100], 'SPY'] = np.nan
        result = walk_forward_backtest(prices, self.yields, self.weights, self.policy,
            strategies=('current', 'strategic', 'bl_max_sharpe'), benchmark='spy', lookback=20)
        self.assertFalse(result['strategies'])
        self.assertTrue(result['returns'].empty)
        self.assertIsInstance(result['returns'].index, pd.DatetimeIndex)
        self.assertTrue(result['calendar_returns'].empty)
        self.assertIsInstance(result['calendar_returns'].index, pd.DatetimeIndex)
        self.assertEqual(set(result['failures']), {'current', 'strategic', 'bl_max_sharpe', 'spy'})
        for error in result['failures'].values():
            self.assertIn('SPY', error)
            self.assertIn(str(prices.index[100].date()), error)

    def test_macro_only_dates_do_not_break_portfolio_returns(self):
        baseline = self.run_test(strategies=('current',))
        prices = self.prices.copy()
        extra = prices.index[30] + pd.Timedelta(days=1)
        while extra in prices.index:
            extra += pd.Timedelta(days=1)
        prices.loc[extra] = np.nan
        prices.loc[extra, 'DXY'] = 100.
        prices = prices.sort_index()
        result = walk_forward_backtest(prices, self.yields, self.weights, self.policy,
            strategies=('current',), benchmark='spy', lookback=20, return_model='historical_mean',
            no_trade_threshold=0)
        self.assertFalse(result['failures'])
        self.assertEqual(result['metadata']['excluded_macro_only_rows'], 1)
        # Inserting/removing a date clears pandas' optional frequency hint;
        # actual dates and every return must still match exactly.
        pd.testing.assert_frame_equal(baseline['returns'], result['returns'], check_freq=False)
        self.assertNotIn(extra, result['returns'].index)

    def test_available_start_requires_full_common_window_after_late_inception(self):
        prices = self.prices.copy()
        prices.loc[prices.index[:50], 'IEF'] = np.nan
        start = backtest_history_start(prices, ['SPY', 'IEF'], lookback=20)
        self.assertEqual(start, prices.index[70])
        result = walk_forward_backtest(prices, self.yields, self.weights, self.policy,
            strategies=('current', 'bl_max_sharpe'), start=start, lookback=20)
        self.assertFalse(result['failures'])
        self.assertTrue(result['strategies']['bl_max_sharpe']['diagnostics'].error.eq('').all())
        self.assertIsNone(backtest_history_start(prices, ['MISSING'], lookback=20))

    def test_all_optimizers_and_constraint_validation(self):
        result = self.run_test(strategies=('current', 'strategic', 'equal_weight', 'balanced_60_40',
                                           'minimum_variance', 'risk_parity', 'cvar'), return_model='bl')
        self.assertFalse(result['failures'])
        for name, item in result['strategies'].items():
            self.assertTrue(item['diagnostics'].error.eq('').all(), (name, item['diagnostics'].error.tolist()))
            np.testing.assert_allclose(item['daily_weights'].sum(axis=1), 1.)
        bad = self.prices.copy()
        bad.loc[bad.index[100], 'IEF'] = np.nan
        result = walk_forward_backtest(bad, self.yields, self.weights, self.policy,
            strategies=('strategic',), benchmark='spy', lookback=20)
        self.assertIn('strategic', result['failures'])

    def test_return_models_and_dated_views(self):
        returns = compute_returns(self.prices[['SPY', 'IEF']]).dropna().iloc[:50]
        mu, cov = annualize_mean_cov(returns, use_shrinkage=True)
        args = dict(returns=returns, covariance=cov, prior_weights=np.array([.6, .4]), as_of=returns.index[-1])
        historical, meta = estimate_expected_returns(**args, model='historical_mean')
        np.testing.assert_allclose(historical, returns.mean() * 252)
        self.assertIn('research', meta['source'])
        baseline, _ = estimate_expected_returns(**args)
        future, _ = estimate_expected_returns(**args, view_history=[{'as_of': self.prices.index[90], 'absolute': {'SPY': (.9, 1.)}}])
        np.testing.assert_allclose(future, baseline)
        dated, _ = estimate_expected_returns(**args, view_history=[{'as_of': returns.index[0], 'absolute': {'SPY': (.2, 1.)}}])
        self.assertAlmostEqual(dated.SPY, .2)

    def test_regime_overlay_ignores_future_completed_outcomes_and_views(self):
        prices, yields, weights, policy = make_data(500)
        cutoff = prices.index[420]
        history = build_regime_history(prices, yields)
        returns = compute_returns(prices[['SPY', 'IEF']]).dropna().loc[:cutoff].tail(126)
        _, covariance = annualize_mean_cov(returns, use_shrinkage=True)
        args = dict(returns=returns, covariance=covariance, prior_weights=np.array([.6, .4]),
                    model='bl_regime', as_of=cutoff, prices=prices, yields=yields, regime_history=history)
        mu, metadata = estimate_expected_returns(**args)
        changed = prices.copy(); changed.loc[changed.index > cutoff] *= .1
        changed_yields = yields.copy(); changed_yields.loc[changed_yields.index > cutoff] = 20
        args.update(prices=changed, yields=changed_yields, regime_history=build_regime_history(changed, changed_yields))
        altered, altered_metadata = estimate_expected_returns(**args)
        pd.testing.assert_series_equal(mu, altered)
        self.assertEqual(metadata['view_sources'], altered_metadata['view_sources'])
        result = walk_forward_backtest(prices, yields, weights, policy, strategies=('bl_max_sharpe',),
            benchmark='spy', start=prices.index[400], lookback=126, return_model='bl_regime')
        self.assertTrue(result['strategies']['bl_max_sharpe']['diagnostics'].error.eq('').all())


class SimulationResearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.returns = pd.DataFrame(rng.normal(.0002, .01, (100, 2)), index=pd.bdate_range('2020-01-01', periods=100), columns=['A', 'B'])
        self.mu = pd.Series([.06, .03], index=['A', 'B'])
        self.cov = pd.DataFrame([[.04, .01], [.01, .02]], index=['A', 'B'], columns=['A', 'B'])
        self.history = pd.DataFrame({'primary_regime': np.resize(REGIMES, 100)}, index=self.returns.index)

    def simulate(self, **kwargs):
        options = dict(years=1, trading_days=24, n_sims=80, initial_value=100., seed=5,
                       historical_returns=self.returns, regime_history=self.history, block_length=5)
        options.update(kwargs)
        return run_simulation([.6, .4], self.mu, self.cov, **options)

    def test_student_t_seed_dimensions_covariance_and_fan_order(self):
        a = self.simulate(method='student_t', return_paths=True)
        b = self.simulate(method='student_t', return_paths=True)
        self.assertEqual(a['paths'].shape, (24, 80))
        pd.testing.assert_frame_equal(a['paths'], b['paths'])
        self.assertTrue((np.diff(a['percentile_paths'].to_numpy(), axis=1) >= 0).all())
        self.assertNotIn('paths', self.simulate())
        # One-step covariance scaling: df=5 uses sqrt((df-2)/chi-square), not inflated covariance.
        one = run_simulation([1., 0.], self.mu, self.cov, method='student_t', years=1/252,
            n_sims=50000, initial_value=1., seed=3)['terminal_values'] - 1
        self.assertAlmostEqual(np.var(one), .04/252, delta=.04/252 * .08)
        with self.assertRaises(ValueError):
            self.simulate(method='student_t', degrees_of_freedom=2)

    def test_block_bootstrap_keeps_asset_rows_synchronized(self):
        indices = synchronized_block_indices(100, 24, 80, 5, np.random.default_rng(5))
        for start in range(0, 20, 5):
            np.testing.assert_array_equal(np.diff(indices[start:start+5], axis=0), np.ones((4, 80)))
        # Perfectly offset asset returns must produce exactly zero portfolio returns under daily rebalance.
        history = self.returns.copy(); history.B = -history.A
        result = run_simulation([.5, .5], self.mu, self.cov, method='block_bootstrap',
            historical_returns=history, years=1, trading_days=24, n_sims=80,
            initial_value=100., block_length=5, rebalance_frequency='daily')
        np.testing.assert_allclose(result['terminal_values'], 100.)

    def test_transition_scenarios_and_future_regimes(self):
        cutoff = self.history.index[59]
        matrix, meta = estimate_transition_matrix(self.history, cutoff)
        changed = self.history.copy(); changed.loc[changed.index > cutoff, 'primary_regime'] = 'stagflation'
        pd.testing.assert_frame_equal(matrix, estimate_transition_matrix(changed, cutoff)[0])
        for scenario in SCENARIOS:
            adjusted, info = scenario_transition_matrix(matrix, scenario)
            np.testing.assert_allclose(adjusted.sum(axis=1), 1.)
            self.assertTrue((adjusted >= 0).all().all())
        for initial in (*REGIMES, 'Current', 'Random historical'):
            a = self.simulate(method='regime_switching', starting_regime=initial, as_of=cutoff)
            b = self.simulate(method='regime_switching', starting_regime=initial, as_of=cutoff, regime_history=changed)
            np.testing.assert_allclose(a['terminal_values'], b['terminal_values'])
            np.testing.assert_allclose(a['regime_occupancy'].sum(axis=1), 1.)

    def test_sparse_regime_fallback_and_uncertainty(self):
        history = self.history.copy(); history['primary_regime'] = 'goldilocks'
        result = self.simulate(method='regime_switching', starting_regime='stagflation', regime_history=history)
        self.assertEqual(result['metadata']['sparse_fallback']['stagflation']['unconditional_probability'], 1.)
        self.assertTrue(np.isfinite(result['terminal_values']).all())
        for method in ('gaussian', 'student_t', 'block_bootstrap', 'regime_switching'):
            a = self.simulate(method=method, include_estimation_uncertainty=True)
            b = self.simulate(method=method, include_estimation_uncertainty=True)
            np.testing.assert_array_equal(a['terminal_values'], b['terminal_values'])
            self.assertEqual(a['metadata']['parameter_populations'], 32)

    def test_cash_flow_timing_inflation_and_no_negative_wealth(self):
        zero_cov = self.cov * 0
        zero_mu = self.mu * 0
        result = run_simulation([.6, .4], zero_mu, zero_cov, years=2, trading_days=2, n_sims=2,
            initial_value=100., annual_contribution=10., annual_withdrawal=3., contribution_growth=.1,
            inflation=.1, return_paths=True, target_value=114.7)
        np.testing.assert_allclose(result['paths'].iloc[:, 0], [100., 107., 107., 114.7])
        self.assertEqual(result['statistics']['mean_max_drawdown'], 0.)
        self.assertAlmostEqual(result['real_percentile_paths']['median'].iloc[-1], 114.7/1.1**2)
        depleted = run_simulation([.6, .4], zero_mu, zero_cov, years=2, trading_days=2, n_sims=2,
            initial_value=100., annual_withdrawal=200.)
        self.assertEqual(depleted['statistics']['prob_unmet_withdrawal'], 1.)
        self.assertTrue((depleted['terminal_values'] == 0).all())
        floored = run_simulation([1., 0.], pd.Series([-4., 0.], index=['A', 'B']), zero_cov,
            years=1, trading_days=1, n_sims=2)
        self.assertEqual(floored['metadata']['floored_asset_draws'], 2)
        self.assertTrue((floored['terminal_values'] == 0).all())

    def test_terminal_statistics_and_legacy_interface(self):
        s = simulation_statistics([50., 100., 200., 300.], [-.6, -.4, -.2, 0.], 100., 200.)
        self.assertEqual(s['prob_loss'], .25)
        self.assertEqual(s['prob_double'], .5)
        self.assertEqual(s['prob_goal'], .5)
        self.assertEqual(s['prob_drawdown_30'], .5)
        self.assertEqual(s['median_terminal_value'], 150.)
        paths = simulate_portfolio_paths([.6, .4], self.mu, self.cov, years=1, trading_days=2, n_sims=2, method='student_t')
        self.assertEqual(paths.shape, (2, 2))


class AnalyticsTests(unittest.TestCase):
    def test_metrics_alignment_rolling_and_stress_missing(self):
        index = pd.bdate_range('2020-01-01', periods=800)
        r = pd.Series(np.resize([.01, -.005], 800), index=index)
        m = backtest_metrics(r, r)
        self.assertAlmostEqual(m['beta'], 1.)
        self.assertAlmostEqual(m['alpha'], 0.)
        self.assertAlmostEqual(m['tracking_error'], 0.)
        rolling = rolling_backtest_metrics(r)
        self.assertTrue(rolling.return_3y.iloc[:755].isna().all())
        self.assertTrue(np.isfinite(rolling.return_3y.iloc[755]))
        prices = pd.DataFrame({'A': [100., 90., 80.], 'B': [np.nan, 100., 100.]}, index=index[:3])
        events = {'Test': (str(index[0].date()), str(index[2].date()))}
        partial = historical_stress_tests(prices, {'A': .5, 'B': .5}, events)
        self.assertEqual(partial.status.iloc[0], 'incomplete coverage')
        self.assertTrue(np.isnan(partial['return'].iloc[0]))
        complete = historical_stress_tests(prices, {'A': 1.}, events)
        self.assertAlmostEqual(complete['return'].iloc[0], -.2)
        self.assertAlmostEqual(complete.max_drawdown.iloc[0], -.2)

    def test_regime_conditioning_has_explicit_counts(self):
        dates = pd.bdate_range('2020-01-01', periods=4)
        r = pd.Series([.1, -.1, .1, -.1], index=dates)
        labels = pd.Series(['goldilocks', 'slowdown', 'goldilocks', 'unavailable'], index=dates)
        table = regime_backtest_metrics(r, labels)
        self.assertEqual(table.observations.sum(), 3)
        self.assertEqual(table.set_index('regime').loc['goldilocks', 'positive_frequency'], 1.)


class ResearchUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prices, yields, current, policy = make_data()
        cls.data = {'prices': prices, 'yields': yields, 'data_quality': [], 'sources': {}}
        cls.bundle = services.run_analysis(cls.data, current, policy, {'robustness_repetitions': 2}, .03, prices.index[-1])
        cls.bundle.update(completed_at='test', config={}, portfolio_value=10000.)

    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'ui' / 'app.py'), default_timeout=40)
        app.session_state['analysis'] = copy.deepcopy(self.bundle)
        return app

    def test_backtest_page_real_execution_and_widget_changes_do_not_download(self):
        app = self.app()
        with patch('ui.services.market_data') as downloader:
            app.run(); app.radio(key='page').set_value('Backtest').run()
            app.multiselect(key='bt_strategies').set_value(['current', 'strategic']).run()
            app.button(key='run_backtest').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertIn('backtest_result', app.session_state)
            saved = app.session_state['backtest_result'][0]['returns'].copy()
            app.number_input(key='bt_cost').set_value(20.).run()
            pd.testing.assert_frame_equal(saved, app.session_state['backtest_result'][0]['returns'])
            downloader.assert_not_called()

    def test_simulation_page_real_execution_and_compact_results(self):
        app = self.app()
        with patch('ui.services.market_data') as downloader:
            app.run(); app.radio(key='page').set_value('Simulation').run()
            app.number_input(key='sim_years').set_value(1).run()
            app.number_input(key='sim_count').set_value(100).run()
            app.selectbox(key='sim_method').set_value('block_bootstrap').run()
            app.button(key='run_simulation').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            output = app.session_state['simulation_result'][0]
            self.assertNotIn('paths', output)
            self.assertEqual(len(output['terminal_values']), 100)
            downloader.assert_not_called()
            for method in ('student_t', 'regime_switching'):
                app.selectbox(key='sim_method').set_value(method).run()
                if method == 'regime_switching':
                    app.selectbox(key='sim_starting').set_value('Random historical').run()
                    app.selectbox(key='sim_scenario').set_value('Severe financial stress').run()
                app.button(key='run_simulation').click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(app.session_state['simulation_result'][0]['metadata']['method'], method)

    def test_backtest_page_shows_coverage_failures_when_no_strategy_succeeds(self):
        app = self.app()
        app.session_state['analysis']['research_data']['prices'].iloc[-3, 0] = np.nan
        app.run()
        app.radio(key='page').set_value('Backtest').run()
        app.button(key='run_backtest').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        warnings = ' '.join(w.value for w in app.warning)
        self.assertIn('SPY', warnings)
        self.assertIn('No strategies have usable history', warnings)

    def test_available_history_button_updates_existing_start_date(self):
        app = self.app()
        prices = app.session_state['analysis']['research_data']['prices']
        prices.iloc[:60, prices.columns.get_loc('IEF')] = np.nan
        app.run()
        app.radio(key='page').set_value('Backtest').run()
        app.date_input(key='bt_start').set_value(prices.index[20].date()).run()
        app.button(key='bt_available_history').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.date_input(key='bt_start').value, prices.index[312].date())
        app.button(key='run_backtest').click().run()
        self.assertFalse(app.error)
        self.assertFalse(app.session_state['backtest_result'][0]['failures'])

    def test_conviction_uses_existing_evidence(self):
        diagnostics = recommendation_diagnostics(self.bundle['recommendation'])
        self.assertEqual(set(diagnostics.Asset), set(self.bundle['recommendation']['policy']))
        self.assertTrue(diagnostics['10th percentile'].notna().all())
        self.assertTrue(diagnostics.Conviction.isin(['HIGH', 'MEDIUM', 'LOW']).all())


if __name__ == '__main__':
    unittest.main()
