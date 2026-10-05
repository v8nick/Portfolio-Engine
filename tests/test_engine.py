"""Small offline regression suite for economic units, timing, and uncertainty."""
import json
import tempfile
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import pandas as pd

from black_litterman import black_litterman_posterior, resolve_strategic_prior_weights
from data import compute_returns, download_treasury_yields, resolve_risk_free_rate
from market_intelligence import (build_market_snapshot, classify_curve_movement,
    curve_spread_bp, get_regime_features, load_market_data, relative_returns,
    rolling_zscore, yield_change_bp)
from mpt import annualize_mean_cov, optimize_max_sharpe_constrained
from report import max_drawdown
from simulation import simulate_portfolio_paths, terminal_value_stats, path_max_drawdown
from walkforward import (get_rebalance_dates, portfolio_return_series_from_weight_history,
                         rolling_black_litterman_backtest)


class MarketTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range('2024-01-01', periods=260)
        self.prices = pd.DataFrame({'SPY': np.linspace(100, 150, 260),
                                    'QQQ': np.linspace(100, 180, 260)}, index=self.index)
        self.yields = pd.DataFrame({'2Y': np.linspace(4, 4.2, 260),
                                    '10Y': np.linspace(4, 4.5, 260)}, index=self.index)

    def test_relative_returns_not_price_ratios(self):
        frame = pd.DataFrame({'QQQ': [100, 110], 'SPY': [200, 210]}, index=self.index[:2])
        self.assertAlmostEqual(relative_returns(frame, 'QQQ', 'SPY').iloc[-1], .05)
        snapshot = build_market_snapshot(self.prices, self.yields)
        for label, n in [('1D', 1), ('5D', 5), ('21D', 21), ('63D', 63), ('252D', 252)]:
            expected = self.prices['QQQ'].iloc[-1] / self.prices['QQQ'].iloc[-n-1] - self.prices['SPY'].iloc[-1] / self.prices['SPY'].iloc[-n-1]
            self.assertAlmostEqual(snapshot['relative_performance']['QQQ_SPY']['returns'][label], expected)

    def test_basis_points_and_curve(self):
        self.assertAlmostEqual(yield_change_bp(4.1, 4), 10)
        self.assertAlmostEqual(curve_spread_bp(4.2, 4.5), 30)
        snapshot = build_market_snapshot(self.prices, self.yields)
        self.assertAlmostEqual(snapshot['curve']['spread_2s10s_bp'], 30)
        self.assertEqual(snapshot['curve']['daily_classification'], 'bear_steepener')
        self.assertAlmostEqual(snapshot['rates']['10Y']['changes']['21D'], (.5 / 259) * 21 * 100)

    def test_curve_classification(self):
        cases = [(-10, -5, 'bull_steepener'), (-5, -10, 'bull_flattener'),
                 (5, 10, 'bear_steepener'), (10, 5, 'bear_flattener'),
                 (0, 0, 'unchanged'), (5, 5, 'parallel_shift'),
                 (-5, 5, 'twist_steepener'), (5, -5, 'twist_flattener'),
                 (np.nan, 5, 'unavailable')]
        for short, long, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_curve_movement(short, long), expected)

    def test_missing_history_stale_and_json(self):
        snapshot = build_market_snapshot(self.prices.iloc[:3], pd.DataFrame())
        self.assertIsNone(snapshot['rates']['2Y']['value'])
        self.assertIsNone(snapshot['relative_performance']['RSP_SPY']['returns']['1D'])
        self.assertIsNone(snapshot['equities']['SPY']['changes']['5D'])
        self.assertEqual(snapshot['curve']['daily_classification'], 'unavailable')
        self.assertTrue(snapshot['data_quality'])
        json.dumps(snapshot, allow_nan=False)
        empty = build_market_snapshot(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(get_regime_features(empty), {})
        stale = build_market_snapshot(self.prices, self.yields, as_of=self.index[-1] + pd.Timedelta(days=10))
        self.assertTrue(stale['equities']['SPY']['stale'])
        self.assertEqual(get_regime_features(stale), {})

    def test_snapshot_truncation_and_numeric_features(self):
        snapshot = build_market_snapshot(self.prices, self.yields, as_of=self.index[100])
        self.assertAlmostEqual(snapshot['equities']['SPY']['value'], self.prices['SPY'].iloc[100])
        self.assertTrue(all(isinstance(x, float) and np.isfinite(x) for x in get_regime_features(snapshot).values()))
        self.assertIsNone(snapshot['equities']['SPY']['changes']['252D'])

    def test_zscore_and_no_price_forward_fill(self):
        values = pd.Series([1., 2., 3.])
        self.assertAlmostEqual(rolling_zscore(values, 3).iloc[-1], 1)
        self.assertTrue(rolling_zscore(pd.Series([1., 1., 1.]), 3).isna().all())
        frame = pd.DataFrame({'A': [100, np.nan, 110]}, index=self.index[:3])
        self.assertTrue(compute_returns(frame).empty)

    def test_risk_free_no_future_and_fallback(self):
        yields = pd.DataFrame({'3MO': [4., 5.]}, index=pd.to_datetime(['2024-01-02', '2024-02-01']))
        with patch('config.shared.RISK_FREE_RATE', None), patch('config.shared.RISK_FREE_FALLBACK', .01):
            rate = resolve_risk_free_rate(as_of='2024-01-03', yields=yields)
            self.assertEqual(rate['rate'], .04)
            self.assertEqual(rate['source'], 'FRED DGS3MO')
            self.assertEqual(resolve_risk_free_rate(as_of='2024-01-03', yields=yields, override=.03)['rate'], .03)
            self.assertEqual(resolve_risk_free_rate(as_of='2024-01-20', yields=yields)['rate'], .01)
            self.assertEqual(resolve_risk_free_rate(as_of='2024-01-03', yields=pd.DataFrame())['rate'], .01)

    def test_fred_adapter_and_outage(self):
        response = unittest.mock.Mock()
        response.text = 'observation_date,DGS3MO,DGS2,DGS10\n2024-01-02,5.3,4.3,3.9\n2024-01-03,.,4.4,4.0\n'
        with patch('requests.get', return_value=response):
            frame = download_treasury_yields('2024-01-02', '2024-01-03')
            self.assertEqual(frame['2Y'].iloc[-1], 4.4)
            self.assertTrue(pd.isna(frame['3MO'].iloc[-1]))
            self.assertTrue(frame['5Y'].isna().all())
        import requests
        with patch('requests.get', side_effect=requests.RequestException('offline')), warnings.catch_warnings(record=True):
            self.assertTrue(download_treasury_yields('2024-01-01').empty)

    def test_cached_market_data_and_partial_outage(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch('market_intelligence.download_prices', return_value=self.prices) as price_loader, patch('market_intelligence.download_treasury_yields', return_value=self.yields) as yield_loader:
                load_market_data('2024-01-01', cache_dir=cache)
                loaded = load_market_data('2024-01-01', cache_dir=cache)
                self.assertEqual(price_loader.call_count, 1)
                self.assertEqual(yield_loader.call_count, 1)
                self.assertTrue(loaded['cached']['prices'])
            with patch('market_intelligence.download_prices', side_effect=RuntimeError('unavailable')), patch('market_intelligence.download_treasury_yields', return_value=self.yields):
                loaded = load_market_data('2024-02-01', cache_dir=cache)
                self.assertTrue(loaded['prices'].empty)
                self.assertFalse(loaded['yields'].empty)
                self.assertTrue(loaded['data_quality'])


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.cov = pd.DataFrame(np.diag([.04, .09]), index=['A', 'B'], columns=['A', 'B'])
        self.weights = np.array([.5, .5])

    def posterior(self, confidence, **kwargs):
        return black_litterman_posterior(self.cov, self.weights, 2.5, .05, {'A': (.2, confidence)}, **kwargs)

    def test_bl_confidence_strength_and_boundaries(self):
        low, prior = self.posterior(.1)
        high, _ = self.posterior(.9)
        zero, _ = self.posterior(0)
        exact, _ = self.posterior(1)
        self.assertGreater(high['A'], low['A'])
        self.assertAlmostEqual(high['A'], prior['A'] + .9 * (.2 - prior['A']))
        np.testing.assert_allclose(zero, prior)
        self.assertAlmostEqual(exact['A'], .2)
        with self.assertRaises(ValueError):
            self.posterior(1.1)

    def test_bl_relative_only_and_total_return_prior(self):
        mu, pi = black_litterman_posterior(self.cov, self.weights, 2.5, .05, relative_views=[('A', 'B', .1, .8)], risk_free_rate=.03)
        self.assertGreater(mu['A'] - mu['B'], pi['A'] - pi['B'])
        self.assertAlmostEqual(pi['A'], .08)
        unchanged, _ = black_litterman_posterior(self.cov, self.weights, 2.5, .05)
        np.testing.assert_allclose(unchanged, pi - .03)

    def test_strategic_fallback_independent_of_holdings(self):
        with warnings.catch_warnings(record=True) as caught:
            weights = resolve_strategic_prior_weights(['A', 'B'], None)
            np.testing.assert_allclose(weights, [.5, .5])
            self.assertIn('fallback', str(caught[0].message))
        with self.assertRaises(ValueError):
            resolve_strategic_prior_weights(['A', 'B'], {'A': .9})

    def test_month_end_uses_last_tradable_observation(self):
        index = pd.bdate_range('2024-08-01', '2024-09-30')
        dates = get_rebalance_dates(index)
        self.assertEqual(dates[0], pd.Timestamp('2024-08-30'))

    def test_weights_become_effective_next_day(self):
        index = pd.bdate_range('2024-01-01', periods=4)
        returns = pd.DataFrame({'A': [.01, .8, .03, .04], 'B': [0., .1, .2, .3]}, index=index)
        history = pd.DataFrame([[1., 0.], [0., 1.]], index=[index[1], index[2]], columns=['A', 'B'])
        out = portfolio_return_series_from_weight_history(returns, history)
        self.assertNotIn(index[1], out.index)
        self.assertAlmostEqual(out.loc[index[2]], .03)
        self.assertAlmostEqual(out.loc[index[3]], .3)

    def test_walkforward_estimation_and_future_perturbation(self):
        index = pd.bdate_range('2024-01-01', '2024-04-05')
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.normal(.001, .01, (len(index), 2)), index=index, columns=['A', 'B'])
        windows = []
        def estimate(sample, **kwargs):
            windows.append(sample.copy())
            return sample.mean() * 252, self.cov
        def optimize(mu, **kwargs):
            return np.array([.7, .3]) if mu['A'] > mu['B'] else np.array([.3, .7])
        kwargs = dict(tickers=['A', 'B'], trading_days=252, risk_free_rate=.01, weight_bounds=(0, 1), fixed_weights={}, min_weights={}, window_days=10, use_black_litterman=False)
        with patch('walkforward.annualize_mean_cov', side_effect=estimate), patch('walkforward.optimize_max_sharpe_constrained', side_effect=optimize):
            out, history, diagnostics = rolling_black_litterman_backtest(returns, **kwargs)
            modified = returns.copy()
            modified.loc[modified.index > history.index[0], 'A'] = .5
            _, changed, _ = rolling_black_litterman_backtest(modified, **kwargs)
        np.testing.assert_allclose(history.iloc[0], changed.iloc[0])
        self.assertNotIn(history.index[0], out.index)
        for dt, window in zip(history.index, windows):
            self.assertEqual(len(window), 10)
            self.assertEqual(window.index[-1], dt)
            self.assertGreater(diagnostics.loc[dt, 'effective_date'], dt)

    def test_empty_walkforward(self):
        returns = pd.DataFrame([[.01, .02]], columns=['A', 'B'], index=pd.to_datetime(['2024-01-02']))
        out, weights, diagnostics = rolling_black_litterman_backtest(returns, ['A', 'B'], 252, 0, (0, 1), {}, {}, use_black_litterman=False)
        self.assertTrue(out.empty and weights.empty and diagnostics.empty)

    def test_existing_shrinkage_and_constraint_solver(self):
        rng = np.random.default_rng(1)
        returns = pd.DataFrame(rng.normal(.001, .01, (100, 2)), columns=['A', 'B'])
        mu, cov = annualize_mean_cov(returns, use_shrinkage=True)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
        weights = optimize_max_sharpe_constrained(pd.Series([.1, .15], index=['A', 'B']), cov, .01, ['A', 'B'], fixed_weights={'A': .35})
        self.assertAlmostEqual(weights.sum(), 1, places=7)
        self.assertAlmostEqual(weights[0], .35, places=7)

    def test_simulation_initial_value_and_initial_drawdown(self):
        paths = pd.DataFrame([[80., 150.], [90., 210.]])
        paths.attrs['initial_value'] = 100.
        stats = terminal_value_stats(paths)
        self.assertEqual(stats['prob_loss'], .5)
        self.assertEqual(stats['prob_double'], .5)
        self.assertAlmostEqual(path_max_drawdown(paths[0], 100), -.2)
        self.assertAlmostEqual(max_drawdown(pd.Series([-.2, .1])), -.2)
        generated = simulate_portfolio_paths(self.weights, pd.Series([.1, .1]), self.cov, years=1, n_sims=2, trading_days=2, initial_value=100)
        self.assertEqual(generated.attrs['method'], 'gaussian')
        self.assertEqual(generated.attrs['initial_value'], 100)
        with self.assertRaises(ValueError):
            simulate_portfolio_paths(self.weights, pd.Series([.1, .1]), self.cov, method='bootstrap')


class IntegrationTests(unittest.TestCase):
    def test_existing_live_and_research_workflows_offline(self):
        import contextlib
        import io
        import main_live
        import main_research
        import matplotlib.pyplot as plt
        from report import summary_table
        rng = np.random.default_rng(2024)
        index = pd.bdate_range('2019-01-01', periods=900)
        tickers = main_live.TICKERS + ['SPY']
        prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(.0005, .012, (900, len(tickers))), axis=0), index=index, columns=tickers)
        yields = pd.DataFrame({'3MO': np.full(900, 3.)}, index=index)
        factors = pd.DataFrame(rng.normal(0, .005, (900, 5)), index=index, columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
        factors['RF'] = .03 / 252
        output = io.StringIO()
        with contextlib.redirect_stdout(output), warnings.catch_warnings(record=True), patch('main_live.download_prices', return_value=prices), patch('main_live.download_treasury_yields', return_value=yields):
            main_live.main()
        with contextlib.redirect_stdout(output), warnings.catch_warnings(record=True), patch('main_research.download_prices', return_value=prices), patch('main_research.download_treasury_yields', return_value=yields), patch('factors.load_ff5_daily', return_value=factors), patch('main_research.MC_N_SIMS', 3), patch('main_research.MC_YEARS', 1), patch('main_research.export_quantstats_report') as exporter, patch('main_research.plt.show'), patch('main_research.summary_table', wraps=summary_table) as summary:
            main_research.main()
            self.assertEqual(exporter.call_count, 2)
            for call in summary.call_args_list:
                self.assertAlmostEqual(call.args[2], .03)
                self.assertEqual(call.args[3], 252)
        plt.close('all')
        self.assertIn('TARGET PORTFOLIO', output.getvalue())
        self.assertIn('IID GAUSSIAN BASELINE', output.getvalue())

    def test_market_console_and_structured_output(self):
        import contextlib
        import io
        import market_intelligence
        index = pd.bdate_range('2024-01-01', periods=70)
        prices = pd.DataFrame({'SPY': np.arange(70) + 100., 'QQQ': np.arange(70) + 50.}, index=index)
        data = {'prices': prices, 'yields': pd.DataFrame(index=index), 'data_quality': [], 'sources': {}}
        for mode in ([], ['--json']):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch('market_intelligence.load_market_data', return_value=data), patch('sys.argv', ['market_intelligence.py', '--end', index[-1].date().isoformat()] + mode):
                market_intelligence.main()
            if mode:
                result = json.loads(output.getvalue())
                self.assertIn('features', result)
                self.assertIn('rates', result)
            else:
                self.assertIn('Market Intelligence', output.getvalue())
                self.assertIn('HYG minus LQD', output.getvalue())


if __name__ == '__main__':
    unittest.main()
