"""Small presentation/adapter tests with deterministic data; no UI internals mocked."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from ui.helpers import (initial_editor, validate_editor, number, reason_text,
                        allocation_table, metric_table, action_table, macro_input_table, import_portfolio_csv,
                        complete_editor_rows, prepare_editor_policy)
from ui import services
from ui.views import PAGES
from config import shared
from market_intelligence import MARKET_SYMBOLS
from portfolio_decision import build_portfolio_recommendation

ROOT = Path(__file__).resolve().parents[1]


class HelperTests(unittest.TestCase):
    def test_formats_and_missing(self):
        self.assertEqual(number(.234, 'percent'), '23.4%')
        self.assertEqual(number(5.34, 'yield'), '5.34%')
        self.assertEqual(number(6.3, 'bp'), '+6.3 bp')
        self.assertEqual(number(1250000, 'money'), '$1,250,000')
        self.assertEqual(number(float('nan')), 'N/A')
        self.assertEqual(number(None), 'N/A')

    def test_macro_inputs_preserve_units_signs_and_missing_values(self):
        regime = {'features': {'growth_iwm_spy_21D': -.012,
                              'inflation_10y_change_bp_63D': 25.,
                              'stress_spy_realized_vol_21d': .18}}
        before = copy.deepcopy(regime)
        frame = macro_input_table(regime).set_index('Metric')
        self.assertEqual(frame.loc['Small caps − large caps (IWM − SPY)', '21 sessions (~1 month)'], '-1.2%')
        self.assertEqual(frame.loc['Small caps − large caps (IWM − SPY)', '63 sessions (~3 months)'], 'N/A')
        self.assertEqual(frame.loc['10-year Treasury yield change', '63 sessions (~3 months)'], '+25.0 bp')
        self.assertEqual(frame.loc['SPY realized volatility (21 sessions, annualized)', 'Latest level'], '18.0%')
        self.assertEqual(regime, before)

    def test_reason_codes_unknown_safe(self):
        self.assertIn('New reason.', reason_text(['new_reason']))
        self.assertIn('rebalance threshold', reason_text(['within_no_trade_band']))

    def test_csv_tickers_merge_preserves_edits_and_initializes_new_rows(self):
        existing = initial_editor()
        existing.loc[existing.Ticker == 'SPY', 'Current %'] = 12.
        before = existing.copy(deep=True)
        imported = import_portfolio_csv(b'\xef\xbb\xbfTicker\n spy \nNEWETF\n', existing)
        self.assertEqual(len(imported), len(existing) + 1)
        self.assertEqual(imported.set_index('Ticker').loc['SPY', 'Current %'], 12.)
        new = imported.set_index('Ticker').loc['NEWETF']
        self.assertEqual(new['Current %'], 0.)
        self.assertEqual(new['Strategic %'], 0.)
        self.assertEqual(new['Role'], 'satellite')
        pd.testing.assert_frame_equal(existing, before)

    def test_simple_editor_preserves_rules_and_defaults_new_rows(self):
        existing = initial_editor()
        pd.testing.assert_frame_equal(complete_editor_rows(existing), existing)
        new = pd.DataFrame([{'Ticker': 'SPY', 'Current %': 60., 'Strategic %': 60.},
                            {'Ticker': 'NEWSTOCK', 'Current %': 40., 'Strategic %': 0.}])
        completed = complete_editor_rows(new)
        self.assertEqual(completed.loc[0, 'Low %'], 55.)
        self.assertEqual(completed.loc[0, 'High %'], 65.)
        self.assertEqual(completed.loc[0, 'Role'], 'core')
        self.assertEqual(completed.loc[0, 'Group'], 'us_equity')
        self.assertEqual(completed.loc[1, 'Role'], 'satellite')
        self.assertEqual(completed.loc[1, 'Low %'], 0.)
        self.assertEqual(completed.loc[1, 'High %'], 5.)
        self.assertNotIn('Role', new.columns)

    def test_main_targets_override_hidden_and_conflicting_advanced_limits(self):
        frame = initial_editor().set_index('Ticker').loc[['SPY', 'IEF']].reset_index()
        frame['Current %'] = [60., 40.]
        frame['Strategic %'] = [60., 40.]
        frame.loc[0, 'Fixed %'] = 35.
        before = frame.copy(deep=True)
        for advanced in (False, True):
            effective, total_cap, individual_cap, notes = prepare_editor_policy(frame, advanced)
            holdings, policy, errors = validate_editor(effective, total_cap, individual_cap)
            self.assertEqual(errors, [], (advanced, errors))
            self.assertEqual(holdings, {'SPY': .6, 'IEF': .4})
            self.assertEqual(policy['SPY']['strategic_weight'], .6)
            self.assertNotIn('fixed_weight', policy['SPY'])
            self.assertLessEqual(policy['SPY']['tactical_low'], .6)
            self.assertGreaterEqual(policy['SPY']['maximum_weight'], .6)
            if advanced:
                self.assertTrue(notes)
        pd.testing.assert_frame_equal(frame, before)

    def test_optional_fields_and_caps_accommodate_satellite_targets(self):
        frame = pd.DataFrame([{'Ticker': 'MSFT', 'Current %': 60., 'Strategic %': 60.},
                              {'Ticker': 'IEF', 'Current %': 40., 'Strategic %': 40.}])
        effective, total_cap, individual_cap, notes = prepare_editor_policy(frame)
        self.assertEqual(total_cap, .6)
        self.assertEqual(individual_cap, .6)
        self.assertTrue(notes)
        self.assertEqual(effective.loc[0, 'Role'], 'satellite')
        self.assertEqual(validate_editor(effective, total_cap, individual_cap)[2], [])
        frame.loc[0, 'Strategic %'] = -1.
        effective, total_cap, individual_cap, _ = prepare_editor_policy(frame)
        self.assertTrue(validate_editor(effective, total_cap, individual_cap)[2])

    def test_csv_replace_and_percentage_units(self):
        imported = import_portfolio_csv(b'Symbol,Weight,Strategic %,Role,Group\nSPY,60%,60,core,stocks\nIEF,40,40,core,bonds\n',
                                        initial_editor(), replace=True)
        self.assertEqual(list(imported.Ticker), ['SPY', 'IEF'])
        self.assertEqual(list(imported['Current %']), [60., 40.])
        self.assertEqual(list(imported.Group), ['stocks', 'bonds'])
        # Fractions are never guessed to be percentages or silently normalized.
        small = import_portfolio_csv(b'Ticker,Current %\nSPY,0.25\n', initial_editor(), replace=True)
        self.assertEqual(small['Current %'].iloc[0], .25)

    def test_csv_headerless_and_fixed_weight_roundtrip(self):
        imported = import_portfolio_csv(b'spy\n\nief\n', initial_editor(), replace=True)
        self.assertEqual(list(imported.Ticker), ['SPY', 'IEF'])
        frame = initial_editor()
        frame.loc[0, 'Fixed %'] = 35.
        restored = import_portfolio_csv(frame.to_csv(index=False).encode(), initial_editor(), replace=True)
        self.assertEqual(restored['Fixed %'].iloc[0], 35.)
        cleared = import_portfolio_csv(b'Ticker,Fixed %\nSPY,\n', restored)
        self.assertTrue(pd.isna(cleared['Fixed %'].iloc[0]))

    def test_csv_rejects_invalid_input_without_changing_editor(self):
        existing = initial_editor()
        before = existing.copy(deep=True)
        for content in (b'', b'Ticker\n', b'Ticker\nSPY\nspy\n', b'Ticker,Quantity\nSPY,2\n',
                        b'Ticker,Current %\nSPY,-1\n', b'Ticker,Current %\nSPY,nan\n',
                        b'Ticker,Current %\nSPY,101\n', b'Ticker,Role\nSPY,other\n',
                        b'Ticker\nSPY,2\n', b'\xff', b'Ticker\n<script>\n'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                import_portfolio_csv(content, existing)
            pd.testing.assert_frame_equal(existing, before)

    def test_default_policy_valid_and_not_mutated(self):
        frame = initial_editor()
        before = frame.copy(deep=True)
        holdings, policy, errors = validate_editor(frame, .15, .05)
        self.assertEqual(errors, [])
        self.assertAlmostEqual(sum(holdings.values()), 1)
        self.assertAlmostEqual(sum(p['strategic_weight'] for p in policy.values()), 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_invalid_strategic_sum(self):
        frame = initial_editor()
        frame.loc[0, 'Strategic %'] -= 1
        errors = validate_editor(frame, .15, .05)[2]
        self.assertTrue(any('Target % values total' in x for x in errors))
        self.assertFalse(any('Strategic' in x for x in errors))

    def test_invalid_tactical_and_hard_bands(self):
        frame = initial_editor()
        frame.loc[0, 'Low %'] = 50
        self.assertTrue(any('tactical' in x for x in validate_editor(frame, .15, .05)[2]))
        frame = initial_editor()
        frame.loc[0, 'Minimum %'] = 90
        self.assertTrue(any('minimum' in x for x in validate_editor(frame, .15, .05)[2]))

    def test_invalid_current_duplicates_and_fixed(self):
        frame = initial_editor()
        frame.loc[0, 'Current %'] = -1
        self.assertTrue(validate_editor(frame, .15, .05)[2])
        frame = initial_editor()
        frame.loc[1, 'Ticker'] = frame.loc[0, 'Ticker']
        self.assertTrue(any('duplicate' in x for x in validate_editor(frame, .15, .05)[2]))
        frame = initial_editor()
        frame.loc[0, 'Fixed %'] = 33
        self.assertTrue(any('fixed' in x for x in validate_editor(frame, .15, .05)[2]))


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(10)
        dates = pd.bdate_range('2020-01-01', periods=650)
        cls.prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0003, .01, (650, len(MARKET_SYMBOLS))), axis=0)),
                                  columns=list(MARKET_SYMBOLS), index=dates)
        cls.yields = pd.DataFrame({'2Y': 3 + rng.normal(0, .2, 650), '10Y': 4 + rng.normal(0, .2, 650), '3MO': 3.}, index=dates)
        cls.holdings = {'SPY': .5, 'IEF': .3, 'GLD': .2}
        cls.policy = {t: dict(strategic_weight=w, minimum_weight=0., maximum_weight=1.,
                             tactical_low=w-.05, tactical_high=w+.05, role='core', group=t) for t, w in cls.holdings.items()}
        cls.options = {'robustness_repetitions': 3}
        cls.data = {'prices': cls.prices, 'yields': cls.yields, 'data_quality': [], 'sources': {}}
        cls.as_of = dates[-1].isoformat()
        cls.bundle = services.run_analysis(cls.data, cls.holdings, cls.policy, cls.options, .03, cls.as_of)
        cls.bundle.update(portfolio_value=1000000, config={}, completed_at=cls.as_of)

    def app(self):
        return AppTest.from_file(str(ROOT / 'ui' / 'app.py'), default_timeout=20)

    def test_first_run_no_backend_or_download(self):
        with patch('ui.services.run_analysis') as engine, patch('ui.services.market_data') as provider:
            app = self.app().run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state['analysis'])
            engine.assert_not_called()
            provider.assert_not_called()

    def test_basic_advanced_toggle_preserves_editor_and_saved_analysis(self):
        app = self.app()
        app.session_state['analysis'] = copy.deepcopy(self.bundle)
        frame = initial_editor()
        frame.loc[0, 'Current %'] = 12.
        app.session_state['editor_base'] = frame
        with patch('ui.services.run_analysis') as engine, patch('ui.services.market_data') as provider:
            app.run()
            self.assertFalse(app.checkbox(key='advanced_allocations').value)
            app.checkbox(key='advanced_allocations').set_value(True).run()
            app.checkbox(key='advanced_allocations').set_value(False).run()
            app.radio(key='page').set_value('Risk').run()
            self.assertFalse(app.exception)
            pd.testing.assert_frame_equal(app.session_state['draft_portfolio'], frame)
            self.assertEqual(app.session_state['analysis']['recommendation'], self.bundle['recommendation'])
            engine.assert_not_called()
            provider.assert_not_called()

    def test_renamed_macro_page_preserves_existing_session(self):
        app = self.app()
        app.session_state['page'] = 'Regime analysis'
        app.session_state['analysis'] = copy.deepcopy(self.bundle)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['page'], 'Macro indicators')

    def test_adapter_matches_direct_backend(self):
        direct = build_portfolio_recommendation(self.prices, self.yields, self.holdings, self.policy,
                                                as_of=self.as_of, risk_free_rate=.03, options=self.options)
        self.assertEqual(self.bundle['recommendation'], direct)

    def test_macro_views_show_evidence_without_economic_classification(self):
        app = self.app()
        app.session_state['analysis'] = copy.deepcopy(self.bundle)
        with patch('ui.services.run_analysis') as engine, patch('ui.services.market_data') as provider:
            app.run()
            for page in ('Executive overview', 'Macro indicators'):
                app.radio(key='page').set_value(page).run()
                self.assertFalse(app.exception)
                self.assertTrue(any('10-year Treasury yield change' in list(df.value.get('Metric', []))
                                    for df in app.dataframe))
                labels = [metric.label for metric in app.metric]
                self.assertNotIn('Primary regime', labels)
                self.assertNotIn('Regime confidence', labels)
                self.assertFalse(any('Primary:' in text.value for text in app.markdown))
            self.assertIn('Available input weight', app.dataframe[1].value.columns)
            engine.assert_not_called()
            provider.assert_not_called()
        self.assertEqual(app.session_state['analysis']['recommendation'], self.bundle['recommendation'])

    def test_helpers_do_not_mutate_json_result(self):
        result = json.loads(json.dumps(self.bundle['recommendation'], allow_nan=False))
        original = copy.deepcopy(result)
        for helper in (allocation_table, metric_table, action_table):
            self.assertFalse(helper(result).empty)
        self.assertEqual(result, original)

    def test_all_pages_render_failed_optimizer_and_preserve_state(self):
        bundle = copy.deepcopy(self.bundle)
        bundle['recommendation']['candidate_allocations']['cvar'] = {'status': 'unavailable', 'weights': None}
        bundle['recommendation']['warnings'].append('cvar unavailable: synthetic test failure')
        before = copy.deepcopy(bundle)
        app = self.app()
        app.session_state['analysis'] = bundle
        with patch('ui.services.run_analysis') as engine, patch('ui.services.market_data') as provider:
            app.run()
            for page in PAGES:
                app.radio(key='page').set_value(page).run()
                self.assertFalse(app.exception, (page, [e.message for e in app.exception]))
            # Chart-only changes must not run the engine either.
            app.radio(key='page').set_value('Market intelligence').run()
            app.selectbox(key='market_horizon').set_value('63D').run()
            engine.assert_not_called()
            provider.assert_not_called()
        self.assertEqual(app.session_state['analysis']['recommendation'], before['recommendation'])
        pd.testing.assert_frame_equal(app.session_state['analysis']['outcomes'], before['outcomes'])

    def test_missing_market_data_warns_without_crash(self):
        empty = {'prices': pd.DataFrame(index=pd.DatetimeIndex([])), 'yields': pd.DataFrame(index=pd.DatetimeIndex([])),
                 'data_quality': ['Treasury source unavailable.']}
        app = self.app()
        app.session_state['market_preview'] = services.snapshot(empty, self.as_of)
        app.run()
        app.radio(key='page').set_value('Market intelligence').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.warning)

    def test_invalid_inputs_prevent_engine_call(self):
        app = self.app()
        frame = initial_editor()
        frame.loc[0, 'Strategic %'] = 34
        app.session_state['editor_base'] = frame
        with patch('ui.services.run_analysis') as engine, patch('ui.services.market_data') as provider:
            app.run()
            app.button(key="run_analysis").click().run()
            engine.assert_not_called()
            provider.assert_not_called()
            self.assertTrue(app.error)

    def test_failed_run_retains_last_success(self):
        app = self.app()
        app.session_state['analysis'] = copy.deepcopy(self.bundle)
        with patch('ui.services.market_data', return_value=self.data), patch('ui.services.run_analysis', side_effect=ValueError('Missing required portfolio price series: TEST')):
            app.run()
            app.button(key="run_analysis").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(app.session_state['analysis']['recommendation'], self.bundle['recommendation'])

    def test_refresh_does_not_run_engine(self):
        app = self.app()
        with patch('ui.services.market_data', return_value=self.data) as provider, patch('ui.services.run_analysis') as engine:
            app.run()
            app.button(key="refresh_market").click().run()
            self.assertFalse(app.exception)
            engine.assert_not_called()
            self.assertTrue(provider.call_args.args[-1])
            self.assertIsNone(app.session_state['analysis'])
            self.assertIsNotNone(app.session_state['market_preview'])

    def test_run_invokes_engine_once_and_stores_output(self):
        app = self.app()
        with patch('ui.services.market_data', return_value=self.data), patch('ui.services.run_analysis', return_value=copy.deepcopy(self.bundle)) as engine:
            app.run()
            app.button(key="run_analysis").click().run()
            self.assertFalse(app.exception)
            self.assertIsNotNone(app.session_state['last_success'])
            app.radio(key='page').set_value('Risk').run()
            self.assertEqual(engine.call_count, 1)

    def test_run_button_with_real_engine_and_navigation(self):
        app = self.app()
        frame = pd.DataFrame([{'Ticker': t, 'Current %': w * 100, 'Strategic %': w * 100,
            'Low %': (w - .05) * 100, 'High %': (w + .05) * 100, 'Minimum %': 0.,
            'Maximum %': 100., 'Fixed %': None, 'Role': 'core', 'Group': t} for t, w in self.holdings.items()])
        app.session_state['editor_base'] = frame
        app.session_state['as_of'] = self.prices.index[-1].date()
        app.session_state['repetitions'] = 3
        with patch('ui.services.market_data', return_value=self.data), patch('ui.services.build_portfolio_recommendation', wraps=build_portfolio_recommendation) as engine:
            app.run()
            app.button(key="run_analysis").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            saved = copy.deepcopy(app.session_state['analysis']['recommendation'])
            self.assertAlmostEqual(sum(saved['recommended_portfolio']['central_weights'].values()), 1.)
            app.radio(key='page').set_value('Robustness').run()
            self.assertFalse(app.exception)
            self.assertEqual(engine.call_count, 1)
            self.assertEqual(app.session_state['analysis']['recommendation'], saved)

    def test_real_engine_accepts_main_targets_above_old_satellite_limits(self):
        app = self.app()
        frame = initial_editor().set_index('Ticker').loc[['QQQ', 'IEF']].reset_index()
        frame['Current %'] = [60., 40.]
        frame['Strategic %'] = [60., 40.]
        app.session_state['editor_base'] = frame
        app.session_state['as_of'] = self.prices.index[-1].date()
        app.session_state['repetitions'] = 3
        with patch('ui.services.market_data', return_value=self.data), patch('ui.services.build_portfolio_recommendation', wraps=build_portfolio_recommendation) as engine:
            app.run()
            app.button(key='run_analysis').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(engine.call_count, 1)
            saved = app.session_state['analysis']['recommendation']
            self.assertEqual(saved['strategic_portfolio']['weights'], {'QQQ': .6, 'IEF': .4})
            self.assertAlmostEqual(sum(saved['recommended_portfolio']['central_weights'].values()), 1.)
            self.assertTrue(any('Satellite limit increased' in warning for warning in app.session_state['analysis']['warnings']))

    def test_sidebar_settings_persist_across_navigation(self):
        app = self.app().run()
        with patch('ui.services.run_analysis') as engine, patch('ui.services.market_data') as provider:
            app.number_input(key='threshold').set_value(2.).run()
            app.radio(key='page').set_value('Risk').run()
            self.assertEqual(app.session_state['threshold'], 2.)
            engine.assert_not_called()
            provider.assert_not_called()

    def test_noncritical_research_failure_retains_recommendation(self):
        with patch('ui.services.research_outputs', side_effect=ValueError('incomplete history')):
            bundle = services.run_analysis(self.data, self.holdings, self.policy, self.options, .03, self.as_of)
        self.assertEqual(bundle['recommendation'], self.bundle['recommendation'])
        self.assertTrue(bundle['outcomes'].empty)
        self.assertTrue(bundle['warnings'])


if __name__ == '__main__':
    unittest.main()
