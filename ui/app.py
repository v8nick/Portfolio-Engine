"""Launch from repository root: streamlit run ui/app.py."""
from __future__ import annotations

# Streamlit executes a file as a script, so make the repository importable without
# environment configuration. The console dashboard.py remains unshadowed.
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy
from datetime import datetime, timezone, date
import json
from uuid import uuid4
import streamlit as st
from config import live, shared
from ui.helpers import initial_editor, validate_editor, number, import_portfolio_csv, prepare_editor_policy
from ui.services import market_data, snapshot, run_analysis
from ui.views import PAGES, render
from ui.components import portfolio_editor

st.set_page_config(page_title='Portfolio Engine', page_icon='📈', layout='wide', initial_sidebar_state='collapsed')
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1550px;}
[data-testid="stMetric"] {border: 1px solid color-mix(in srgb, currentColor 18%, transparent); border-radius: 8px; padding: 14px;}
h1 {font-weight: 600;} h2,h3 {font-weight: 500;}
.st-key-page [role="radiogroup"] {gap: .4rem .6rem; flex-wrap: wrap;}
.st-key-page label {padding: .35rem .65rem; border-radius: 6px; border: 1px solid transparent;}
.st-key-page label:has(input:checked) {background: color-mix(in srgb, #26847E 14%, transparent); border-color: #26847E;}
@media (max-width: 640px) {.block-container {padding-left: 1rem; padding-right: 1rem;}}
</style>""", unsafe_allow_html=True)

INPUT_DEFAULTS = dict(portfolio_value=1_000_000., advanced_allocations=False,
    satellite_cap=live.MAX_SATELLITE_ALLOCATION * 100, individual_cap=live.MAX_INDIVIDUAL_SATELLITE_WEIGHT * 100,
    threshold=shared.DECISION_SETTINGS['rebalance_threshold'] * 100,
    overlay_strength=shared.DECISION_SETTINGS['regime_bl_strength'],
    repetitions=shared.DECISION_SETTINGS['robustness_repetitions'], tail_confidence=shared.DECISION_SETTINGS['tail_confidence'],
    rf_override=False, rf_percent=3., history_start=date(2007, 1, 1), as_of=date.today(), csv_mode='Add / update tickers')
for key, value in INPUT_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

for key, value in {'editor_base': initial_editor(), 'analysis': None, 'market_preview': None,
                   'last_success': None, 'last_refresh': None, 'refresh_token': 0,
                   'refresh_warnings': [], 'analysis_error': None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Preserve navigation in sessions opened before the metrics page was renamed.
if st.session_state.get('page') == 'Regime analysis':
    st.session_state['page'] = 'Macro indicators'


def apply_csv_import():
    state = st.session_state
    try:
        frame = import_portfolio_csv(state.portfolio_csv.getvalue(),
            state.get('draft_portfolio', state.editor_base), replace=state.csv_mode == 'Replace table')
    except ValueError as exc:
        state.csv_message = ('error', str(exc))
        return
    state.editor_base = frame
    state.pop('policy_editor', None)
    state.csv_message = ('success', 'CSV imported. Review weights, roles and policy limits, then click Run Analysis.')


def save_editor_navigation():
    # Commit table edits before Streamlit removes an unrendered editor widget.
    switch_editor_view()


def switch_editor_view():
    state = st.session_state
    state.editor_base = state.get('draft_portfolio', state.editor_base).copy(deep=True)
    state.pop('policy_editor', None)


# Keep widget values alive when their page is not rendered. Buttons/uploads are
# intentionally excluded; the imported portfolio, rather than its upload, persists.
for key in list(st.session_state):
    if key in INPUT_DEFAULTS or (key.startswith(('bt_', 'sim_')) and key != 'bt_available_history'):
        st.session_state[key] = st.session_state[key]

st.title('Portfolio Engine')
st.caption('Quantitative portfolio optimization, backtesting, risk analysis, and market simulation.')
page = st.radio('Workspace', PAGES, key='page', horizontal=True,
                format_func=lambda x: x.title(), label_visibility='collapsed', on_change=save_editor_navigation)
st.divider()
run = refresh = False
if page == 'Home':
    st.write('Build your portfolio, explore market conditions, evaluate allocation strategies, and simulate potential outcomes. '
             'Start by entering your current investments and target allocations below, then use the navigation menu to explore the analysis.')
    for col, title, detail in zip(st.columns(3),
            ('1. Build your portfolio', '2. Run the analysis', '3. Explore results'),
            ('Enter assets, current holdings, targets and portfolio value.',
             'Evaluate market conditions, allocation, risk and robustness.',
             'Compare strategies, backtest performance and simulate outcomes.')):
        with col:
            st.markdown(f'**{title}**')
            st.caption(detail)
    actions = st.columns([2, 1])
    run = actions[0].button('Apply Portfolio & Run Analysis', type='primary', width='stretch', key='run_analysis')
    refresh = actions[1].button('Refresh Market Data', width='stretch', key='refresh_market')
    st.caption('Apply uses the portfolio below. Refresh updates market data; run analysis to incorporate it into your results.')
    with st.expander('Portfolio editor', expanded=True):
        edited, advanced = portfolio_editor(apply_csv_import, switch_editor_view)
else:
    edited = st.session_state.get('draft_portfolio', st.session_state.editor_base)
    advanced = st.session_state.advanced_allocations


state = st.session_state
effective_editor, satellite_cap, individual_cap, policy_notes = prepare_editor_policy(
    edited, advanced, state.satellite_cap / 100, state.individual_cap / 100)
holdings, policy, errors = validate_editor(effective_editor, satellite_cap, individual_cap)
if state.history_start >= state.as_of:
    errors.append('History start must precede the analysis date.')
options = {'rebalance_threshold': state.threshold / 100, 'regime_bl_strength': state.overlay_strength,
           'robustness_repetitions': int(state.repetitions), 'tail_confidence': state.tail_confidence,
           'max_satellite_allocation': satellite_cap, 'max_individual_satellite_weight': individual_cap}
config = {'holdings': holdings, 'policy': policy, 'options': options, 'portfolio_value': state.portfolio_value,
          'risk_free': state.rf_percent / 100 if state.rf_override else None,
          'start': state.history_start.isoformat(), 'as_of': state.as_of.isoformat()}

if run or refresh:
    state.analysis_error = None
    if errors:
        state.analysis_error = ('Please correct the portfolio configuration.', '\n'.join(errors))
    else:
        try:
            with st.spinner('Refreshing market data…' if refresh else 'Running portfolio analysis and robustness checks…'):
                if refresh:
                    state.refresh_token = uuid4().hex
                data = market_data(config['start'], tuple(sorted(policy)), state.refresh_token)
                if refresh:
                    state.market_preview = snapshot(data, config['as_of'])
                    state.refresh_warnings = list(data.get('data_quality', []))
                    state.last_refresh = datetime.now(timezone.utc).isoformat(timespec='seconds')
                    st.success('Market refresh finished. Run Analysis to update the portfolio recommendation.')
                else:
                    bundle = run_analysis(data, holdings, policy, options, config['risk_free'], config['as_of'])
                    bundle['portfolio_value'] = config['portfolio_value']
                    bundle['config'] = copy.deepcopy(config)
                    bundle['completed_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
                    bundle['warnings'].extend(policy_notes)
                    # Commit state only on success; failed runs retain the previous analysis.
                    state.analysis = bundle
                    state.last_success = bundle['completed_at']
                    state.market_preview = None
                    st.success('Analysis saved. All pages now use this result.')
        except Exception as exc:
            if isinstance(exc, ValueError):
                message = str(exc)
            else:
                message = 'Analysis or data loading failed. Check provider availability and portfolio history, then try again.'
            state.analysis_error = (message, f'{type(exc).__name__}: {exc}')

st.subheader('Executive summary' if page == 'Home' else page.title())
DESCRIPTIONS = {
    'Executive overview': 'Your saved portfolio recommendation and the evidence behind it.',
    'Market intelligence': 'Market trends, asset performance and financial conditions.',
    'Macro indicators': 'The observed metrics behind the market-pattern model.',
    'Portfolio construction': 'Compare allocation methods and your current and target weights.',
    'Risk': 'Explore diversification, correlations and portfolio risk contributions.',
    'Robustness': 'Assess allocation stability and the evidence supporting each recommendation.',
    'Rebalance': 'Review proposed changes, trading costs and approximate tax effects.',
    'Backtest': 'Compare historical strategy performance using point-in-time decisions.',
    'Simulation': 'Explore possible portfolio outcomes under explicit model assumptions.'}
if page in DESCRIPTIONS:
    st.caption(DESCRIPTIONS[page])
if policy_notes:
    with st.expander('Allocation rules adjusted to your targets', expanded=False):
        for note in policy_notes:
            st.info(note)
if state.analysis_error:
    st.error(state.analysis_error[0])
    with st.expander('Details'):
        st.text(state.analysis_error[1])
if errors:
    with st.expander(f'Configuration needs attention ({len(errors)})', expanded=True):
        for error in errors:
            st.warning(error)
bundle = state.analysis
if bundle:
    result = bundle['recommendation']
    st.caption(f"Assets: {len(result.get('policy', {}))} · Analysis completed: {bundle['completed_at']} · Portfolio observations through: {result.get('as_of', 'N/A')} · "
               f"Market snapshot: {bundle['snapshot'].get('timestamp', 'N/A')} · Portfolio value: {number(bundle['portfolio_value'], 'money')}")
    if bundle['config'] != config:
        st.info('Configuration has changed. These are the last successful results; return to Home and click Apply Portfolio & Run Analysis to apply your edits.')
    if state.market_preview:
        st.info('Market data was refreshed after this analysis. Portfolio results remain saved; run analysis to incorporate it.')
    warnings = list(dict.fromkeys(result.get('warnings', []) + bundle.get('warnings', []) + bundle['snapshot'].get('data_quality', [])))
    if warnings:
        with st.expander(f'Warnings and data quality ({len(warnings)})', expanded=False):
            for warning in warnings:
                st.warning(warning)
    if result.get('recommended_portfolio', {}).get('confidence', {}).get('label') == 'low':
        st.warning('Recommendation confidence is low. Review evidence, policy and implementation limits.')
    st.download_button('Download recommendation JSON', json.dumps(result, indent=2, allow_nan=False),
                       file_name='portfolio_recommendation.json', mime='application/json', key='download')
elif page != 'Market intelligence':
    st.caption('No portfolio analysis has been run. Inputs persist while you navigate this session.')
if state.last_refresh:
    st.caption(f'Last market refresh attempt: {state.last_refresh}')
    for warning in state.refresh_warnings:
        st.warning(warning)
render('Executive overview' if page == 'Home' else page, bundle, state.market_preview)
