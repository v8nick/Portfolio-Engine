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
from ui.helpers import initial_editor, validate_editor, number, import_portfolio_csv, complete_editor_rows, prepare_editor_policy
from ui.services import market_data, snapshot, run_analysis
from ui.views import PAGES, render

st.set_page_config(page_title='Portfolio Engine', page_icon='▦', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@media (min-width: 1000px) { [data-testid="stSidebar"] {min-width: 400px;} }
.block-container {padding-top: 2rem; max-width: 1550px;}
[data-testid="stMetric"] {border: 1px solid #DDE3E9; border-radius: 6px; padding: 14px;}
h1 {font-weight: 600;} h2,h3 {font-weight: 500;}
</style>''', unsafe_allow_html=True)

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


def switch_editor_view():
    state = st.session_state
    state.editor_base = state.get('draft_portfolio', state.editor_base).copy(deep=True)
    state.pop('policy_editor', None)


with st.sidebar:
    st.title('Portfolio Engine')
    st.caption('Portfolio policy · macro context · implementation review')
    page = st.radio('Workspace', PAGES, key='page')
    st.divider()
    st.subheader('Portfolio and policy')
    st.number_input('Portfolio value ($)', min_value=1., value=1_000_000., step=10_000., key='portfolio_value')
    with st.expander('Import portfolio CSV'):
        st.caption('Upload a Ticker or Symbol column, or one ticker per line without a header. Optional columns match the editor. '
                   'Weights are percentages: 25 or 25%, not 0.25. New tickers start at 0% and as satellites; review their roles and groups.')
        st.file_uploader('Portfolio CSV', type=['csv'], key='portfolio_csv')
        st.selectbox('Import mode', ['Add / update tickers', 'Replace table'], key='csv_mode')
        st.caption('Add / update keeps other rows and preserves columns omitted from the CSV. Replace table keeps only imported tickers. '
                   'No weights are automatically normalized.')
        st.button('Import CSV', key='import_csv', on_click=apply_csv_import,
                  disabled=st.session_state.portfolio_csv is None)
        template = initial_editor()
        template = template.loc[template['Strategic %'] > 0, ['Ticker', 'Current %', 'Strategic %']].copy()
        template['Current %'] = template['Strategic %']
        st.download_button('Download CSV template', template.rename(columns={'Strategic %': 'Target %'}).to_csv(index=False),
            file_name='portfolio_template.csv', mime='text/csv', key='csv_template')
        st.download_button('Download advanced CSV template', 'Ticker,Current %,Target %,Low %,High %,Minimum %,Maximum %,Fixed %,Role,Group\n'
            'SPY,60,60,50,70,0,100,,core,us_equity\nIEF,40,40,30,50,0,100,,core,treasury\n',
            file_name='portfolio_advanced_template.csv', mime='text/csv', key='csv_advanced_template')
        if 'csv_message' in st.session_state:
            kind, message = st.session_state.csv_message
            getattr(st, kind)(message)
    advanced = st.checkbox('Show advanced allocation settings', key='advanced_allocations', on_change=switch_editor_view)
    st.caption('Current % is the share you own today. Target % is the share you want in your long-term portfolio '
               '(20 means 20% of the portfolio). Each column must total 100%.')
    numeric = {name: st.column_config.NumberColumn(name, min_value=0., max_value=100., format='%.2f')
               for name in ('Current %', 'Strategic %', 'Low %', 'High %', 'Minimum %', 'Maximum %', 'Fixed %')}
    numeric['Strategic %'] = st.column_config.NumberColumn('Target %', min_value=0., max_value=100., format='%.2f',
        help='The percentage you want this holding to represent in your long-term portfolio. This is the input previously called Strategic %.')
    numeric['Role'] = st.column_config.SelectboxColumn('Role', options=['core', 'satellite'], required=True)
    if not advanced:
        numeric.update({name: None for name in ('Low %', 'High %', 'Minimum %', 'Maximum %', 'Fixed %', 'Role', 'Group')})
    else:
        st.caption('Low / High: tactical allocation band. Minimum / Maximum: hard limits. Fixed: optional locked target. '
                   'Role: core or satellite. Group: category used for risk summaries. Optional rules are used in this view, '
                   'but Target % takes priority if a band, limit or fixed weight conflicts.')
    edited = st.data_editor(st.session_state.editor_base, key='policy_editor', num_rows='dynamic',
        hide_index=True, width='stretch', column_config=numeric, height=360)
    edited = complete_editor_rows(edited)
    st.session_state['draft_portfolio'] = edited.copy(deep=True)
    st.caption(f"Current total: {edited['Current %'].sum():.2f}% · Target total: {edited['Strategic %'].sum():.2f}%")
    if not advanced:
        st.caption('Targets take priority. Bands are generated at target ±5 percentage points (within 0–100%); '
                   'hidden hard limits and fixed weights are ignored. Classifications are supplied automatically when omitted.')
    with st.expander('Optional concentration limits'):
        st.caption('These limits expand automatically if needed to accommodate your entered targets.')
        st.number_input('Max satellite allocation (%)', min_value=0., max_value=100., value=live.MAX_SATELLITE_ALLOCATION * 100, key='satellite_cap')
        st.number_input('Max individual satellite (%)', min_value=0., max_value=100., value=live.MAX_INDIVIDUAL_SATELLITE_WEIGHT * 100, key='individual_cap')
    with st.expander('Analysis settings', expanded=False):
        defaults = shared.DECISION_SETTINGS
        st.number_input('Rebalance threshold (%)', min_value=0., max_value=100., value=defaults['rebalance_threshold'] * 100, key='threshold')
        st.slider('Historical market-pattern overlay strength', 0., 1., defaults['regime_bl_strength'], .05, key='overlay_strength')
        st.number_input('Bootstrap repetitions', min_value=0, max_value=300, value=defaults['robustness_repetitions'], step=10, key='repetitions')
        st.slider('VaR / CVaR confidence', .80, .99, defaults['tail_confidence'], .01, key='tail_confidence')
        st.checkbox('Override risk-free rate', key='rf_override')
        st.number_input('Annual risk-free rate (%)', min_value=-10., max_value=100., value=3., key='rf_percent')
        st.date_input('History start', value=date(2007, 1, 1), key='history_start')
        st.date_input('Analysis as of', value=date.today(), key='as_of')
    run = st.button('Run Analysis', type='primary', width='stretch', key='run_analysis')
    refresh = st.button('Refresh Market Data', width='stretch', key='refresh_market')
    st.caption('Default strategy is illustrative. Engine settings, manual views and tax assumptions remain those in configuration. No trades are executed.')

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

st.title(page)
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
    st.caption(f"Analysis completed: {bundle['completed_at']} · Portfolio observations through: {result.get('as_of', 'N/A')} · "
               f"Market snapshot: {bundle['snapshot'].get('timestamp', 'N/A')} · Portfolio value: {number(bundle['portfolio_value'], 'money')}")
    if bundle['config'] != config:
        st.info('Configuration has changed. These are the last successful results; click Run Analysis to apply your edits.')
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
render(page, bundle, state.market_preview)
