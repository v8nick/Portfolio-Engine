"""Reusable presentation components; financial calculations remain in services."""
from datetime import date
import streamlit as st
from config import live, shared
from ui.helpers import initial_editor, complete_editor_rows


def portfolio_import(apply_csv_import):
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


def portfolio_editor(switch_editor_view):
    st.subheader('Portfolio allocations')
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
    return edited, advanced
