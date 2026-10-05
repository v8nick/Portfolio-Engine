"""Seven read-only dashboard views of stored backend outputs."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui.helpers import number, allocation_table, metric_table, action_table, macro_input_table

PAGES = ('Executive overview', 'Market intelligence', 'Macro indicators',
         'Portfolio construction', 'Risk', 'Robustness', 'Rebalance')
COLORS = ('#8B98A8', '#344E68', '#26847E', '#AB8963', '#7C7495', '#566878')
METHODS = {'strategic': 'Strategic', 'minimum_variance': 'Minimum variance',
           'bl_max_sharpe': 'BL max Sharpe', 'risk_parity': 'Risk parity', 'cvar': 'CVaR'}


def chart(fig):
    fig.update_layout(template='plotly_white', font=dict(family='Arial', size=12, color='#344054'),
                      colorway=COLORS, margin=dict(l=20, r=20, t=35, b=20),
                      legend=dict(orientation='h', y=-.18), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, width='stretch')


def table(frame, percentages=()):
    if frame.empty:
        st.info('No results available for this table.')
        return
    shown = frame.copy(deep=True)
    for column in percentages:
        if column in shown:
            shown[column] = shown[column].map(lambda v: number(v, 'percent'))
    for column in shown.select_dtypes(include='number'):
        if column not in ('Sample count', 'Horizon'):
            shown[column] = shown[column].map(number)
    st.dataframe(shown, hide_index=True, width='stretch')


def score_cards(regime):
    for col, key, label in zip(st.columns(4), ('growth', 'inflation', 'financial_conditions', 'stress'),
                             ('Growth proxy score', 'Inflation / rate-pressure proxy score', 'Tightening proxy score', 'Stress proxy score')):
        col.metric(label, number(regime.get('scores', {}).get(key)))
    st.caption('Smoothed standardized market-proxy scores, not GDP growth, CPI inflation, or an economic diagnosis. '
               'Positive means above the proxy’s trailing historical average; negative means below it.')


def macro_inputs(regime, groups=None):
    st.dataframe(macro_input_table(regime, groups), hide_index=True, width='stretch')
    st.caption('Saved inputs used by this analysis. Returns and return differences are percentages; Treasury yield changes are basis points. '
               'N/A means missing or stale input; it is not treated as zero. Relative performance is a difference in returns, not a price ratio. '
               'Rising yields and gold can have several causes; these inputs do not establish measured inflation.')


def allocation_chart(result, core_only=False):
    frame = allocation_table(result)
    if core_only:
        core = [t for t, item in result.get('policy', {}).items() if item.get('role') == 'core']
        frame = frame[frame['Asset'].isin(core)]
    if frame.empty:
        st.info('No allocation data available.')
        return
    fig = go.Figure()
    for name, color in zip(('Current', 'Strategic', 'Recommended'), COLORS):
        interval = dict(type='data', symmetric=False, array=frame['High'] - frame['Recommended'],
                        arrayminus=frame['Recommended'] - frame['Low']) if name == 'Recommended' else None
        fig.add_bar(name=name, x=frame['Asset'], y=frame[name], marker_color=color, error_y=interval)
    chart(fig.update_layout(barmode='group', yaxis_tickformat='.0%', height=390))


def metrics(result):
    table(metric_table(result), ('Expected return', 'Volatility', 'Historical max drawdown',
                               'Downside deviation', 'Daily VaR', 'Daily CVaR'))
    confidence = result.get('recommended_portfolio', {}).get('historical_tail_risk', {}).get('confidence')
    st.caption(f'Expected return / volatility are annual model estimates. VaR and CVaR are daily empirical losses at {number(confidence, "percent")} confidence. '
               'Historical drawdown uses constant daily portfolio weights; it is not a forecast.')


def overview(bundle):
    result, regime = bundle['recommendation'], bundle['recommendation'].get('regime', {})
    st.subheader('Market indicators')
    macro_inputs(regime, ('Growth', 'Inflation / rate pressure'))
    with st.expander('Composite market-proxy scores'):
        score_cards(regime)
    st.caption('See Macro indicators for financial conditions, stress inputs and score construction. '
               'The portfolio engine still uses historical market-pattern weights to adjust return estimates; these are heuristic inputs, not economic forecasts.')
    st.subheader('Core allocation')
    allocation_chart(result, core_only=True)
    st.caption('Bars show central allocations. Intervals show the backend recommended ranges, not confidence intervals.')
    st.subheader('Portfolio comparison')
    metrics(result)
    confidence = result.get('recommended_portfolio', {}).get('confidence', {})
    st.subheader('Action summary')
    st.write(f"**{result.get('rebalance', {}).get('portfolio_action', 'N/A')}** · Recommendation confidence: "
             f"{confidence.get('label', 'unavailable')} ({number(confidence.get('overall_score'), 'percent')})")
    table(action_table(result).drop(columns=['Estimated cost'], errors='ignore'),
          ('Current', 'Target', 'Low', 'High', 'Difference'))


def market(bundle, preview=None):
    snapshot = preview or (bundle or {}).get('snapshot')
    if not snapshot:
        st.info('Refresh Market Data or run analysis to load market intelligence.')
        return
    st.caption(f"Market snapshot as of {snapshot.get('timestamp', 'N/A')}. "
               + ('Refreshed independently; portfolio recommendation retains its original analysis date.' if preview else 'Stored with this portfolio analysis.'))
    horizon = st.selectbox('Market change horizon', ['1D', '5D', '21D', '63D'], key='market_horizon')
    rates, curve = snapshot.get('rates', {}), snapshot.get('curve', {})
    cols = st.columns(4)
    for col, tenor in zip(cols, ('2Y', '10Y')):
        metric = rates.get(tenor, {})
        col.metric(f'{tenor} Treasury', number(metric.get('value'), 'yield'),
                   number(metric.get('changes', {}).get(horizon), 'bp'), delta_color='off')
    cols[2].metric('2s10s spread', number(curve.get('value'), 'bp'))
    cols[3].metric('Daily curve movement', curve.get('daily_classification', 'unavailable').replace('_', ' ').title())
    rows = []
    for group in ('rates', 'equities', 'fixed_income', 'credit', 'fx', 'volatility', 'other'):
        for asset, item in snapshot.get(group, {}).items():
            if not isinstance(item, dict):
                continue
            change = item.get('changes', {}).get(horizon)
            rows.append({'Series': asset, 'Value': number(item.get('value'), 'yield' if group == 'rates' else 'number'),
                         f'Change {horizon}': number(change, 'bp' if group == 'rates' else 'percent', signed=True),
                         'As of': item.get('as_of'), 'Status': 'Stale' if item.get('stale') else 'Missing' if item.get('value') is None else 'Available'})
    st.subheader('Market observations')
    table(pd.DataFrame(rows))
    st.caption('Treasury changes are basis points; price changes use adjusted prices. Each series has its own observation date.')
    st.subheader('Relative performance')
    relative = pd.DataFrame([{'Pair': k.replace('_', ' / '), 'Return difference': v.get('returns', {}).get(horizon),
                             'As of': v.get('as_of'), 'Stale': v.get('stale')}
                            for k, v in snapshot.get('relative_performance', {}).items()])
    table(relative, ('Return difference',))
    if not relative.empty:
        chart(go.Figure(go.Bar(x=relative['Pair'], y=relative['Return difference'], marker_color=COLORS[1]))
              .update_layout(yaxis_tickformat='.1%', height=300, title=f'Relative return differences · {horizon}'))
    st.subheader('Deterministic market interpretation')
    st.caption('Signals and hypotheses below are supplied by the backend for its 1D horizon.')
    for col, key, title in zip(st.columns(3), ('observations', 'signals', 'interpretation'),
                               ('Observed facts', 'Derived signals', 'Interpretation / hypotheses')):
        with col:
            st.markdown(f'**{title}**')
            entries = snapshot.get(key, [])
            if not entries:
                st.info('Unavailable for the current data.')
            if key == 'observations':
                st.dataframe(pd.DataFrame(entries).drop(columns=['type'], errors='ignore'), hide_index=True)
            else:
                for item in entries:
                    st.write(item.get('text', ''))
    for message in snapshot.get('data_quality', []):
        st.warning(message)
    with st.expander('Data sources'):
        st.json(snapshot.get('sources', {}))


def regimes(bundle):
    regime = bundle['recommendation'].get('regime', {})
    st.subheader('Underlying market metrics')
    macro_inputs(regime)
    st.subheader('How the inputs combine')
    score_cards(regime)
    labels = {'growth': 'Growth proxy', 'inflation': 'Inflation / rate-pressure proxy',
              'financial_conditions': 'Tightening proxy', 'stress': 'Stress proxy'}
    table(pd.DataFrame([{'Composite': label, 'Before smoothing': regime.get('raw_scores', {}).get(key),
                        'After smoothing': regime.get('scores', {}).get(key),
                        'Available input weight': regime.get('coverage', {}).get(key)}
                       for key, label in labels.items()]), ('Available input weight',))
    st.caption('Available input weight measures data coverage, not confidence. Missing families are excluded and available weights renormalized; '
               'at least two families are required for each score.')
    st.info('The default model compares 21- and 63-session changes with each input’s trailing three-year history '
            '(at least one year required). Standardized values are capped at ±3, horizons are averaged within each input family, '
            'then families are combined with equal weights except gold at half weight. Scores use exponential smoothing with a 10-session span. '
            'The backend retains heuristic market-pattern weights for portfolio calculations; no current economic regime or probability is asserted here.')
    history = bundle.get('history', pd.DataFrame())
    if not history.empty:
        with st.expander('Historical market-proxy scores'):
            columns = [f'smoothed_{key}_score' for key in labels if f'smoothed_{key}_score' in history]
            st.line_chart(history[columns].rename(columns={f'smoothed_{k}_score': v for k, v in labels.items()}))
    st.subheader('Historical returns by market pattern')
    horizon = st.selectbox('Forward horizon', ['1 month', '3 months', '6 months'], key='conditional_horizon')
    patterns = {'Growth above / pressure below average': 'goldilocks',
                'Growth above / pressure above average': 'reflation',
                'Growth below / pressure below average': 'slowdown',
                'Growth below / pressure above average': 'stagflation'}
    selected = patterns[st.selectbox('Historical market pattern', list(patterns), key='conditional_pattern')]
    st.info('Historical conditional outcomes are descriptive and are not forecasts.')
    st.caption('Cumulative simple returns over 21 / 63 / 126 trading observations. Completed, globally non-overlapping samples; not annualized.')
    frame = bundle.get('outcomes', pd.DataFrame())
    if frame.empty:
        st.warning('Historical conditional outcomes are unavailable.')
        return
    frame = frame[(frame['horizon'] == {'1 month': 21, '3 months': 63, '6 months': 126}[horizon]) & (frame['regime'] == selected)]
    frame = frame.rename(columns={'asset': 'Asset', 'n_observations': 'Sample count', 'mean': 'Mean',
        'median': 'Median', 'std': 'Std', 'positive_frequency': 'Positive frequency', 'q25': '25th percentile', 'q75': '75th percentile'})
    table(frame.drop(columns=['regime', 'horizon']), ('Mean', 'Median', 'Std', 'Positive frequency', '25th percentile', '75th percentile'))
    chart(go.Figure(go.Bar(x=frame['Asset'], y=frame['Mean'], marker_color=COLORS[1]))
          .update_layout(yaxis_tickformat='.1%', height=300, title='Historical conditional mean'))


def portfolio(bundle):
    result = bundle['recommendation']
    consensus = result.get('optimizer_consensus', {})
    confidence = result.get('recommended_portfolio', {}).get('confidence', {})
    cols = st.columns(2)
    cols[0].metric('Optimizer agreement', number(consensus.get('agreement_score'), 'percent'))
    cols[1].metric('Recommendation confidence', number(confidence.get('overall_score'), 'percent'))
    st.caption('Both are heuristic evidence scores, not statistical certainty.')
    candidates = result.get('candidate_allocations', {})
    assets = list(result.get('policy', {}))
    matrix, names, rows = [], [], []
    for method, label in METHODS.items():
        item = candidates.get(method, {})
        if item.get('status') == 'unavailable':
            st.warning(f'{label} unavailable; excluded from optimizer consensus.')
        elif not item:
            st.info(f'{label} was not enabled in this analysis.')
        names.append(label)
        matrix.append([(item.get('weights') or {}).get(t) for t in assets])
    names.append('Recommended')
    matrix.append([result.get('recommended_portfolio', {}).get('central_weights', {}).get(t) for t in assets])
    chart(go.Figure(go.Heatmap(z=matrix, x=assets, y=names, colorscale=[[0, '#F3F5F7'], [1, '#344E68']],
                             colorbar=dict(tickformat='.0%'), hovertemplate='%{y} · %{x}: %{z:.1%}<extra></extra>'))
          .update_layout(height=360))
    allocation = allocation_table(result).set_index('Asset')
    for asset in assets:
        item = consensus.get('per_asset', {}).get(asset, {})
        row = {'Asset': asset, **{label: (candidates.get(method, {}).get('weights') or {}).get(asset) for method, label in METHODS.items()},
               'Optimizer mean': item.get('mean'), 'Optimizer median': item.get('median'), 'Dispersion': item.get('standard_deviation')}
        row.update({k: allocation.loc[asset, k] for k in ('Recommended', 'Low', 'High')})
        rows.append(row)
    table(pd.DataFrame(rows), (*METHODS.values(), 'Optimizer mean', 'Optimizer median', 'Dispersion', 'Recommended', 'Low', 'High'))
    with st.expander('Expected returns and Black-Litterman views'):
        st.caption('Annual arithmetic model estimates; existing configured manual views are retained by the backend.')
        expected = result.get('expected_returns', {})
        frame = pd.DataFrame({label: expected.get(key, {}) for label, key in [('Strategic prior', 'prior'),
            ('Regime weighted', 'regime_weighted'), ('Regime delta', 'regime_delta'), ('BL posterior', 'black_litterman_posterior')]})
        table(frame.rename_axis('Asset').reset_index(), frame.columns)
        st.json({key: expected.get(key) for key in ('regime_bl_views', 'view_sources', 'manual_relative_views', 'excluded_manual_views')})
    with st.expander('Confidence components and policy constraints'):
        st.json(confidence)
        st.json(result.get('constraints', {}))


def risk(bundle):
    result = bundle['recommendation']
    metrics(result)
    selection = st.selectbox('Risk contribution portfolio', ['Current', 'Strategic', 'Recommended'], key='risk_portfolio')
    p = result.get({'Current': 'current_portfolio', 'Strategic': 'strategic_portfolio', 'Recommended': 'recommended_portfolio'}[selection], {})
    rows = [{'Asset': t, 'Weight': w, 'Risk contribution': p.get('risk_contributions', {}).get(t, {}).get('percentage')}
            for t, w in p.get('weights', {}).items()]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        fig = go.Figure()
        for label, color in zip(('Weight', 'Risk contribution'), COLORS):
            fig.add_bar(name=label, x=frame['Asset'], y=frame[label], marker_color=color)
        chart(fig.update_layout(barmode='group', yaxis_tickformat='.0%', height=360))
        table(frame, ('Weight', 'Risk contribution'))
    st.subheader('Configured asset-group exposure')
    groups = pd.DataFrame([{'Group': key, 'Weight': item.get('weight'), 'Risk contribution': item.get('percentage_risk_contribution')}
                           for key, item in p.get('groups', {}).items()])
    table(groups, ('Weight', 'Risk contribution'))
    st.caption('Groups come from portfolio policy; no classifications are inferred by the UI.')
    table(pd.DataFrame([{'Role': role, 'Weight': w} for role, w in p.get('role_weights', {}).items()]), ('Weight',))


def robustness(bundle):
    result = bundle['recommendation']
    stability = result.get('stability', {})
    st.subheader('BL max-Sharpe weight stability')
    st.caption(stability.get('method', 'Method unavailable'))
    cols = st.columns(3)
    cols[0].metric('Overall weight stability', number(stability.get('overall_score'), 'percent'))
    cols[1].metric('Successful resamples', str(stability.get('successful_repetitions', 0)))
    cols[2].metric('Failed resamples', str(stability.get('failed_repetitions', 0)))
    if not stability.get('successful_repetitions'):
        st.warning('No successful bootstrap runs. Intervals are unavailable; the backend discounts recommendation confidence.')
    rows = [{'Asset': t, 'Recommended central': result.get('recommended_portfolio', {}).get('central_weights', {}).get(t),
             'Bootstrap median': v.get('median'), '25th': v.get('q25'), '75th': v.get('q75'),
             '10th': v.get('q10'), '90th': v.get('q90'), 'Weight std': v.get('standard_deviation'),
             'Stability score': v.get('weight_stability_score')} for t, v in stability.get('per_asset', {}).items()]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        available = frame.dropna(subset=['Bootstrap median', '25th', '75th'])
        if not available.empty:
            fig = go.Figure(go.Scatter(x=available['Bootstrap median'], y=available['Asset'], mode='markers', name='Bootstrap median / IQR',
                marker=dict(color=COLORS[1]), error_x=dict(type='data', symmetric=False,
                array=available['75th'] - available['Bootstrap median'], arrayminus=available['Bootstrap median'] - available['25th'])))
            fig.add_scatter(x=available['Recommended central'], y=available['Asset'], mode='markers',
                            name='Recommended central', marker=dict(color=COLORS[2], symbol='diamond'))
            chart(fig.update_layout(xaxis_tickformat='.0%', height=max(320, 26 * len(frame))))
        table(frame, frame.columns.drop('Asset'))
    st.info('Only BL max-Sharpe is resampled. These intervals describe estimation sensitivity; other portfolio methods were not resampled.')
    for message in stability.get('failure_examples', []):
        st.warning(message)


def rebalance(bundle):
    result, value = bundle['recommendation'], bundle.get('portfolio_value', 0.)
    reb = result.get('rebalance', {})
    st.subheader(f"Portfolio decision: {reb.get('portfolio_action', 'N/A')}")
    cols = st.columns(3)
    cols[0].metric('Two-sided turnover', number(reb.get('estimated_turnover'), 'percent'))
    for col, key, label in zip(cols[1:], ('estimated_transaction_cost', 'estimated_tax_drag'), ('Transaction cost', 'Approximate tax drag')):
        rate = reb.get(key)
        col.metric(label, number(rate, 'percent'))
        col.caption(number(rate * value if rate is not None else None, 'money'))
    table(action_table(result), ('Current', 'Target', 'Low', 'High', 'Difference'))
    st.info('Tax estimates are approximate and do not represent tax-lot-level optimization.')
    st.caption(reb.get('notes', ''))
    st.caption('Per-asset costs are N/A because the backend exposes only aggregate costs. Dollar values scale those rates by the portfolio value saved at analysis time. Action labels are not executable orders.')
    with st.expander('Prospective full-rebalance estimates'):
        st.json(reb.get('prospective_full_rebalance', {}))


def render(page, bundle, preview=None):
    if page == 'Market intelligence':
        market(bundle, preview)
    elif bundle is None:
        st.info('Configure the portfolio in the sidebar and click Run Analysis.')
    else:
        {'Executive overview': overview, 'Macro indicators': regimes, 'Portfolio construction': portfolio,
         'Risk': risk, 'Robustness': robustness, 'Rebalance': rebalance}[page](bundle)
