"""Read-only custom-week Meridian contribution explorer."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1] / 'model_results' / 'meridian_truth_moderate'

@st.cache_data(show_spinner=False)
def _read_weekly(outcome):
    folder = BASE / outcome
    fit = pd.read_csv(folder / 'weekly_model_fit.csv', parse_dates=['week_start_date'])
    media = pd.read_csv(folder / 'weekly_media_contribution.csv', parse_dates=['week_start_date'])
    with np.load(folder / 'weekly_posterior_draws.npz', allow_pickle=True) as z:
        draws = z['incremental'].copy()
        dates = pd.to_datetime(z['dates'])
        channels = z['channels'].astype(str)
    if len(fit) != len(dates) or not np.array_equal(fit.week_start_date.to_numpy(), dates.to_numpy()):
        raise ValueError(f'{outcome}: weekly posterior dates do not match fitted sales')
    if draws.shape[1:] != (len(dates), len(channels)):
        raise ValueError(f'{outcome}: invalid weekly posterior shape')
    if set(media.subchannel) != set(channels):
        raise ValueError(f'{outcome}: subchannels do not match')
    return fit, media, draws, channels


def render_weekly_contributions():
    st.markdown('#### Weekly MMM Sales & Contribution')
    st.caption('Select exact modeled weeks. Aggregates fitted posterior contributions, not full-history ROI allocated to weeks. Truth-informed synthetic calibration.')
    try:
        stores = _read_weekly('stores')
        ecom = _read_weekly('ecommerce')
    except FileNotFoundError:
        st.info('Weekly Meridian exports are not published yet. Run the updated GitHub Actions training workflow and upload both outcome folders.')
        return
    except Exception as exc:
        st.error(f'Could not load weekly MMM results: {exc}')
        return
    if not stores[0].week_start_date.equals(ecom[0].week_start_date):
        st.error('Store and E-commerce week dates do not match.')
        return
    dates = stores[0].week_start_date
    labels = [d.strftime('%b %d, %Y') for d in dates]
    opts = list(range(len(dates)))
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        view = st.selectbox('Business', ['Total Business', 'Stores', 'E-commerce'], key='mmm_weekly_business')
    with col2:
        first = st.selectbox('Start week', opts, index=max(0, len(dates)-52), format_func=lambda i: labels[i], key='mmm_start_week')
    with col3:
        last = st.selectbox('End week', opts, index=len(dates)-1, format_func=lambda i: labels[i], key='mmm_end_week')
    if first > last:
        st.warning('Start week must be on or before end week.')
        return
    sl = slice(first, last+1)
    if view == 'Stores':
        fit, media, draws, channels = stores
    elif view == 'E-commerce':
        fit, media, draws, channels = ecom
    else:
        sf, sm, sd, channels = stores
        ef, em, ed, ec = ecom
        if not np.array_equal(channels, ec):
            st.error('Outcome subchannel mappings do not match.')
            return
        s = sm.sort_values(['week_start_date', 'subchannel']).reset_index(drop=True)
        e = em.sort_values(['week_start_date', 'subchannel']).reset_index(drop=True)
        if not s[['week_start_date','subchannel']].equals(e[['week_start_date','subchannel']]) or not np.allclose(s.spend,e.spend):
            st.error('Store and E-commerce media spend does not reconcile.')
            return
        fit = sf.copy()
        for col in ['actual_sales','fitted_sales','baseline_and_controls','marketing_contribution','residual']:
            fit[col] = sf[col] + ef[col]
        media = sm.copy()
        media['incremental_revenue'] = sm['incremental_revenue'] + em['incremental_revenue']
        draws = None  # Do not claim joint credible intervals without posterior pairing.
    window = fit.iloc[sl].copy()
    selected_dates = set(window.week_start_date)
    selected = media[media.week_start_date.isin(selected_dates)]
    spend = float(selected.spend.sum())
    inc = float(selected.incremental_revenue.sum())
    actual = float(window.actual_sales.sum())
    baseline = float(window.baseline_and_controls.sum())
    residual = float(window.residual.sum())
    if not np.isclose(actual, baseline + inc + residual, atol=0.05):
        st.error('Weekly sales reconciliation failed.')
        return
    kpis = [
        ('Demand Sales', actual), ('Marketing Sales', inc), ('Base + Controls', baseline),
        ('Spend', spend), ('iROAS', inc/spend if spend else np.nan),
        ('% Marketing Sales', inc/actual if actual else np.nan),
    ]
    cols = st.columns(3)
    for i,(name,value) in enumerate(kpis):
        display = f'{value:.2f}x' if name=='iROAS' else f'{value:.1%}' if name.startswith('%') else f'${value/1e6:,.2f}M'
        cols[i%3].metric(name, display)
    st.caption(f'{last-first+1} modeled weeks · {labels[first]} through {labels[last]} · Fitted residual: ${residual:,.0f}. Base includes seasonality, promotions, holidays and weather; these are not separately attributed.')
    if draws is not None:
        roi_draws = draws[:, sl, :].sum(axis=(1,2)) / spend if spend else np.array([])
        if roi_draws.size:
            st.caption(f'Modeled iROAS 90% posterior interval for selected weeks: {np.quantile(roi_draws,.05):.2f}x to {np.quantile(roi_draws,.95):.2f}x (joint weekly draws).')
    # Trend: weekly observed/fitted, baseline and media
    fig = go.Figure()
    for name, field, color in [('Observed sales','actual_sales','#F8FAFC'), ('Fitted sales','fitted_sales','#A78BFA'), ('Base + controls','baseline_and_controls','#64748B'), ('Marketing contribution','marketing_contribution','#10B981')]:
        fig.add_trace(go.Scatter(x=window.week_start_date,y=window[field],name=name,mode='lines',line=dict(color=color,width=2)))
    fig.update_layout(title='Weekly sales decomposition',template='plotly_dark',height=340,hovermode='x unified',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',yaxis_title='Revenue ($)',legend=dict(orientation='h',y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    pivot = selected.pivot_table(index='week_start_date',columns='subchannel',values='incremental_revenue',aggfunc='sum').fillna(0)
    stacked = px.area(pivot.reset_index().melt('week_start_date',var_name='Subchannel',value_name='Incremental revenue'),x='week_start_date',y='Incremental revenue',color='Subchannel',title='Weekly contribution by subchannel')
    stacked.update_layout(template='plotly_dark',height=400,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',hovermode='x unified')
    st.plotly_chart(stacked, use_container_width=True)
    table = selected.groupby('subchannel',as_index=False)[['spend','incremental_revenue']].sum()
    table['iROAS'] = table.incremental_revenue / table.spend.replace(0,np.nan)
    table['% of Marketing Sales'] = table.incremental_revenue / inc if inc else np.nan
    table = table.sort_values('incremental_revenue',ascending=False)
    fig2=px.bar(table.sort_values('incremental_revenue'),x='incremental_revenue',y='subchannel',orientation='h',title='Ranked incremental revenue by subchannel')
    fig2.update_layout(template='plotly_dark',height=470,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis_title='Incremental revenue ($)',yaxis_title='')
    st.plotly_chart(fig2,use_container_width=True)
    st.dataframe(table.rename(columns={'subchannel':'Subchannel','spend':'Spend','incremental_revenue':'Marketing Sales'}).style.format({'Spend':'${:,.0f}','Marketing Sales':'${:,.0f}','iROAS':'{:.2f}x','% of Marketing Sales':'{:.1%}'}),hide_index=True,use_container_width=True)
    st.caption('Selected-week mROAS and YoY contribution are not displayed until the matching counterfactual and comparison-period exports exist. Full-history mROAS from the separate aggregate table should not be reused for a custom period.')
