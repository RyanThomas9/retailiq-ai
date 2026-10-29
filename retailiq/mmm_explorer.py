"""MMM training-data explorer and honest pre-fit validation. No fitted Meridian outputs."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from retailiq.mmm_synthetic import generate, validate

@st.cache_data(show_spinner=False)
def load_mmm_data():
    model, spend, truth, baseline = generate()
    checks = validate(model, spend, truth, baseline)
    return model, spend, checks

def _fmt(v):
    return f'${v/1e6:,.2f}M'

def render_mmm_explorer():
    st.markdown('<div class="ri-section">MMM Data Explorer</div>', unsafe_allow_html=True)
    st.caption('Separate 156-week synthetic MMM training scenario. These sales are NOT the live Executive/Sales Intelligence sales series.')
    try:
        weekly, long, checks = load_mmm_data()
    except Exception as exc:
        st.error(f'MMM training data could not be generated: {exc}')
        return
    spend_by_week = long.groupby('week_start_date', as_index=False).spend.sum()
    total_spend = float(long.spend.sum())
    kpis = [('Weeks', str(len(weekly))), ('Total sales', _fmt(weekly.total_sales.sum())),
            ('Store sales', _fmt(weekly.store_sales.sum())), ('E-commerce sales', _fmt(weekly.ecom_sales.sum())),
            ('Marketing spend / sales', f'{total_spend/weekly.total_sales.sum():.2%}')]
    for col, (label, val) in zip(st.columns(5), kpis):
        col.metric(label, val)
    st.markdown('#### Weekly sales by business')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.week_start_date, y=weekly.store_sales, name='Stores', line_color='#A78BFA'))
    fig.add_trace(go.Scatter(x=weekly.week_start_date, y=weekly.ecom_sales, name='E-commerce', line_color='#34D399'))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=330, margin=dict(l=5,r=5,t=20,b=5), yaxis_title='Weekly sales ($)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('#### Marketing investment')
    channel = long.groupby('channel', as_index=False).spend.sum().sort_values('spend')
    fig = px.bar(channel, x='spend', y='channel', orientation='h', color_discrete_sequence=['#A78BFA'])
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=330, margin=dict(l=5,r=5,t=10,b=5), xaxis_title='Three-year spend ($)', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    with st.expander('13-subchannel spend breakdown'):
        subs = long.groupby(['channel','subchannel'], as_index=False).spend.sum().sort_values(['channel','subchannel'])
        st.dataframe(subs.style.format({'spend':'${:,.0f}'}), hide_index=True, use_container_width=True)
    st.markdown('#### Retail calendar effects')
    flags = ['black_friday','cyber_monday','pre_christmas','christmas_week','thanksgiving_week','new_year_week','back_to_school','promotion','severe_weather']
    summary = pd.DataFrame({'Event':flags,'Weeks flagged':[int(weekly[f].sum()) for f in flags]})
    st.dataframe(summary, hide_index=True, use_container_width=True)
    with st.expander('Weekly training inputs and validation checks'):
        st.dataframe(weekly.tail(20).sort_values('week_start_date', ascending=False), hide_index=True, use_container_width=True)
        st.success(f'{checks["weeks"]} weeks · {checks["subchannels"]} subchannels · weekly spend ratio {checks["weekly_spend_ratio_min"]:.2%}–{checks["weekly_spend_ratio_max"]:.2%}.')
        st.caption('Store + e-commerce = total sales; 13 subchannels reconcile to planned spend. Hidden ground truth is excluded from training inputs.')
    st.caption('The fitted results are shown above; this explorer documents the separate synthetic training scenario.')

def render_mmm_validation():
    from retailiq.mmm_results import render_mmm_model_validation
    render_mmm_model_validation()
