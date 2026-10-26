import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from retailiq.executive_v4 import render_executive, render_macro
from retailiq.theme import apply_theme
from retailiq.weather import render_weather
from retailiq.data import generate
from retailiq.models import forecast
from retailiq.discovery import discover
from retailiq.testing import difference_in_differences, randomized_test

st.set_page_config(page_title='RetailIQ AI | Executive Intelligence', page_icon='✦', layout='wide', initial_sidebar_state='expanded')
apply_theme()

@st.cache_data(show_spinner=False)
def prepare():
    data = generate()
    scored, model, quality = forecast(data)
    findings = discover(scored)
    return data, scored, quality, findings

with st.spinner('Preparing synthetic retail intelligence...'):
    data, scored, quality, findings = prepare()

NAV_GROUPS = {
    "OVERVIEW": [
        ("◈  Executive Intelligence", "Executive Intelligence"),
    ],
    "BUSINESS INTELLIGENCE": [
        ("▥  Sales Intelligence", "Sales Intelligence"),
        ("↗  Marketing Intelligence", "Marketing Intelligence"),
        ("♙  Customer Intelligence", "Customer Intelligence"),
        ("▤  Inventory Intelligence", "Inventory Intelligence"),
        ("♧  Labor Intelligence", "Labor Intelligence"),
        ("▦  Store Intelligence", "Store Intelligence"),
        ("◫  Merchandise Intelligence", "Merchandise Intelligence"),
        ("◎  Macro Intelligence", "Macro Intelligence"),
        ("☁  Weather Intelligence", "Weather Intelligence"),
    ],
    "ANALYTICS & VALIDATION": [
        ("◇  Testing Intelligence", "Testing Intelligence"),
        ("◷  Model Performance", "Model Performance"),
    ],
}

# These two areas are planned; don't imply that their models already exist.
AVAILABLE = {"Executive Intelligence", "Sales Intelligence", "Marketing Intelligence",
             "Customer Intelligence", "Merchandise Intelligence", "Store Intelligence",
             "Macro Intelligence", "Weather Intelligence", "Testing Intelligence", "Model Performance"}
if "ri_page" not in st.session_state:
    st.session_state.ri_page = "Executive Intelligence"

def _navigate(destination):
    st.session_state.ri_page = destination

with st.sidebar:
    st.markdown('<div class="ri-brand">◈ RetailIQ <span>AI</span></div><div class="ri-sidebar-caption">RETAIL INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    for group, destinations in NAV_GROUPS.items():
        st.markdown(f'<div class="ri-nav-heading">{group}</div>', unsafe_allow_html=True)
        for label, destination in destinations:
            active = st.session_state.ri_page == destination
            st.button(label + ("  · SOON" if destination not in AVAILABLE else ""),
                      key="ri_nav_" + destination.replace(" ", "_"),
                      type="primary" if active else "secondary",
                      disabled=destination not in AVAILABLE,
                      on_click=_navigate, args=(destination,), width="stretch")
    st.markdown('<div class="ri-nav-footer">● SYNTHETIC DATA DEMO<br><span>Retail intelligence · Portfolio edition</span></div>', unsafe_allow_html=True)
page = ('◈  Executive Intelligence' if st.session_state.ri_page == 'Executive Intelligence'
        else st.session_state.ri_page)

if page.startswith('◈'):

    render_executive(scored, quality, findings)
else:
    sections = ['Sales Intelligence', 'Marketing Intelligence', 'Customer Intelligence', 'Merchandise Intelligence', 'Store Intelligence', 'Weather Intelligence', 'Macro Intelligence', 'Testing Intelligence', 'Model Performance']
    chosen = next((x for x in sections if x in page), sections[0])
    tabs = st.tabs([chosen])
    # Each navigation item has its own page; reuse the existing section implementations below.
    section = sections.index(chosen)
    class _HiddenTab:
        def __enter__(self): return None
        def __exit__(self, *args): return False
    tabs = [tabs[0] if i == section else _HiddenTab() for i in range(len(sections))]
    if section == 0:
        st.markdown('#### Top financial discoveries')
        if findings.empty:
            st.write('No qualifying findings.')
        else:
            st.dataframe(findings[['group','stores','relative_dollars','relative_pct','breadth']].style.format({'relative_dollars':'${:,.0f}','relative_pct':'{:+.1f}%','breadth':'{:.0f}%'}), use_container_width=True, hide_index=True)
            choice = st.selectbox('Investigate a group', findings.group.tolist())
            row = findings[findings.group == choice].iloc[0]
            keys = row.attribute.split(' + ')
            values = choice.split(' / ')
            subset = scored.copy()
            for key, value in zip(keys, values):
                subset = subset[subset[key].astype(str) == value]
            trend = subset.groupby('date', as_index=False)[['sales','expected']].sum()
            st.plotly_chart(px.line(trend, x='date', y=['sales','expected'], title='Actual vs expected sales'), use_container_width=True)
            st.caption('Brand-relative dollars are descriptive deviations, not incremental lost sales.')
    if section == 1:
        st.markdown('#### Marketing Intelligence')
        st.write('Next module: synthetic marketing mix data, channel response curves, and constrained spend optimization. No fabricated ROI estimates are shown.')
    if section == 2:
        st.markdown('#### Customer Intelligence')
        st.write('Next module: synthetic customer cohorts, churn risk, and lifetime value.')
    if section == 3:
        st.markdown('#### Merchandise Intelligence')
        st.write('Next module: division trends, price mix, and merchandise relationships.')
    if section == 4:
        st.markdown('#### Store Intelligence')
        st.write('Explore store performance, traffic, and conversion in the synthetic dataset.')
        stores = data.groupby(['region','store'], as_index=False).agg(sales=('sales','sum'), traffic=('traffic','sum'), orders=('orders','sum'))
        stores['conversion'] = 100 * stores.orders / stores.traffic.replace(0, np.nan)
        st.dataframe(stores.sort_values('sales', ascending=False).head(25), hide_index=True, use_container_width=True)
    if section == 5:
        render_weather(scored)
    if section == 6:
        render_macro()
    if section == 7:
        st.markdown('#### Testing Intelligence')
        st.write('Design and analyze a simple hypothetical store experiment.')
        x, y = st.columns(2)
        with x:
            tb = st.number_input('Treatment before ($)', value=100000.0)
            ta = st.number_input('Treatment after ($)', value=115000.0)
        with y:
            cb = st.number_input('Control before ($)', value=100000.0)
            ca = st.number_input('Control after ($)', value=105000.0)
        lift = difference_in_differences(tb, ta, cb, ca)
        st.metric('Difference-in-differences estimate', f'${lift:+,.0f}')
        st.caption('Illustrative point estimate only. Requires pre-trend validation, independent units, uncertainty intervals, and design assumptions before causal interpretation.')
    if section == 8:
        st.markdown('#### Historical holdout validation')
        st.write(f'Trained on {quality["train_rows"]:,} synthetic store-days before {quality["cutoff"]}. Tested on the last 90 days.')
        st.metric('WAPE', f'{quality["wape"]:.2f}%')
        st.metric('Bias (predicted minus actual)', f'{quality["bias"]:+.2f}%')
        st.caption('This is a first chronological holdout, not yet a multi-fold rolling backtest.')
