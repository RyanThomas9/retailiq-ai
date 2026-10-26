import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from retailiq.data import generate
from retailiq.models import forecast
from retailiq.discovery import discover
from retailiq.testing import difference_in_differences, randomized_test

st.set_page_config(page_title='RetailIQ', page_icon='🧠', layout='wide')

@st.cache_data(show_spinner=False)
def prepare():
    data = generate()
    scored, model, quality = forecast(data)
    findings = discover(scored)
    return data, scored, quality, findings

with st.spinner('Preparing synthetic retail intelligence...'):
    data, scored, quality, findings = prepare()

st.caption('RETAILIQ  /  AI-POWERED BUSINESS INTELLIGENCE  /  SYNTHETIC DEMO')
st.title('Your business, analyzed.')
st.write('An executive briefing driven by predictive models and automated financial discovery — not a conventional dashboard.')

recent = scored[scored.date >= scored.date.max() - pd.Timedelta(days=6)]
actual, expected = recent.sales.sum(), recent.expected.sum()
a, b, c = st.columns(3)
a.metric('Last 7 days sales', f'${actual/1e6:,.2f}M')
b.metric('Versus expected', f'${(actual-expected)/1e3:+,.0f}K', f'{100*(actual/expected-1):+.1f}%')
c.metric('Daily store forecast WAPE', f'{quality["wape"]:.1f}%')

st.subheader('AI-style executive assessment')
if not findings.empty:
    top = findings.iloc[0]
    direction = 'below' if top.relative_dollars < 0 else 'above'
    st.info(f'In this fictional retail business, {top.group} stores are ${abs(top.relative_dollars):,.0f} {direction} brand-adjusted expectations over the last 28 days. The pattern spans {top.stores} stores. Investigate whether conversion, discounting, or customer mix differs from comparable stores. This is an observed association, not a proven cause.')
else:
    st.info('No store-attribute combinations passed the current financial deviation threshold. This is a valid result.')
st.caption('Narrative generated deterministically from model outputs; no external language-model API or live macroeconomic claims.')

st.subheader('Explore intelligence')
tabs = st.tabs(['Sales Intelligence', 'Marketing Intelligence', 'Customer Intelligence', 'Merchandise Intelligence', 'Store Intelligence', 'Macro Intelligence', 'Testing Intelligence', 'Model Performance'])
with tabs[0]:
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
with tabs[1]:
    st.markdown('#### Marketing Intelligence')
    st.write('Next module: synthetic marketing mix data, channel response curves, and constrained spend optimization. No fabricated ROI estimates are shown.')
with tabs[2]:
    st.markdown('#### Customer Intelligence')
    st.write('Next module: synthetic customer cohorts, churn risk, and lifetime value.')
with tabs[3]:
    st.markdown('#### Merchandise Intelligence')
    st.write('Next module: division trends, price mix, and merchandise relationships.')
with tabs[4]:
    st.markdown('#### Store Intelligence')
    st.write('Explore store performance, traffic, and conversion in the synthetic dataset.')
    stores = data.groupby(['region','store'], as_index=False).agg(sales=('sales','sum'), traffic=('traffic','sum'), orders=('orders','sum'))
    stores['conversion'] = 100 * stores.orders / stores.traffic.replace(0, np.nan)
    st.dataframe(stores.sort_values('sales', ascending=False).head(25), hide_index=True, use_container_width=True)
with tabs[5]:
    st.markdown('#### Macro Intelligence')
    st.write('Planned: documented public macroeconomic sources, freshness dates, and carefully qualified associations with retail performance.')
with tabs[6]:
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
with tabs[7]:
    st.markdown('#### Historical holdout validation')
    st.write(f'Trained on {quality["train_rows"]:,} synthetic store-days before {quality["cutoff"]}. Tested on the last 90 days.')
    st.metric('WAPE', f'{quality["wape"]:.2f}%')
    st.metric('Bias (predicted minus actual)', f'{quality["bias"]:+.2f}%')
    st.caption('This is a first chronological holdout, not yet a multi-fold rolling backtest.')
