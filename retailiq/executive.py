"""Executive briefing: synthetic retail metrics and documented public macro signals."""
import io
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from html import escape
from retailiq.weather import forecast_summary

PERIODS = {'Yesterday': 1, '1 Week': 7, '13 Weeks': 91, '52 Weeks': 364}


def period_metrics(scored, days):
    end = scored.date.max()
    start = end - pd.Timedelta(days=days - 1)
    current = scored.loc[scored.date.between(start, end)]
    prev = scored.loc[scored.date.between(start - pd.Timedelta(days=days), start - pd.Timedelta(days=1))]
    last_year = scored.loc[scored.date.between(start - pd.Timedelta(days=364), end - pd.Timedelta(days=364))]
    actual, expected = current.sales.sum(), current.expected.sum()
    return {'actual': actual, 'expected': expected, 'gap': actual-expected,
            'gap_pct': 100*(actual/expected-1) if expected else np.nan,
            'prior_pct': 100*(actual/prev.sales.sum()-1) if len(prev) and prev.sales.sum() else np.nan,
            'yoy_pct': 100*(actual/last_year.sales.sum()-1) if len(last_year)==len(current) and last_year.sales.sum() else np.nan,
            'rows': len(current)}


def dollars(value):
    return f'${value/1e6:,.2f}M' if abs(value)>=1e6 else f'${value/1e3:,.0f}K'


def pct(value):
    return f'{value:+.1f}%' if np.isfinite(value) else 'N/A'


@st.cache_data(ttl=86400, show_spinner=False)
def macro_series(series_id):
    """Read a public FRED series, no token or credentials required. Never invent fallback values."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    response = requests.get(url, timeout=7, headers={'User-Agent':'RetailIQ-portfolio-demo/1.0'})
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = ['date', 'value']
    frame['date'] = pd.to_datetime(frame.date)
    frame['value'] = pd.to_numeric(frame.value, errors='coerce')
    return frame.dropna().sort_values('date')


def render_macro():
    st.subheader('Real-world economic intelligence')
    st.caption('Public FRED observations. These are context, not evidence that an event caused a change in synthetic sales.')
    indicators = [('Unemployment rate', 'UNRATE', '%'), ('Consumer Price Index', 'CPIAUCSL', 'index'),
                  ('Retail & food services sales', 'RSAFS', '$ millions')]
    cols = st.columns(3)
    for col, (label, sid, unit) in zip(cols, indicators):
        with col:
            try:
                frame = macro_series(sid)
                if frame.empty:
                    raise ValueError('No observations available')
                latest = frame.iloc[-1]
                st.metric(label, f'{latest.value:,.1f} {unit}' if unit != '$ millions' else f'${latest.value:,.0f}M')
                st.caption(f'Observation: {latest.date:%b %Y} · Source: FRED ({sid})')
                st.line_chart(frame.tail(24).set_index('date')['value'], height=125)
            except Exception:
                st.info(f'{label}: source temporarily unavailable')
                st.caption(f'FRED series {sid}; no substitute values displayed.')
    st.markdown('[FRED source and definitions](https://fred.stlouisfed.org/)')
    st.info('External conditions are not automatically treated as sales drivers. A future model will align external signals with geography and time, test lagged relationships, and validate incremental predictive value.')



def render_executive(scored, quality, findings):
    end = scored.date.max()
    st.markdown('<div class="ri-eyebrow">◈ &nbsp; RetailIQ / Executive Intelligence</div>', unsafe_allow_html=True)
    st.title('Your business, analyzed.')
    st.markdown(f'<div class="ri-subtitle">A model-driven briefing on retail performance &nbsp;·&nbsp; Synthetic data through {end:%B %d, %Y}</div>', unsafe_allow_html=True)
    metrics = {label: period_metrics(scored, days) for label, days in PERIODS.items()}
    cols = st.columns(4, gap='medium')
    for col, (label, m) in zip(cols, metrics.items()):
        tone = 'pos' if m['gap'] >= 0 else 'neg'
        sign = '↑' if m['gap'] >= 0 else '↓'
        with col:
            st.markdown(f'''<div class="ri-kpi"><div class="ri-kpi-label">{label}</div>
            <div class="ri-kpi-value">{dollars(m['actual'])}</div>
            <span class="ri-chip {tone}">{sign} {pct(m['gap_pct'])} vs expected</span>
            <div class="ri-meta">Expected {dollars(m['expected'])} &nbsp;·&nbsp; Gap {dollars(m['gap'])}<br>
            Prior {pct(m['prior_pct'])} &nbsp;·&nbsp; YoY {pct(m['yoy_pct'])}</div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="ri-section">Executive briefing</div>', unsafe_allow_html=True)
    short = metrics['1 Week']; long = metrics['13 Weeks']
    gap_desc = 'below' if short['gap'] < 0 else 'above'
    message = (f'Over the past seven days, sales of <b>{dollars(short["actual"])}</b> were '
               f'<b>{abs(short["gap_pct"]):.1f}% {gap_desc} model expectations</b> '
               f'({dollars(abs(short["gap"]))} in absolute variance). '
               f'The 13-week comparison is <b>{pct(long["gap_pct"])} versus expected</b>.')
    if not findings.empty:
        top = findings.iloc[0]
        message += (f'<br><br><b>Priority investigation:</b> {escape(str(top.group))} stores show '
                    f'a {dollars(abs(top.relative_dollars))} brand-relative deviation over 28 days '
                    f'across {top.stores} stores. This is a descriptive pattern, not a causal estimate.')
    st.markdown(f'<div class="ri-brief"><div class="ri-brief-title">✦ &nbsp; AI-style executive assessment</div><div class="ri-brief-body">{message}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1], gap='large')
    with left:
        st.markdown('<div class="ri-section">Sales trajectory</div>', unsafe_allow_html=True)
        days = st.segmented_control('View', ['4 Weeks', '13 Weeks', '52 Weeks'], default='13 Weeks', label_visibility='collapsed')
        n = {'4 Weeks':28, '13 Weeks':91, '52 Weeks':364}[days]
        trend = scored.loc[scored.date > end - pd.Timedelta(days=n)].groupby('date', as_index=False)[['sales','expected']].sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend.date, y=trend.sales, name='Actual', mode='lines', line=dict(color='#B59AFF', width=2.8), fill='tozeroy', fillcolor='rgba(155,124,235,0.07)'))
        fig.add_trace(go.Scatter(x=trend.date, y=trend.expected, name='Model expected', mode='lines', line=dict(color='#63C5C9', width=2, dash='dot')))
        fig.update_layout(height=310, margin=dict(l=4,r=8,t=10,b=5), paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#A8B6CD',size=11),legend=dict(orientation='h',y=1.16,x=0),hovermode='x unified',xaxis=dict(showgrid=False),yaxis=dict(gridcolor='#263347',tickprefix='$',tickformat='.2s'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.caption('Historical fitted values before the 90-day holdout; do not interpret the full 52-week line as out-of-sample accuracy.')
    with right:
        st.markdown('<div class="ri-section">What needs your attention</div>', unsafe_allow_html=True)
        if findings.empty:
            st.info('No qualifying store-group findings.')
        else:
            for i, r in findings.head(3).reset_index(drop=True).iterrows():
                st.markdown(f'<div class="ri-finding"><div class="ri-finding-title">{i+1:02d} &nbsp; {escape(str(r.group))} &nbsp;·&nbsp; {pct(r.relative_pct)}</div><div class="ri-finding-meta">{r.stores} stores &nbsp;·&nbsp; {dollars(abs(r.relative_dollars))} brand-relative deviation</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="ri-section">Weather risk · next 7 days</div>', unsafe_allow_html=True)
    wx, wx_errors = forecast_summary()
    if wx.empty:
        st.info('Public weather forecast unavailable. Weather risk not assessed.')
    else:
        active = wx.loc[wx.severe | wx.snow]
        if active.empty:
            st.success('No snow or severe-weather threshold events forecast for the four representative regional cities.')
        else:
            regions = ', '.join(sorted(active.region.unique()))
            st.warning(f'{len(active)} region-days above severe-weather or snow thresholds across {regions}. See Weather Intelligence for details.')
        if wx_errors:
            st.caption('Forecast unavailable for: ' + ', '.join(wx_errors))
    st.caption('Forecasts are for representative cities, not actual stores. Weather-related lost sales are not estimated.')
    st.markdown('<div class="ri-section">Real-world economic intelligence</div>', unsafe_allow_html=True)
    render_macro()
    st.markdown(f'<div class="ri-footnote">Synthetic portfolio demonstration · Holdout WAPE {quality["wape"]:.1f}% over 90 days · External indicators are context, not proven causes of sales changes.</div>', unsafe_allow_html=True)
