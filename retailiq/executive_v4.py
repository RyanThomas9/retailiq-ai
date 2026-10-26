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
    # Focused executive home
    from retailiq.market_watch import render_market_watch
    from retailiq.weather import forecast_summary
    end = scored.date.max()
    st.markdown('<div class="ri-eyebrow">RETAILIQ AI / EXECUTIVE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title('Your business, analyzed.')
    st.markdown(f'<div class="ri-subtitle">Performance · AI discoveries · Market intelligence &nbsp; | &nbsp; Synthetic data through {end:%b %d, %Y}</div>', unsafe_allow_html=True)
    metrics = {label: period_metrics(scored, days) for label, days in PERIODS.items()}
    cols = st.columns(4, gap='medium')
    for col, (label, m) in zip(cols, metrics.items()):
        tone = 'pos' if m['gap'] >= 0 else 'neg'
        with col:
            st.markdown(f'''<div class="ri-kpi"><div class="ri-kpi-label">{label}</div>
                <div class="ri-kpi-value">{dollars(m['actual'])}</div>
                <span class="ri-chip {tone}">{pct(m['gap_pct'])} vs expected</span>
                <div class="ri-kpi-bottom">YoY {pct(m['yoy_pct'])}</div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="ri-section">✦ &nbsp; Top AI Discoveries Found Yesterday</div>', unsafe_allow_html=True)
    st.caption('Most recent available model run · These are exploratory store-group patterns, not confirmed causal findings. Daily scheduling is not yet enabled.')
    if findings.empty:
        st.info('No qualifying discoveries in the latest synthetic data.')
    else:
        top = findings.iloc[0]
        positive = float(top.relative_dollars) >= 0
        label = 'Outperforming peers' if positive else 'Underperforming peers'
        impact = ('+' if positive else '−') + dollars(abs(float(top.relative_dollars)))
        color = 'positive' if positive else 'negative'
        st.markdown(f'''<div class="ri-discovery {color}">
            <div class="ri-discovery-tags"><span>TOP DISCOVERY</span><span>28-DAY PATTERN</span><span>{int(top.stores)} STORES</span><span>EXPLORATORY</span></div>
            <div class="ri-discovery-heading">{escape(str(top.group))} stores are {label.lower()}</div>
            <div class="ri-discovery-impact">{impact} <small>brand-relative deviation</small></div>
            <div class="ri-discovery-detail">Relative variance {pct(float(top.relative_pct))} · {float(top.breadth):.0f}% of stores share the direction.</div>
            <div class="ri-discovery-action">What to investigate: Review traffic, conversion, order size, inventory availability and labor coverage in this store group.</div>
            </div>''', unsafe_allow_html=True)
        for idx, row in findings.iloc[1:3].iterrows():
            st.markdown(f'''<div class="ri-compact-discovery"><span>{escape(str(row.group))}</span><strong>{pct(float(row.relative_pct))}</strong><small>{int(row.stores)} stores · {dollars(abs(float(row.relative_dollars)))} relative deviation</small></div>''', unsafe_allow_html=True)
        if len(findings) > 3:
            with st.expander('View additional discoveries'):
                st.dataframe(findings[['group','stores','relative_dollars','relative_pct','breadth']], hide_index=True, use_container_width=True)
    # Severe weather is elevated only when a public representative-city forecast crosses thresholds.
    # Failure to retrieve forecasts is not interpreted as 'no risk'.
    try:
        wx, errors = forecast_summary()
        if not wx.empty:
            events = wx.loc[wx.severe | wx.snow]
            if not events.empty:
                places = ', '.join(sorted(events.city.unique()))
                st.warning(f'Weather alert: snow or severe weather threshold forecast in {places} ({len(events)} city-days). Representative cities only; no store-level sales impact estimated.')
    except Exception:
        pass
    st.divider()
    render_market_watch()
    st.markdown(f'<div class="ri-footnote">Synthetic portfolio demonstration · 90-day model holdout WAPE {quality["wape"]:.1f}% · No company data or credentials.</div>', unsafe_allow_html=True)
