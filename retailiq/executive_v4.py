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
                <div class="ri-kpi-bottom {'positive' if np.isfinite(m['yoy_pct']) and m['yoy_pct'] >= 0 else 'negative' if np.isfinite(m['yoy_pct']) else ''}">YoY {pct(m['yoy_pct'])}</div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="ri-section">✦ &nbsp; Top AI Discoveries Found Yesterday</div>', unsafe_allow_html=True)
    st.caption('Most recent available model run · These are exploratory store-group patterns, not confirmed causal findings. Daily scheduling is not yet enabled.')
    st.markdown("""<style>
    /* Discovery cards: deliberate, legible semantic badge hierarchy */
    .ri-discovery {padding:27px 29px!important; margin:16px 0 20px!important; border-radius:16px!important;}
    .ri-discovery-tags {align-items:center!important;gap:9px!important;margin-bottom:18px!important;}
    .ri-discovery-tags span {display:inline-flex;align-items:center;min-height:29px;padding:6px 12px!important;border-radius:999px!important;font-size:.78rem!important;font-weight:800!important;letter-spacing:.025em!important;line-height:1.2!important;}
    .ri-discovery-tags .ri-rank {background:transparent!important;color:#F8FAFC!important;font-size:1rem!important;padding-left:0!important;padding-right:4px!important;}
    .ri-discovery-tags .ri-status.good {background:#064E3B!important;color:#6EE7B7!important;border:1px solid #10B981!important;}
    .ri-discovery-tags .ri-status.bad {background:#5B2037!important;color:#FFB0C1!important;border:1px solid #FB7185!important;}
    .ri-discovery-tags .ri-trend {background:#35204F!important;color:#E9D5FF!important;border:1px solid #8555C6!important;}
    .ri-discovery-tags .ri-count {background:#283146!important;color:#E2E8F0!important;border:1px solid #56657F!important;}
    .ri-discovery-tags .ri-exploratory {background:transparent!important;color:#D8B4FE!important;border:1px solid #A855F7!important;}
    .ri-discovery-heading {font-size:clamp(1.35rem,1.7vw,1.75rem)!important;font-weight:800!important;line-height:1.35!important;letter-spacing:-.02em!important;}
    .ri-discovery-impact {font-size:clamp(2.15rem,2.7vw,2.8rem)!important;line-height:1.15!important;margin:14px 0!important;}
    .ri-discovery-evidence {font-size:.96rem!important;line-height:1.6!important;}
    .ri-discovery-action {padding:14px 17px!important;font-size:.94rem!important;line-height:1.55!important;border-left:4px solid #FACC15!important;}
    .ri-discovery-foot {font-size:.8rem!important;line-height:1.5!important;}
    .ri-kpi .ri-kpi-value {color:#F7F9FF}
    .ri-kpi .ri-kpi-bottom.positive {color:#34D399!important;font-weight:700}
    .ri-kpi .ri-kpi-bottom.negative {color:#FB7185!important;font-weight:700}
    .ri-discovery {margin:13px 0 18px!important;padding:23px 26px!important;}
    .ri-discovery.positive {border-left:4px solid #34D399!important}
    .ri-discovery.negative {border-left:4px solid #FB7185!important}
    .ri-discovery.positive .ri-discovery-impact,.ri-discovery.positive .ri-number {color:#34D399!important}
    .ri-discovery.negative .ri-discovery-impact,.ri-discovery.negative .ri-number {color:#FB7185!important}
    .ri-discovery-heading {font-size:1.27rem!important;line-height:1.4!important}
    .ri-discovery-action {border-left:3px solid #FACC15;color:#FFE59B!important}
    .ri-discovery-evidence {color:#C6B4F3;font-size:.85rem;margin:10px 0}
    .ri-discovery-foot {font-size:.75rem;color:#9AAAC3;margin-top:12px}
    </style>""", unsafe_allow_html=True)
    if findings.empty:
        st.info('No qualifying discoveries in the latest synthetic data.')
    else:
        for rank, (_, row) in enumerate(findings.head(3).iterrows(), start=1):
            delta = float(row.relative_dollars)
            positive = delta >= 0
            direction = 'outperforming' if positive else 'underperforming'
            impact = ('+' if positive else '−') + dollars(abs(delta))
            color = 'positive' if positive else 'negative'
            group = escape(str(row.group))
            relative_pct = pct(float(row.relative_pct))
            stores = int(row.stores)
            breadth = float(row.breadth)
            st.markdown(f'''<div class="ri-discovery {color}">
                <div class="ri-discovery-tags"><span class="ri-rank">#{rank}</span><span class="ri-status {'good' if positive else 'bad'}">{'OUTPERFORMING' if positive else 'UNDERPERFORMING'}</span><span class="ri-trend">28-DAY PATTERN</span><span class="ri-count">{stores} STORES</span><span class="ri-exploratory">EXPLORATORY</span></div>
                <div class="ri-discovery-heading">{group} stores are {direction} their brand-adjusted benchmark</div>
                <div class="ri-discovery-impact">{impact} <small>brand-relative deviation over the analysis window</small></div>
                <div class="ri-discovery-evidence">Relative performance: <strong class="ri-number">{relative_pct}</strong> · Stores sharing the direction: {breadth:.0f}%</div>
                <div class="ri-discovery-action"><strong>What to check:</strong> Compare traffic, conversion, order size, product availability and staffing coverage with comparable stores.</div>
                <div class="ri-discovery-foot">Exploratory store-group finding · Descriptive association, not a confirmed causal or incremental sales impact.</div>
            </div>''', unsafe_allow_html=True)
        if len(findings) > 3:
            with st.expander('View additional discoveries'):
                st.dataframe(findings[['group','stores','relative_dollars','relative_pct','breadth']], hide_index=True, width='stretch')
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
