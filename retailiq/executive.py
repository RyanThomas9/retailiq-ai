"""Executive briefing: synthetic retail metrics and documented public macro signals."""
import io
import pandas as pd
import numpy as np
import requests
import streamlit as st

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
    st.caption('RETAILIQ AI  /  EXECUTIVE INTELLIGENCE  /  SYNTHETIC RETAIL DATA')
    st.title('Your business, analyzed.')
    st.write('What happened. Why it matters. What to investigate next.')
    st.caption(f'Synthetic sales through {scored.date.max():%b %d, %Y} · Public economic observations are live when available')
    st.subheader('Sales performance')
    cols = st.columns(4)
    metrics = {}
    for col, (label, days) in zip(cols, PERIODS.items()):
        m = period_metrics(scored, days)
        metrics[label] = m
        with col:
            st.metric(label, dollars(m['actual']), pct(m['gap_pct']) + ' vs expected')
            st.caption(f'Expected {dollars(m["expected"])} · Gap {dollars(m["gap"])}')
            st.caption(f'Prior period: {pct(m["prior_pct"])} · YoY: {pct(m["yoy_pct"])}')
    st.subheader('Executive assessment')
    short = metrics['1 Week']
    long = metrics['13 Weeks']
    direction = 'below' if short['gap'] < 0 else 'above'
    st.markdown(f'**Sales for the last seven days were {pct(short["gap_pct"])} versus the model expectation**, '
                f'with a {dollars(abs(short["gap"]))} {"shortfall" if short["gap"] < 0 else "surplus"}. '
                f'The 13-week comparison is {pct(long["gap_pct"])} versus expected.')
    if not findings.empty:
        top = findings.iloc[0]
        st.write(f'**Priority investigation:** {top.group} stores show a {dollars(abs(top.relative_dollars))} '
                 f'brand-relative deviation over the latest 28 days, across {top.stores} stores. '
                 'This is a descriptive association, not proven incremental lost sales.')
    else:
        st.write('No store-group patterns met the current discovery thresholds.')
    st.caption('Briefing generated from model outputs, not an LLM. External data is not used to assert causes.')
    st.subheader('What needs your attention')
    if not findings.empty:
        for i, r in findings.head(3).reset_index(drop=True).iterrows():
            with st.container(border=True):
                st.markdown(f'**{i+1:02d} · {r.group}** — {pct(r.relative_pct)} brand-relative deviation')
                st.caption(f'{r.stores} stores · {dollars(abs(r.relative_dollars))} deviation · {r.breadth:.0f}% same-direction breadth')
    else:
        st.info('No qualifying high-priority discoveries in this run.')
    render_macro()
    st.subheader('Explore intelligence')
    st.write('Use the navigation on the left to open Sales, Marketing, Customer, Merchandise, Store, Macro, Testing or Model Performance.')
    st.caption(f'Holdout WAPE: {quality["wape"]:.1f}% · Forecast performance is measured on the most recent 90 days, not the entire history.')
