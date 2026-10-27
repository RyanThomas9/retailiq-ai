"""Synthetic marketing signals anchored to the existing fictional retailer.

GA4 revenue is a digital attribution proxy, not additive to total store sales.
No incrementality or causal ROI is inferred from these signals.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from html import escape

CHANNELS = {
    'Paid Search': ('Brand', 'Non-Brand', 'Shopping'),
    'Paid Social': ('Meta', 'TikTok'),
    'Display': ('Awareness', 'Retargeting'),
    'Video': ('YouTube',),
    'Affiliate': ('Partners',),
    'Email': ('Lifecycle',),
}
PERIODS = {'Yesterday': 1, 'Last 7 Days': 7, 'Last Month': 28,
           'Last 6 Months': 182, 'Last Year': 364}

@st.cache_data(show_spinner=False)
def make_marketing(data):
    """Deterministic simulation: common dates, and sales/traffic anchors.

    Marketing spend is simulated after sales were generated. It must NOT be
    interpreted as the data-generating cause of sales.
    """
    daily = (data.groupby('date', as_index=False)
             .agg(total_sales=('sales', 'sum'), store_traffic=('traffic', 'sum'),
                  store_orders=('orders', 'sum'))).sort_values('date')
    rng = np.random.default_rng(20261027)
    channel_weight = {'Paid Search': .34, 'Paid Social': .24, 'Display': .12,
                      'Video': .12, 'Affiliate': .08, 'Email': .10}
    records = []
    for i, r in enumerate(daily.itertuples(index=False)):
        # A common underlying seasonal demand signal links the two datasets.
        seasonal = 1 + .12 * np.sin(2 * np.pi * i / 365.25)
        for channel, subchannels in CHANNELS.items():
            for sub in subchannels:
                weight = channel_weight[channel] / len(subchannels)
                spend = max(0, r.total_sales * .075 * weight * seasonal * rng.lognormal(0, .11))
                cpm = {'Paid Search': 20, 'Paid Social': 13, 'Display': 7,
                       'Video': 11, 'Affiliate': 15, 'Email': 2}[channel] * rng.lognormal(0, .08)
                impressions = spend * 1000 / cpm
                ctr = {'Paid Search': .038, 'Paid Social': .013, 'Display': .004,
                       'Video': .007, 'Affiliate': .025, 'Email': .055}[channel] * rng.lognormal(0, .09)
                clicks = int(round(impressions * ctr))
                sessions = int(round(clicks * np.clip(rng.normal(.84, .045), .65, .96)))
                # Session-to-order proxy is bounded and tied to actual store demand.
                digital_cv = np.clip(.021 * (r.store_orders / max(r.store_traffic, 1)) / .26
                                     * rng.lognormal(0, .10), .004, .07)
                transactions = int(round(sessions * digital_cv))
                # This is attributed GA4 proxy revenue, NOT additional business revenue.
                aov = r.total_sales / max(r.store_orders, 1)
                ga4_revenue = transactions * aov * rng.lognormal(0, .045)
                records.append((r.date, channel, sub, spend, impressions, clicks,
                                sessions, transactions, ga4_revenue))
    return pd.DataFrame(records, columns=['date','channel','subchannel','spend','impressions',
                                           'clicks','sessions','transactions','ga4_revenue'])


def _metrics(df):
    spend = df.spend.sum()
    clicks = df.clicks.sum()
    impressions = df.impressions.sum()
    sessions = df.sessions.sum()
    tx = df.transactions.sum()
    return {'spend': spend, 'sessions': sessions, 'ga4_revenue': df.ga4_revenue.sum(),
            'ctr': 100 * clicks / max(impressions, 1),
            'cpc': spend / max(clicks, 1),
            'cvr': 100 * tx / max(sessions, 1)}


def find_marketing_discoveries(df, days=7, minimum=25000):
    """Rank *descriptive* GA4-revenue changes, not incremental marketing impacts."""
    end = df.date.max()
    cur = df[df.date.between(end-pd.Timedelta(days=days-1), end)]
    prior = df[df.date.between(end-pd.Timedelta(days=2*days-1), end-pd.Timedelta(days=days))]
    rows = []
    for (channel, sub), group in cur.groupby(['channel','subchannel']):
        old = prior[(prior.channel == channel) & (prior.subchannel == sub)]
        if old.empty or len(old) != days or len(group) != days:
            continue
        a, b = _metrics(group), _metrics(old)
        gap = a['ga4_revenue'] - b['ga4_revenue']
        if abs(gap) < minimum:
            continue
        session_delta = a['sessions'] / max(b['sessions'], 1) - 1
        conversion_delta = a['cvr'] / max(b['cvr'], .0001) - 1
        aov_now = a['ga4_revenue'] / max(group.transactions.sum(), 1)
        aov_old = b['ga4_revenue'] / max(old.transactions.sum(), 1)
        aov_delta = aov_now / max(aov_old, .0001) - 1
        drivers = {'Sessions': session_delta, 'Conversion': conversion_delta, 'Order value': aov_delta}
        driver = max(drivers, key=lambda x: abs(drivers[x]))
        rows.append({'channel':channel,'subchannel':sub,'gap':gap,'current':a,'prior':b,
                     'driver':driver,'driver_pct':drivers[driver]*100,
                     'spend_change':100*(a['spend']/max(b['spend'],1)-1)})
    return sorted(rows, key=lambda x: abs(x['gap']), reverse=True)[:3]


def _money(x):
    return f'${abs(x)/1e6:,.2f}M' if abs(x)>=1e6 else f'${abs(x)/1e3:,.0f}K'


def render_marketing(data):
    st.markdown('<div class="ri-eyebrow">RETAILIQ AI / MARKETING INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title('Make every marketing dollar count.')
    st.caption('Synthetic marketing signals aligned to existing store sales dates and demand. GA4 is an attribution proxy; no incremental ROI claims.')
    df = make_marketing(data)
    end = df.date.max()
    current = df[df.date.between(end-pd.Timedelta(days=6),end)]
    m = _metrics(current)
    sales = data.loc[data.date.between(end-pd.Timedelta(days=6),end),'sales'].sum()
    cols = st.columns(4)
    for col, (label, value, detail) in zip(cols, [
        ('Platform spend', _money(m['spend']), 'Last 7 days'),
        ('GA4 sessions', f'{m["sessions"]:,.0f}', 'Modeled digital visits'),
        ('GA4 attributed revenue', _money(m['ga4_revenue']), 'Non-incremental proxy'),
        ('Business sales', _money(sales), 'Same source as Sales Intelligence'),
    ]):
        with col:
            st.markdown(f'<div class="ri-kpi"><div class="ri-kpi-label">{label}</div>'
                        f'<div class="ri-kpi-value">{value}</div>'
                        f'<div class="ri-kpi-bottom">{detail}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="ri-section">✦ &nbsp; Top Marketing AI Discoveries</div>', unsafe_allow_html=True)
    st.caption('Largest absolute GA4 attributed-revenue changes versus the immediately preceding equal-length period. Minimum $25K. These are descriptive, not causal.')
    tabs = st.tabs(list(PERIODS))
    for tab, (label, days) in zip(tabs, PERIODS.items()):
        with tab:
            discoveries = find_marketing_discoveries(df, days)
            if not discoveries:
                st.info('No subchannel exceeds the $25K threshold for this period, or insufficient comparable history.')
            for i, item in enumerate(discoveries, 1):
                positive = item['gap'] >= 0
                color = '#34D399' if positive else '#FB7185'
                status = 'OUTPERFORMING' if positive else 'UNDERPERFORMING'
                direction = 'higher' if positive else 'lower'
                recommendation = {
                    'Sessions': ('Investigate reach, search demand, delivery and landing-page traffic.' if not positive else 'Check which demand and acquisition patterns can be replicated.'),
                    'Conversion': ('Review landing-page experience, product availability and checkout friction.' if not positive else 'Identify conversion improvements and test replication.'),
                    'Order value': ('Review discounting, merchandise mix and basket composition.' if not positive else 'Investigate higher-value baskets and merchandising practices.'),
                }[item['driver']]
                st.markdown(f'''<div style="background:#20172e;border:1px solid #49325f;border-left:4px solid {color};border-radius:13px;padding:22px;margin:14px 0;">
                    <div style="color:{color};font-weight:800;letter-spacing:.08em;font-size:13px">#{i} &nbsp; {status} &nbsp; · &nbsp; {days}-DAY WINDOW</div>
                    <div style="font-size:23px;font-weight:750;color:#f8fafc;margin-top:12px">{escape(item['channel'])} / {escape(item['subchannel'])}: {escape(item['driver'].lower())} is the largest relative change</div>
                    <div style="font-size:34px;font-weight:850;color:{color};margin:10px 0">{'+' if positive else '−'}{_money(item['gap'])}</div>
                    <div style="color:#cbd5e1">GA4 attributed revenue is {direction} than the preceding period. {escape(item['driver'])} changed {item['driver_pct']:+.1f}%; spend changed {item['spend_change']:+.1f}%.</div>
                    <div style="background:#342a43;border-left:3px solid #facc15;color:#fde68a;padding:12px;margin-top:14px;border-radius:6px"><b>What to investigate:</b> {escape(recommendation)}</div>
                    <div style="color:#a6a1b7;font-size:12px;margin-top:10px">Exploratory · GA4 attribution proxy · Not incremental revenue or proven financial impact</div>
                    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="ri-section">Channel performance</div>', unsafe_allow_html=True)
    by_channel = (current.groupby('channel',as_index=False)
                  .agg(spend=('spend','sum'),sessions=('sessions','sum'),
                       ga4_revenue=('ga4_revenue','sum'),clicks=('clicks','sum')))
    fig = px.bar(by_channel.sort_values('spend'),x='spend',y='channel',orientation='h',
                 color_discrete_sequence=['#9b7ae4'],title='Platform spend · Last 7 days')
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',height=350,margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander('GA4 and platform details'):
        st.dataframe(by_channel.sort_values('spend',ascending=False),hide_index=True,use_container_width=True)
        st.caption('Platform spend, clicks, and impressions are simulated. GA4 sessions and revenue are synthetic attributed signals, not additions to business sales.')
    st.info('MMM and budget optimization are intentionally not shown yet: this dataset was constructed after sales generation and cannot validate causal media contribution.')
