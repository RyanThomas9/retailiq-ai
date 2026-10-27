"""Synthetic marketing signals anchored to the existing fictional retailer.

GA4 revenue is a digital attribution proxy, not additive to total store sales.
No incrementality or causal ROI is inferred from these signals.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from html import escape
from retailiq.ecommerce import generate_ecommerce

# Exact requested taxonomy. Subchannel shares are conditional on channel spend.
CHANNELS = {
    'Paid Search': {'Paid Search- Conversion': 1.0},
    'Paid Social': {'Paid Social- Awareness': .20, 'Paid Social- Consideration': .15, 'Paid Social- Conversion': .65},
    'Video': {'Video -Awareness': .70, 'Video -Consideration': .25, 'Video -Conversion': .05},
    'Display': {'Display-Awareness': .20, 'Display-Consideration': .15, 'Display-Conversion': .65},
    'Email': {'Email- Conversion': 1.0},
    'Affiliate': {'Affiliate- Conversion': 1.0},
    'Linear TV': {'Linear TV- Awareness': 1.0},
}
CHANNEL_WEIGHTS = {'Paid Search': .32, 'Paid Social': .23, 'Video': .12,
                   'Display': .09, 'Email': .04, 'Affiliate': .06, 'Linear TV': .14}
TARGET_SPEND_SHARE = .05
assert abs(sum(CHANNEL_WEIGHTS.values()) - 1) < 1e-12
assert all(abs(sum(parts.values()) - 1) < 1e-12 for parts in CHANNELS.values())

PERIODS = {'Yesterday': 1, 'Last 7 Days': 7, 'Last Month': 28,
           'Last 6 Months': 182, 'Last Year': 364}

@st.cache_data(show_spinner=False)
def make_marketing(data):
    """Generate synthetic platform activity; reconcile GA4 revenue to online sales.

    The simulator is observational. Platform spend is not the causal driver of
    the previously generated store or e-commerce sales.
    """
    from retailiq.ecommerce import generate_ecommerce
    daily = generate_ecommerce(data)
    rng = np.random.default_rng(20261027)
    records = []
    for i, r in enumerate(daily.itertuples(index=False)):
        day_records = []
        # Every day's channel + subchannel spend reconciles to 5% of total sales.
        # Small stochastic shifts in mix are normalized so total remains anchored.
        mix = {(channel, sub): CHANNEL_WEIGHTS[channel] * sub_share * rng.lognormal(0, .07)
               for channel, parts in CHANNELS.items() for sub, sub_share in parts.items()}
        mix_total = sum(mix.values())
        for (channel, sub), raw_weight in mix.items():
            spend = float(r.total_business_sales * TARGET_SPEND_SHARE * raw_weight / mix_total)
            if channel == 'Linear TV':
                # Offline awareness delivery is not represented as website clicks.
                impressions = spend * 1000 / 17.0
                clicks, sessions = 0, 0
                weight_tx = 0.0
            else:
                cpm = {'Paid Search':20, 'Paid Social':13, 'Display':7,
                       'Video':11, 'Affiliate':15, 'Email':2}[channel] * rng.lognormal(0, .08)
                impressions = spend * 1000 / cpm
                ctr = {'Paid Search':.038, 'Paid Social':.013, 'Display':.004,
                       'Video':.007, 'Affiliate':.025, 'Email':.055}[channel] * rng.lognormal(0, .09)
                clicks = int(round(impressions * ctr))
                sessions = max(1, int(round(clicks * np.clip(rng.normal(.84, .045), .65, .96))))
                weight_tx = sessions * {'Paid Search':1.3,'Paid Social':.8,'Display':.5,
                                        'Video':.45,'Affiliate':1.4,'Email':1.65}[channel]
            day_records.append([r.date,channel,sub,spend,impressions,clicks,sessions,weight_tx])
        weights = np.array([x[-1] for x in day_records], dtype=float)
        weights /= weights.sum()
        coverage = float(np.clip(rng.normal(.945, .012), .91, .97))
        tracked_revenue = float(r.ecom_sales * coverage)
        allocation = tracked_revenue * weights
        # Put rounding residual on a digital subchannel, never on Linear TV.
        digital_last = max(k for k, item in enumerate(day_records) if item[1] != "Linear TV")
        allocation[digital_last] += tracked_revenue - allocation.sum()
        estimated_orders = np.maximum(0, np.rint(r.ecom_orders * coverage * weights).astype(int))
        for j, item in enumerate(day_records):
            records.append((*item[:7], int(estimated_orders[j]), float(allocation[j])))
    return pd.DataFrame(records, columns=['date','channel','subchannel','spend','impressions',
                                           'clicks','sessions','transactions','ga4_revenue'])


def _metrics(df):
    spend = float(df.spend.sum())
    clicks = float(df.clicks.sum())
    impressions = float(df.impressions.sum())
    sessions = float(df.sessions.sum())
    tx = float(df.transactions.sum())
    revenue = float(df.ga4_revenue.sum())
    return {'spend': spend, 'impressions': impressions, 'clicks': clicks,
            'sessions': sessions, 'transactions': tx, 'ga4_revenue': revenue,
            'ctr': clicks / impressions if impressions else np.nan,
            'cpm': 1000 * spend / impressions if impressions else np.nan,
            'cpc': spend / clicks if clicks else np.nan,
            'click_to_session': sessions / clicks if clicks else np.nan,
            'cvr': tx / sessions if sessions else np.nan,
            'aov': revenue / tx if tx else np.nan,
            'rps': revenue / sessions if sessions else np.nan}


def _pct_change(new, old):
    if old is None or not np.isfinite(old) or old <= 0 or not np.isfinite(new):
        return np.nan
    return 100 * (new / old - 1)


def _contributions(a, b):
    """Exact symmetric Shapley decomposition of SESSIONS x CVR x AOV.

    A zero-transaction period has undefined AOV; avoid inventing a breakdown.
    """
    import itertools
    names = ('sessions', 'cvr', 'aov')
    if any(not np.isfinite(a[k]) or not np.isfinite(b[k]) for k in names):
        return None
    effects = {k: 0.0 for k in names}
    for ordering in itertools.permutations(names):
        values = {k: b[k] for k in names}
        for k in ordering:
            before = np.prod(list(values.values()))
            values[k] = a[k]
            after = np.prod(list(values.values()))
            effects[k] += (after - before) / 6.0
    # The identity is exact to floating point precision when revenue = sessions*cvr*aov.
    assert np.isclose(sum(effects.values()), a['ga4_revenue'] - b['ga4_revenue'], atol=0.01)
    return effects


def _diagnosis(a, b, driver, gap):
    """Evidence-based investigation, not a causal attribution."""
    improving = gap >= 0
    spend = _pct_change(a['spend'], b['spend'])
    impressions = _pct_change(a['impressions'], b['impressions'])
    clicks = _pct_change(a['clicks'], b['clicks'])
    cpc = _pct_change(a['cpc'], b['cpc'])
    ctr = _pct_change(a['ctr'], b['ctr'])
    if driver == 'sessions':
        if np.isfinite(cpc) and cpc > 10 and np.isfinite(clicks) and clicks < -5:
            return ('Traffic declined as CPC increased; review auction pressure, targeting, '
                    'keyword/creative mix, and landing-page delivery. Validate in platform reports.')
        if np.isfinite(impressions) and impressions < -5:
            return ('Reach declined; check budgets, campaign eligibility, impression share, '
                    'audience saturation and delivery changes.')
        if np.isfinite(ctr) and ctr < -5:
            return ('Click-through weakened; review creative, targeting and placements before changing spend.')
        return ('Identify the campaigns and audiences behind stronger traffic; test whether the '
                'improvement can be replicated.' if improving else
                'Review campaign delivery, click-to-session tracking, landing pages and demand changes.')
    if driver == 'cvr':
        return ('Identify the products, landing pages and customer segments behind improved '
                'conversion; validate and test replication.' if improving else
                'Check checkout friction, site speed, inventory availability, promotion changes and '
                'device-level conversion before altering media budgets.')
    return ('Investigate which merchandise, baskets and price types drove higher order value; '
            'test whether the mix is sustainable.' if improving else
            'Review discounting, product mix, basket composition and high-value product availability.')


def _offsetting_signal(contributions, net_gap):
    """Detect opposing Shapley components that mask a material gross movement.

    Cancellation is the smaller of total positive and total negative components.
    The gross driver movement is the sum of absolute component contributions.
    Values are descriptive GA4 attributed revenue, not incremental sales.
    """
    if not contributions:
        return {'flag': False, 'cancellation': 0.0, 'gross': abs(net_gap),
                'opposing': [], 'ratio': 0.0}
    positives = sum(max(v, 0.0) for v in contributions.values())
    negatives = sum(max(-v, 0.0) for v in contributions.values())
    cancellation = min(positives, negatives)
    gross = positives + negatives
    # Flag substantial opposing contributions, not tiny rounding artifacts.
    flag = (cancellation >= 1000 and gross >= 5000 and
            cancellation / gross >= 0.20 and gross >= 1.5 * max(abs(net_gap), 1))
    opposing = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
    return {'flag': flag, 'cancellation': cancellation, 'gross': gross,
            'opposing': opposing, 'ratio': cancellation / gross if gross else 0.0}


def _offsetting_recommendation(a, b, contributions):
    """Investigate both sides of an offsetting pattern, without causal claims."""
    negatives = [k for k,v in contributions.items() if v < 0]
    positives = [k for k,v in contributions.items() if v > 0]
    labels = {'sessions':'traffic', 'cvr':'conversion', 'aov':'order value'}
    weak = ', '.join(labels[k] for k in negatives)
    strong = ', '.join(labels[k] for k in positives)
    cpc = _pct_change(a['cpc'], b['cpc'])
    clicks = _pct_change(a['clicks'], b['clicks'])
    delivery = (' Check auction costs and targeting: CPC rose while clicks declined.'
                if np.isfinite(cpc) and cpc > 5 and np.isfinite(clicks) and clicks < -5 else '')
    return (f'Investigate weaker {weak} while validating the improvement in {strong}. '
            'Segment by campaign, device, audience and landing page before reallocating budget.' + delivery)


def find_marketing_discoveries(df, days=7, minimum=0):
    """Rank the largest net changes OR masked offsetting movements.

    No minimum-dollar exclusion. Revenue movements are descriptive GA4
    attribution, not incremental marketing contribution.
    """
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
        # Linear TV has no digital attribution and is not eligible for GA4 revenue discoveries.
        if channel == 'Linear TV' or (a['ga4_revenue'] == 0 and b['ga4_revenue'] == 0):
            continue
        if abs(gap) < minimum:
            continue
        contributions = _contributions(a, b)
        if contributions is None:
            driver = 'unavailable'
            driver_amount = np.nan
        else:
            driver = max(contributions, key=lambda k: abs(contributions[k]))
            driver_amount = contributions[driver]
        offset = _offsetting_signal(contributions, gap)
        score = max(abs(gap), offset['gross'] if offset['flag'] else 0.0)
        rows.append({'channel': channel, 'subchannel': sub, 'gap': gap,
                     'offsetting': offset, 'ranking_score': score,
                     'current': a, 'prior': b, 'driver': driver,
                     'driver_amount': driver_amount, 'contributions': contributions,
                     'recommendation': (_offsetting_recommendation(a, b, contributions)
                                        if offset['flag'] else _diagnosis(a, b, driver, gap)) if contributions else
                     'Investigate transaction and revenue tracking before interpreting this change.'})
    return sorted(rows, key=lambda x: (x['ranking_score'], abs(x['gap'])), reverse=True)[:3]


def _fmt_delta(x):
    return f'{x:+.1f}%' if np.isfinite(x) else 'N/A'


def _money(x):
    return f'${abs(x)/1e6:,.2f}M' if abs(x)>=1e6 else f'${abs(x)/1e3:,.0f}K'


def render_marketing(data):
    st.markdown('<div class="ri-eyebrow">RETAILIQ AI / MARKETING INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title('Make every marketing dollar count.')
    st.caption('Synthetic store and e-commerce business with reconciled GA4 online attribution. No incremental ROI claims.')
    df = make_marketing(data)
    end = df.date.max()
    current = df[df.date.between(end-pd.Timedelta(days=6),end)]
    m = _metrics(current)
    commerce = generate_ecommerce(data)
    week = commerce.loc[commerce.date.between(end-pd.Timedelta(days=6),end)]
    store_sales = week.store_sales.sum()
    ecom_sales = week.ecom_sales.sum()
    total_sales = week.total_business_sales.sum()
    cols = st.columns(5)
    for col, (label, value, detail) in zip(cols, [
        ('Platform spend', _money(m['spend']), 'Last 7 days'),
        ('GA4 sessions', f'{m["sessions"]:,.0f}', 'Modeled digital visits'),
        ('GA4 attributed revenue', _money(m['ga4_revenue']), 'Non-incremental proxy'),
        ('E-commerce sales', _money(ecom_sales), 'Synthetic online business'),
        ('Total business sales', _money(total_sales), 'Stores + e-commerce'),
    ]):
        with col:
            st.markdown(f'<div class="ri-kpi"><div class="ri-kpi-label">{label}</div>'
                        f'<div class="ri-kpi-value">{value}</div>'
                        f'<div class="ri-kpi-bottom">{detail}</div></div>', unsafe_allow_html=True)

    st.caption(f'Store sales (existing): {_money(store_sales)} · Online sales: {_money(ecom_sales)} · GA4 coverage of online revenue: {m["ga4_revenue"]/ecom_sales:.1%}. GA4 revenue is a subset of online revenue, not additive.')
    st.markdown('<div class="ri-section">✦ &nbsp; Top Marketing AI Discoveries</div>', unsafe_allow_html=True)
    st.caption('Top three largest net revenue changes or significant offsetting drivers versus the preceding equal-length period. No minimum-dollar threshold. Descriptive, not incremental.')
    tabs = st.tabs(list(PERIODS))
    for tab, (label, days) in zip(tabs, PERIODS.items()):
        with tab:
            discoveries = find_marketing_discoveries(df, days, minimum=0)
            if not discoveries:
                st.info('Insufficient comparable history or no eligible GA4 revenue changes for this period.')
            for i, item in enumerate(discoveries, 1):
                positive = item['gap'] >= 0
                color = '#34D399' if positive else '#FB7185'
                offset = item['offsetting']
                status = ('OFFSETTING DRIVERS' if offset['flag'] else
                          ('GA4 REVENUE UP' if positive else 'GA4 REVENUE DOWN'))
                driver = item['driver']
                driver_label = {'sessions':'Traffic', 'cvr':'Conversion', 'aov':'Order value'}.get(driver, 'Unclassified')
                driver_amount = item['driver_amount']
                if np.isfinite(driver_amount):
                    if offset['flag']:
                        title = f'Opposing traffic, conversion or order-value movements mask changes in {item["subchannel"]}'
                    else:
                        title = f'{driver_label} is the largest contributor to the {item["channel"]} / {item["subchannel"]} revenue change'
                    component = (('+' if driver_amount >= 0 else '−') + _money(driver_amount))
                else:
                    title = f'{item["channel"]} / {item["subchannel"]}: investigate the GA4 revenue change'
                    component = 'N/A'
                a, b = item['current'], item['prior']
                evidence = (f'Spend {_fmt_delta(_pct_change(a["spend"],b["spend"]))} · '
                            f'Impressions {_fmt_delta(_pct_change(a["impressions"],b["impressions"]))} · '
                            f'Clicks {_fmt_delta(_pct_change(a["clicks"],b["clicks"]))} · '
                            f'CPC {_fmt_delta(_pct_change(a["cpc"],b["cpc"]))} · '
                            f'GA4 sessions {_fmt_delta(_pct_change(a["sessions"],b["sessions"]))} · '
                            f'Conversion {_fmt_delta(_pct_change(a["cvr"],b["cvr"]))}')
                contrib = item['contributions']
                if contrib:
                    parts = ' · '.join(f'{name}: {( "+" if value >= 0 else "−")}{_money(value)}'
                                       for name, value in [('Traffic',contrib['sessions']),
                                                           ('Conversion',contrib['cvr']),
                                                           ('Order value',contrib['aov'])])
                else:
                    parts = 'Driver decomposition unavailable due to missing/zero transaction values.'
                offset_line = (f'<div style="color:#facc15;margin-top:10px">'
                               f'Offsetting movements: gross driver change {escape(_money(offset["gross"]))} · '
                               f'cancellation {escape(_money(offset["cancellation"]))} · '
                               f'net change {("+" if positive else "−")}{escape(_money(item["gap"]))}'
                               f'</div>') if offset['flag'] else ''
                st.markdown(f'''<div style="background:#20172e;border:1px solid #49325f;border-left:4px solid {color};border-radius:13px;padding:22px;margin:14px 0;">
                    <div style="color:{color};font-weight:800;letter-spacing:.08em;font-size:13px">#{i} &nbsp; {status} &nbsp; · &nbsp; {days}-DAY WINDOW</div>
                    <div style="font-size:23px;font-weight:750;color:#f8fafc;margin-top:12px">{escape(title)}</div>
                    <div style="font-size:34px;font-weight:850;color:{color};margin:10px 0">{('+' if positive else '−')}{_money(item['gap'])}</div>
                    <div style="color:#cbd5e1">GA4 attributed-revenue change vs preceding period · Largest calculated component: {escape(driver_label)} {escape(component)}</div>
                    <div style="color:#c4b5fd;margin-top:10px">{escape(parts)}</div>
                    {offset_line}
                    <div style="color:#cbd5e1;margin-top:10px">{escape(evidence)}</div>
                    <div style="background:#342a43;border-left:3px solid #facc15;color:#fde68a;padding:12px;margin-top:14px;border-radius:6px"><b>What to investigate:</b> {escape(item['recommendation'])}</div>
                    <div style="color:#a6a1b7;font-size:12px;margin-top:10px">Exploratory · Ranked by gross movement for offsetting patterns, otherwise absolute net change · GA4 attributed revenue is not incremental revenue · No causal conclusion</div>
                    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="ri-section">Marketing investment by channel and subchannel</div>', unsafe_allow_html=True)
    st.caption('Last 7 days · Subchannel spend sums exactly to channel spend. Linear TV is offline awareness and has no GA4 click attribution.')
    sub_detail = (current.groupby(['channel','subchannel'], as_index=False)
                  .agg(spend=('spend','sum'), impressions=('impressions','sum'),
                       clicks=('clicks','sum'), sessions=('sessions','sum'),
                       ga4_revenue=('ga4_revenue','sum')))
    channel_detail = (sub_detail.groupby('channel', as_index=False)
                      .agg(spend=('spend','sum'), ga4_revenue=('ga4_revenue','sum')))
    actual_week_spend = float(current.spend.sum())
    target_week_spend = float(total_sales * TARGET_SPEND_SHARE)
    assert np.isclose(actual_week_spend, target_week_spend, atol=.01)
    assert np.isclose(sub_detail.spend.sum(), channel_detail.spend.sum(), atol=.01)
    st.metric('Marketing spend / total business sales', f'{actual_week_spend / total_sales:.1%}')
    with st.expander('Channel and subchannel spend reconciliation', expanded=True):
        st.dataframe(sub_detail.style.format({'spend':'${:,.0f}', 'impressions':'{:,.0f}',
                         'clicks':'{:,.0f}', 'sessions':'{:,.0f}', 'ga4_revenue':'${:,.0f}'}),
                     hide_index=True, width='stretch')
        st.caption(f'Subchannels ${actual_week_spend:,.0f} = channels ${channel_detail.spend.sum():,.0f} = 5% of total business sales ${target_week_spend:,.0f}.')
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
    st.markdown('<div class="ri-section">Platform + GA4 diagnostics</div>', unsafe_allow_html=True)
    selected_channel = st.selectbox('Channel', sorted(df.channel.unique()), key='marketing_diag_channel')
    subchoices = sorted(df.loc[df.channel == selected_channel, 'subchannel'].unique())
    selected_sub = st.selectbox('Subchannel', subchoices, key='marketing_diag_subchannel')
    view = df[(df.channel == selected_channel) & (df.subchannel == selected_sub)].copy()
    view['cpm'] = np.where(view.impressions > 0, 1000 * view.spend / view.impressions, np.nan)
    view['cpc'] = np.where(view.clicks > 0, view.spend / view.clicks, np.nan)
    view['ctr_pct'] = np.where(view.impressions > 0, 100 * view.clicks / view.impressions, np.nan)
    view['conversion_pct'] = np.where(view.sessions > 0, 100 * view.transactions / view.sessions, np.nan)
    view['revenue_per_session'] = np.where(view.sessions > 0, view.ga4_revenue / view.sessions, np.nan)
    diag_days = st.selectbox('Diagnostic window', list(PERIODS), index=1, key='marketing_diag_window')
    diag_end = view.date.max()
    view = view.loc[view.date >= diag_end - pd.Timedelta(days=PERIODS[diag_days]-1)]
    st.dataframe(view[['date','spend','impressions','clicks','cpm','cpc','ctr_pct',
                       'sessions','transactions','conversion_pct','revenue_per_session','ga4_revenue']].sort_values('date',ascending=False),
                 hide_index=True, width='stretch')
    if selected_channel == 'Linear TV':
        st.caption('Linear TV has offline awareness impressions and no modeled GA4 click or revenue attribution.')
    else:
        st.caption('Synthetic delivery and attributed GA4 signals; changes are diagnostic, not incremental media effects.')
    st.info('MMM and budget optimization are intentionally not shown yet: this dataset was constructed after sales generation and cannot validate causal media contribution.')
