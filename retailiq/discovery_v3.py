"""Exploratory, calendar-aligned peer-growth discovery engine.

Uses raw synthetic sales for 7/28/91/364-day windows. Does NOT use
in-sample LightGBM predictions as out-of-sample forecasts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

WINDOWS = {'Yesterday': 1, 'Last 7 Days': 7, 'Last Month': 28, 'Last 6 Months': 182, 'Last Year': 364}
GROUPS = [('region', 'format'), ('region', 'size'), ('format', 'size')]


def _bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), np.clip(p, 0, 1), 1.)
    n = len(p)
    if not n:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(adjusted, 1)
    return result


def discover_v3(data: pd.DataFrame, min_stores: int = 5, max_results_per_period: int = 30) -> pd.DataFrame:
    """Discover group-vs-peer YoY trends, FDR-adjusted over all tested groups/windows.

    A peer-rate dollar gap is a *descriptive counterfactual*, not causal lift.
    Significance is exploratory: store-level Welch test, not randomized evidence.
    """
    required = {'date', 'store', 'region', 'format', 'size', 'sales', 'traffic', 'orders'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    d = data[list(required)].copy()
    d['date'] = pd.to_datetime(d['date'])
    d['sales'] = pd.to_numeric(d['sales'], errors='coerce')
    d['traffic'] = pd.to_numeric(d['traffic'], errors='coerce')
    d['orders'] = pd.to_numeric(d['orders'], errors='coerce')
    d = d.dropna(subset=['date', 'sales', 'store'])
    end = d.date.max()
    out = []
    for label, days in WINDOWS.items():
        start = end - pd.Timedelta(days=days-1)
        prior_start = start - pd.Timedelta(days=364)
        prior_end = end - pd.Timedelta(days=364)
        current = d[d.date.between(start, end)]
        previous = d[d.date.between(prior_start, prior_end)]
        if current.empty or previous.empty:
            continue
        # Keep only stores with a complete comparable window; no invented zero history.
        cur = current.groupby('store').agg(ty=('sales','sum'), ty_days=('date','nunique'), ty_traffic=('traffic','sum'), ty_orders=('orders','sum'))
        ly = previous.groupby('store').agg(ly=('sales','sum'), ly_days=('date','nunique'), ly_traffic=('traffic','sum'), ly_orders=('orders','sum'))
        sums = cur.join(ly, how='inner')
        sums = sums[(sums.ty_days == days) & (sums.ly_days == days) & (sums.ly > 0)].copy()
        if len(sums) < 2 * min_stores:
            continue
        meta = current.sort_values('date').drop_duplicates('store').set_index('store')
        sums = sums.join(meta[['region', 'format', 'size']])
        sums['growth'] = sums.ty / sums.ly - 1
        weekly_comparisons = []
        if days >= 28:
            # Compute each comparable weekly store vector once per window.
            for offset in range(days // 7):
                we = end - pd.Timedelta(days=offset*7)
                ws = we - pd.Timedelta(days=6)
                a = d[d.date.between(ws, we)].groupby('store').sales.sum()
                b = d[d.date.between(ws-pd.Timedelta(days=364), we-pd.Timedelta(days=364))].groupby('store').sales.sum()
                weekly_comparisons.append((a, b))
        for keys in GROUPS:
            for vals, group in sums.groupby(list(keys), observed=True):
                peers = sums.drop(group.index)
                if len(group) < min_stores or len(peers) < min_stores:
                    continue
                peer_ly = peers.ly.sum()
                if peer_ly <= 0:
                    continue
                group_growth = group.ty.sum()/group.ly.sum()-1
                peer_growth = peers.ty.sum()/peer_ly-1
                delta_pp = 100 * (group_growth-peer_growth)
                impact = group.ly.sum() * (group_growth-peer_growth)
                if abs(delta_pp) < 1.0:
                    continue
                test = ttest_ind(group.growth, peers.growth, equal_var=False, nan_policy='omit')
                pval = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0
                breadth = 100 * (np.sign(group.growth - peer_growth) == np.sign(delta_pp)).mean()
                # Weekly persistence based on full 7-day aligned weeks in the current period.
                week_signs = []
                if days >= 28:
                    for a, b in weekly_comparisons:
                        g_ids, p_ids = group.index, peers.index
                        if not set(g_ids).issubset(a.index) or not set(g_ids).issubset(b.index) or not set(p_ids).issubset(a.index) or not set(p_ids).issubset(b.index):
                            continue
                        g_prev, p_prev = b.reindex(g_ids).sum(), b.reindex(p_ids).sum()
                        if g_prev > 0 and p_prev > 0:
                            weekly_gap = a.reindex(g_ids).sum()/g_prev - a.reindex(p_ids).sum()/p_prev
                            week_signs.append(float(weekly_gap))
                persistence = (100 * np.mean(np.sign(week_signs)==np.sign(delta_pp))) if week_signs else np.nan
                trend = 'Insufficient trend history'
                if len(week_signs) >= 4:
                    chronological = np.asarray(week_signs[::-1])
                    slope = np.polyfit(np.arange(len(chronological)), chronological, 1)[0]
                    if slope * np.sign(delta_pp) > .001:
                        trend = 'Strengthening'
                    elif slope * np.sign(delta_pp) < -.001:
                        trend = 'Weakening'
                    else:
                        trend = 'Persistent'
                # Exact, order-independent additive attribution of the peer-rate gap.
                # Sales = traffic * conversion * AOV; relative log changes add.
                # Distribute descriptive dollar gap in proportion to log growth differences.
                def _factors(frame):
                    t0, t1 = frame.ly_traffic.sum(), frame.ty_traffic.sum()
                    o0, o1 = frame.ly_orders.sum(), frame.ty_orders.sum()
                    s0, s1 = frame.ly.sum(), frame.ty.sum()
                    if min(t0,t1,o0,o1,s0,s1) <= 0: return None
                    return np.array([t1/t0, (o1/t1)/(o0/t0), (s1/o1)/(s0/o0)])
                gf, pf = _factors(group), _factors(peers)
                components = np.array([np.nan]*3)
                if gf is not None and pf is not None:
                    logdiff = np.log(gf/pf)
                    total_log = logdiff.sum()
                    if abs(total_log) > 1e-10:
                        components = impact * logdiff / total_log
                if np.isfinite(components).all():
                    driver_names = ['Traffic', 'Conversion', 'Order size']
                    lead = int(np.argmax(np.abs(components)))
                    lead_driver = driver_names[lead]
                    lead_impact = float(components[lead])
                else:
                    lead_driver, lead_impact = 'Not available', np.nan
                out.append(dict(group=' / '.join(map(str, vals)), attribute=' + '.join(keys),
                                stores=len(group), period=label, days=days,
                                relative_dollars=float(impact), relative_pct=float(delta_pp),
                                group_yoy=float(group_growth*100), peer_yoy=float(peer_growth*100),
                                breadth=float(breadth), persistence_pct=float(persistence),
                                trend=trend, p_value=pval, tested_stores=len(sums),
                                traffic_dollars=float(components[0]), conversion_dollars=float(components[1]),
                                order_size_dollars=float(components[2]), lead_driver=lead_driver,
                                lead_impact=lead_impact,
                                end_date=end.date()))
    if not out:
        return pd.DataFrame(columns=['group','attribute','stores','period','days','relative_dollars','relative_pct','group_yoy','peer_yoy','breadth','persistence_pct','trend','p_value','q_value','confidence','priority','traffic_dollars','conversion_dollars','order_size_dollars','lead_driver','lead_impact'])
    frame = pd.DataFrame(out)
    frame['q_value'] = _bh_adjust(frame.p_value)
    frame['confidence'] = np.where(frame.q_value < .05, 'FDR significant', 'Exploratory')
    # Executive priority is strictly absolute descriptive financial deviation.
    # Deduplicate identical groups WITHIN each period, never across periods.
    frame['priority'] = frame.relative_dollars.abs()
    frame = (frame.sort_values(['period','priority'], ascending=[True,False])
                  .drop_duplicates(['period','attribute','group'])
                  .groupby('period', group_keys=False).head(max_results_per_period))
    return frame.reset_index(drop=True)
