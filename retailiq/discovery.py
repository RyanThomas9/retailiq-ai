import numpy as np
import pandas as pd


def discover(scored, min_stores=5):
    d = scored.copy()
    d['error'] = d.sales - d.expected
    daily = d.groupby('date').agg(a=('sales', 'sum'), e=('expected', 'sum'))
    daily['brand_rate'] = daily.a / daily.e.replace(0, np.nan) - 1
    d = d.join(daily['brand_rate'], on='date')
    d['relative_error'] = d.error - d.expected * d.brand_rate
    recent = d[d.date >= d.date.max() - pd.Timedelta(days=27)]
    findings = []
    # Include two-attribute interactions, rather than only obvious region totals.
    for keys in [('region', 'format'), ('region', 'size'), ('format', 'size')]:
        for values, group in recent.groupby(list(keys), observed=True):
            if group.store.nunique() < min_stores or group.expected.sum() <= 0:
                continue
            dollars = group.relative_error.sum()
            pct = 100 * dollars / group.expected.sum()
            if abs(pct) < 2:
                continue
            store_err = group.groupby('store').relative_error.sum()
            breadth = 100 * (np.sign(store_err) == np.sign(dollars)).mean()
            label = ' / '.join(str(x) for x in values)
            findings.append({'group': label, 'attribute': ' + '.join(keys), 'stores': group.store.nunique(),
                             'relative_dollars': dollars, 'relative_pct': pct, 'breadth': breadth,
                             'priority': abs(dollars) * (breadth / 100)})
    return pd.DataFrame(findings).sort_values('priority', ascending=False).head(10) if findings else pd.DataFrame()
