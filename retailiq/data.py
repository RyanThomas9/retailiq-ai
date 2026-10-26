import numpy as np
import pandas as pd


def generate(seed=42, stores=80, days=1000):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end='2026-10-17', periods=days, freq='D')
    meta = pd.DataFrame({
        'store': [f'S{n:03d}' for n in range(1, stores + 1)],
        'region': rng.choice(['North', 'South', 'East', 'West'], stores),
        'format': rng.choice(['Mall', 'Open Air', 'Street'], stores),
        'size': rng.choice(['Small', 'Medium', 'Large'], stores),
        'base': rng.uniform(4500, 14000, stores),
    })
    rows = []
    for r in meta.itertuples(index=False):
        n = len(dates)
        weekday = np.array([d.dayofweek for d in dates])
        seasonal = 1 + .15 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        weekend = np.where(weekday >= 5, 1.32, 1.0)
        traffic = rng.poisson(np.maximum(20, r.base / 85 * seasonal * weekend)).astype(float)
        conversion = np.clip(.26 + rng.normal(0, .025, n), .08, .50)
        discount = np.clip(.15 + .035 * np.sin(np.arange(n) / 33) + rng.normal(0, .015, n), 0, .45)
        # Inject discoverable, fictional operational weakness late in the period.
        affected = r.region == 'North' and r.format == 'Mall'
        if affected:
            conversion[-50:] *= .77
            discount[-50:] += .06
        orders = rng.binomial(traffic.astype(int), conversion)
        aov = np.maximum(35, r.base / 30 + rng.normal(0, 18, n))
        sales = orders * aov
        for i, dt in enumerate(dates):
            rows.append((dt, r.store, r.region, r.format, r.size, float(sales[i]),
                         float(traffic[i]), int(orders[i]), float(discount[i])))
    return pd.DataFrame(rows, columns=['date', 'store', 'region', 'format', 'size', 'sales', 'traffic', 'orders', 'discount'])
