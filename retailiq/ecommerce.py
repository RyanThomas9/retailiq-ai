"""Deterministic synthetic online business; store sales are never modified."""
import numpy as np
import pandas as pd


def generate_ecommerce(store_data, seed=20261028):
    """Generate daily e-commerce sales on the same dates as the store ledger.

    Store activity is a correlated demand signal, not a causal media response.
    The generated online revenue is additive to store sales, never a duplicate.
    """
    daily = (store_data.groupby('date', as_index=False)
             .agg(store_sales=('sales', 'sum'), store_orders=('orders', 'sum'))
             .sort_values('date').reset_index(drop=True))
    rng = np.random.default_rng(seed)
    n = len(daily)
    t = np.arange(n)
    weekdays = pd.to_datetime(daily.date).dt.dayofweek.to_numpy()
    seasonal = 1 + .13 * np.sin(2*np.pi*t/365.25 + .8)
    demand = daily.store_sales.to_numpy() / max(float(daily.store_sales.median()), 1)
    online_orders = rng.poisson(np.maximum(30, 640 * seasonal * np.power(demand, .45)
                                             * np.where(weekdays >= 5, 1.12, 1.0)))
    aov = np.maximum(50, 165 * (1 + .07*np.sin(t/37)) * rng.lognormal(0, .045, n))
    daily['ecom_orders'] = online_orders.astype(int)
    daily['ecom_aov'] = aov
    daily['ecom_sales'] = daily.ecom_orders * daily.ecom_aov
    daily['total_business_sales'] = daily.store_sales + daily.ecom_sales
    return daily
