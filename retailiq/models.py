import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


def build_features(df):
    d = df.sort_values(['store', 'date']).copy()
    g = d.groupby('store', sort=False)
    for window in (7, 28):
        d[f'lag_avg_{window}'] = g['sales'].transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    d['lag_7'] = g['sales'].shift(7)
    d['day_of_week'] = d.date.dt.dayofweek
    d['month'] = d.date.dt.month
    d['day_of_year'] = d.date.dt.dayofyear
    for col in ('store', 'region', 'format', 'size'):
        d[col] = d[col].astype('category')
    return d


FEATURES = ['store', 'region', 'format', 'size', 'lag_avg_7', 'lag_avg_28', 'lag_7', 'day_of_week', 'month', 'day_of_year']


def forecast(df, test_days=90):
    d = build_features(df)
    cutoff = d.date.max() - pd.Timedelta(days=test_days - 1)
    train = d[(d.date < cutoff) & d.lag_avg_28.notna()].copy()
    test = d[(d.date >= cutoff) & d.lag_avg_28.notna()].copy()
    model = LGBMRegressor(n_estimators=140, learning_rate=.055, num_leaves=15,
                          min_child_samples=80, verbosity=-1, random_state=42, n_jobs=2)
    model.fit(train[FEATURES], train.sales)
    test['expected'] = np.maximum(0, model.predict(test[FEATURES]))
    wape = 100 * np.abs(test.sales - test.expected).sum() / max(test.sales.abs().sum(), 1)
    bias = 100 * (test.expected - test.sales).sum() / max(test.sales.abs().sum(), 1)
    # Score full feature-ready history for long-window descriptive comparisons.
    # These are retrospective fitted values before cutoff, not out-of-sample forecasts.
    scored = d[d.lag_avg_28.notna()].copy()
    scored['expected'] = np.maximum(0, model.predict(scored[FEATURES]))
    return scored, model, {'wape': wape, 'bias': bias, 'cutoff': cutoff.date(), 'train_rows': len(train)}
