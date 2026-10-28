"""Standalone, reproducible synthetic MMM training scenario.

No corporate data. Does not modify the application's existing store/e-commerce data.
Ground truth is intentionally separate from model inputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd

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
SPEND_RATE = .05


def _adstock(x, decay):
    """Normalized geometric adstock: steady input retains steady level."""
    out = np.zeros(len(x), dtype=float)
    for t, val in enumerate(x):
        out[t] = (1-decay)*val + (decay*out[t-1] if t else 0)
    return out


def generate(seed=20261028, n_weeks=156, last_week='2026-10-19'):
    if n_weeks < 104:
        raise ValueError('At least 104 weeks required for annual validation')
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp(last_week), periods=n_weeks, freq='W-MON')
    t = np.arange(n_weeks)
    yearly = np.sin(2*np.pi*t/52.18)
    winter = np.cos(2*np.pi*t/52.18)
    holiday = np.isin(dates.month, [11, 12]).astype(float)
    promotion = (rng.random(n_weeks) < .19).astype(float)
    severe_weather = (rng.random(n_weeks) < .07).astype(float)
    trend = 1 + .00055*t
    baseline = 4_450_000 * trend * (1 + .085*yearly + .045*winter + .16*holiday + .055*promotion - .045*severe_weather)
    baseline = np.maximum(baseline, 2_000_000)
    noise = rng.normal(0, .012, n_weeks) * baseline
    # Spend mix is driven by plans, calendar and random operational variation, not realized sales noise.
    keys = [(ch, sub) for ch, subs in CHANNELS.items() for sub in subs]
    base_mix = np.array([CHANNEL_WEIGHTS[ch]*CHANNELS[ch][sub] for ch, sub in keys])
    raw = np.exp(rng.normal(0, .095, (n_weeks, len(keys)))) * base_mix
    raw *= (1 + .09*holiday[:, None]*np.array([1 if k[0] in ('Paid Search','Paid Social') else 0 for k in keys]))
    shares = raw / raw.sum(axis=1, keepdims=True)
    # Hidden true parameters differ by objective, with longer adstock for awareness.
    decays = np.array([.66 if 'Linear TV' in sub else (.55 if 'Awareness' in sub else (.35 if 'Consideration' in sub else .12)) for _,sub in keys])
    # Hill response in dollars per week, parameters intentionally retained only in truth table.
    max_lifts = np.array([165000 if 'Linear TV' in sub else (115000 if 'Awareness' in sub else (90000 if 'Consideration' in sub else 140000)) for _,sub in keys], dtype=float)
    half_sats = np.array([45000 if 'Linear TV' in sub else (32000 if 'Awareness' in sub else 20000) for _,sub in keys], dtype=float)
    hill_exponent = 1.35
    # Fixed-point iteration: weekly spend equals 5% of realized total sales, including media lift.
    sales = baseline + noise
    for _ in range(40):
        spend = SPEND_RATE * sales[:, None] * shares
        ad = np.column_stack([_adstock(spend[:, j], decays[j]) for j in range(len(keys))])
        lift = max_lifts * ad**hill_exponent / (half_sats**hill_exponent + ad**hill_exponent)
        new_sales = baseline + noise + lift.sum(axis=1)
        if np.max(np.abs(new_sales-sales)) < .00001:
            sales = new_sales
            break
        sales = new_sales
    spend = SPEND_RATE * sales[:, None] * shares
    ad = np.column_stack([_adstock(spend[:, j], decays[j]) for j in range(len(keys))])
    lift = max_lifts * ad**hill_exponent / (half_sats**hill_exponent + ad**hill_exponent)
    media_total = lift.sum(axis=1)
    # Split total sales across store and e-commerce with variable but reconciled share.
    ecom_share = np.clip(.145 + .018*yearly + .012*holiday + rng.normal(0,.005,n_weeks), .10, .22)
    store_sales = sales*(1-ecom_share)
    ecom_sales = sales*ecom_share
    weekly = pd.DataFrame({'week_start_date':dates, 'store_sales':store_sales,
                           'ecom_sales':ecom_sales, 'total_sales':sales,
                           'baseline_demand':baseline, 'promotion':promotion.astype(int),
                           'holiday':holiday.astype(int), 'severe_weather':severe_weather.astype(int),
                           'season_sin':yearly, 'season_cos':winter, 'time_index':t})
    long = pd.DataFrame({'week_start_date':np.repeat(dates,len(keys)),
                         'channel':[ch for _ in dates for ch,_ in keys],
                         'subchannel':[sub for _ in dates for _,sub in keys],
                         'spend':spend.ravel()})
    # Modeling data excludes baseline and truth parameters to avoid leakage.
    wide = long.pivot(index='week_start_date', columns='subchannel', values='spend').reset_index()
    model = weekly.drop(columns=['baseline_demand']).merge(wide, on='week_start_date',validate='one_to_one')
    truth = pd.DataFrame({'week_start_date':np.repeat(dates,len(keys)),
                          'subchannel':[sub for _ in dates for _,sub in keys],
                          'true_incremental_revenue':lift.ravel(),
                          'true_adstock':ad.ravel(),
                          'true_decay':np.tile(decays,n_weeks),
                          'true_half_saturation':np.tile(half_sats,n_weeks),
                          'true_max_lift':np.tile(max_lifts,n_weeks),
                          'true_hill_exponent':hill_exponent})
    baseline_truth = weekly[['week_start_date','baseline_demand']].copy()
    baseline_truth['unobserved_noise'] = noise
    return model, long, truth, baseline_truth


def validate(model, long, truth, baseline_truth):
    by_week = long.groupby('week_start_date').spend.sum().sort_index()
    m = model.set_index('week_start_date').sort_index()
    assert len(m) == 156, f'Expected 156 weeks, got {len(m)}'
    assert len(long) == len(m)*13
    assert long.groupby('week_start_date').subchannel.nunique().eq(13).all()
    assert np.allclose(m.store_sales+m.ecom_sales,m.total_sales,atol=.01)
    assert np.allclose(by_week.to_numpy(), .05*m.total_sales.to_numpy(), atol=.01)
    assert (m.total_sales > 0).all() and (long.spend > 0).all()
    effects = truth.groupby('week_start_date').true_incremental_revenue.sum().sort_index()
    b = baseline_truth.set_index('week_start_date').sort_index()
    assert np.allclose(b.baseline_demand+b.unobserved_noise+effects,m.total_sales,atol=.01)
    assert np.isfinite(model.select_dtypes(include='number').to_numpy()).all()
    return {'weeks':len(m), 'subchannels':13, 'weekly_spend_ratio_min':float((by_week/m.total_sales).min()),
            'weekly_spend_ratio_max':float((by_week/m.total_sales).max()),
            'mean_weekly_sales':float(m.total_sales.mean()), 'mean_weekly_spend':float(by_week.mean()),
            'mean_true_incremental_revenue':float(effects.mean())}


def export(directory, seed=20261028):
    out = Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    model,long,truth,baseline = generate(seed=seed)
    stats = validate(model,long,truth,baseline)
    model.to_csv(out/'meridian_weekly_inputs.csv',index=False)
    long.to_csv(out/'marketing_spend_long.csv',index=False)
    truth.to_csv(out/'GROUND_TRUTH_DO_NOT_TRAIN.csv',index=False)
    baseline.to_csv(out/'GROUND_TRUTH_BASELINE_DO_NOT_TRAIN.csv',index=False)
    return stats

if __name__ == '__main__':
    import argparse, json
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='output/mmm')
    p.add_argument('--seed',type=int,default=20261028)
    a=p.parse_args()
    print(json.dumps(export(a.output,a.seed),indent=2))
