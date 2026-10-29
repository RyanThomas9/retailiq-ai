"""Standalone, reproducible synthetic MMM training scenario.

No corporate data. Does not modify the application's existing store/e-commerce data.
Ground truth is intentionally separate from model inputs.
"""
from pathlib import Path
from datetime import date, timedelta
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


def _retail_calendar(dates):
    """Calendar events at Monday-start weekly grain, independently of media spend."""
    def thanksgiving(year):
        nov1 = date(year, 11, 1)
        first_thursday = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
        return first_thursday + timedelta(weeks=3)

    flags = {k: np.zeros(len(dates), dtype=int) for k in (
        'black_friday', 'cyber_monday', 'christmas_week', 'pre_christmas',
        'thanksgiving_week', 'new_year_week', 'back_to_school')}
    for i, stamp in enumerate(dates):
        start = stamp.date()
        end = start + timedelta(days=6)
        def within(d): return start <= d <= end
        for year in (start.year, end.year):
            thanks = thanksgiving(year)
            if within(thanks): flags['thanksgiving_week'][i] = 1
            if within(thanks + timedelta(days=1)): flags['black_friday'][i] = 1
            if within(thanks + timedelta(days=4)): flags['cyber_monday'][i] = 1
            if within(date(year,12,25)): flags['christmas_week'][i] = 1
            if within(date(year,1,1)): flags['new_year_week'][i] = 1
        # Two weeks preceding the week containing Christmas, avoiding double-counting.
        christmas_week_start = date(start.year,12,25) - timedelta(days=date(start.year,12,25).weekday())
        if start in (christmas_week_start-timedelta(weeks=1),christmas_week_start-timedelta(weeks=2)):
            flags['pre_christmas'][i] = 1
        if start.month in (7,8): flags['back_to_school'][i] = 1
    return flags


def generate(seed=20261028, n_weeks=156, last_week='2026-10-19'):
    if n_weeks < 104:
        raise ValueError('At least 104 weeks required for annual validation')
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp(last_week), periods=n_weeks, freq='W-MON')
    t = np.arange(n_weeks)
    yearly = np.sin(2*np.pi*t/52.18)
    winter = np.cos(2*np.pi*t/52.18)
    events = _retail_calendar(dates)
    holiday = np.isin(dates.month, [11, 12]).astype(float)
    promotion = (rng.random(n_weeks) < .19).astype(float)
    severe_weather = (rng.random(n_weeks) < .07).astype(float)
    trend = 1 + .00055*t
    holiday_lift = (.09*holiday + .22*events['black_friday'] + .09*events['cyber_monday']
                    + .13*events['pre_christmas'] + .07*events['christmas_week']
                    - .035*events['new_year_week'] + .045*events['back_to_school'])
    baseline = 4_450_000 * trend * (1 + .085*yearly + .045*winter + holiday_lift
                                     + .055*promotion - .045*severe_weather)
    baseline = np.maximum(baseline, 2_000_000)
    # Budget is set from demand forecasts available before actual sales occur.
    # A planning forecast has uncertainty, preventing perfect spend/sales coupling.
    planning_error = rng.normal(0, .028, n_weeks)
    planned_demand = baseline * (1 + planning_error)
    # Plan around 5% of anticipated total sales, including an expected media contribution.
    expected_media_lift = 850_000
    weekly_budget = SPEND_RATE * (planned_demand + expected_media_lift)
    weekly_budget *= rng.lognormal(-.5*.045**2, .045, n_weeks)
    # Spend has no dependence on the current week's realized sales or sales noise.
    keys = [(ch, sub) for ch, subs in CHANNELS.items() for sub in subs]
    base_mix = np.array([CHANNEL_WEIGHTS[ch]*CHANNELS[ch][sub] for ch, sub in keys])
    raw = np.exp(rng.normal(0, .095, (n_weeks, len(keys)))) * base_mix
    raw *= (1 + (.09*holiday + .20*events['black_friday'] + .12*events['cyber_monday'])[:, None]
               * np.array([1 if k[0] in ('Paid Search','Paid Social') else 0 for k in keys]))
    shares = raw / raw.sum(axis=1, keepdims=True)
    # Separate, hidden outcome response parameters. Media inputs remain shared.
    # Weekly spend is planned before either outcome is realized.
    decays = np.array([.66 if 'Linear TV' in sub else (.55 if 'Awareness' in sub else (.35 if 'Consideration' in sub else .12)) for _, sub in keys])
    half_sats = np.array([45000 if 'Linear TV' in sub else (32000 if 'Awareness' in sub else 20000) for _, sub in keys], dtype=float)
    hill_exponent = 1.35
    spend = weekly_budget[:, None] * shares
    ad = np.column_stack([_adstock(spend[:, j], decays[j]) for j in range(len(keys))])
    response = ad**hill_exponent / (half_sats**hill_exponent + ad**hill_exponent)

    # Channel-specific store vs online strengths, NOT passed to Meridian.
    # Positive but heterogeneous responses make the two outcomes distinguishable.
    store_max = np.array([110000 if 'Linear TV' in sub else
                          70000 if 'Awareness' in sub else
                          45000 if 'Consideration' in sub else
                          (40000 if 'Email' in sub else 85000) for _, sub in keys], dtype=float)
    ecom_max = np.array([18000 if 'Linear TV' in sub else
                         26000 if 'Awareness' in sub else
                         23000 if 'Consideration' in sub else
                         (55000 if 'Email' in sub else 65000) for _, sub in keys], dtype=float)
    store_lift = response * store_max
    ecom_lift = response * ecom_max
    media_total = (store_lift + ecom_lift).sum(axis=1)

    # Demand and nonmedia effects are split BEFORE adding channel effects.
    # This avoids the old proportional allocation of one common marketing effect.
    ecom_baseline_share = np.clip(.145 + .018*yearly + .012*holiday + rng.normal(0,.005,n_weeks), .10, .22)
    store_baseline = baseline * (1 - ecom_baseline_share)
    ecom_baseline = baseline * ecom_baseline_share
    store_noise = rng.normal(0, .010, n_weeks) * store_baseline
    ecom_noise = rng.normal(0, .016, n_weeks) * ecom_baseline
    store_sales = store_baseline + store_noise + store_lift.sum(axis=1)
    ecom_sales = ecom_baseline + ecom_noise + ecom_lift.sum(axis=1)
    sales = store_sales + ecom_sales
    weekly = pd.DataFrame({'week_start_date':dates, 'store_sales':store_sales,
                           'ecom_sales':ecom_sales, 'total_sales':sales,
                           'baseline_demand':baseline, 'promotion':promotion.astype(int),
                           'holiday':holiday.astype(int), 'severe_weather':severe_weather.astype(int),
                           'season_sin':yearly, 'season_cos':winter, 'time_index':t,
                           'planned_demand':planned_demand, 'planned_media_budget':weekly_budget,
                           **events})
    long = pd.DataFrame({'week_start_date':np.repeat(dates,len(keys)),
                         'channel':[ch for _ in dates for ch,_ in keys],
                         'subchannel':[sub for _ in dates for _,sub in keys],
                         'spend':spend.ravel()})
    # Modeling data excludes baseline and truth parameters to avoid leakage.
    wide = long.pivot(index='week_start_date', columns='subchannel', values='spend').reset_index()
    model = weekly.drop(columns=['baseline_demand', 'planned_demand']).merge(wide, on='week_start_date',validate='one_to_one')
    truth = pd.DataFrame({'week_start_date':np.repeat(dates,len(keys)),
                          'subchannel':[sub for _ in dates for _,sub in keys],
                          'true_store_incremental_revenue':store_lift.ravel(),
                          'true_ecom_incremental_revenue':ecom_lift.ravel(),
                          'true_incremental_revenue':(store_lift+ecom_lift).ravel(),
                          'true_adstock':ad.ravel(),
                          'true_decay':np.tile(decays,n_weeks),
                          'true_half_saturation':np.tile(half_sats,n_weeks),
                          'true_store_max_lift':np.tile(store_max,n_weeks),
                          'true_ecom_max_lift':np.tile(ecom_max,n_weeks),
                          'true_hill_exponent':hill_exponent})
    baseline_truth = weekly[['week_start_date','baseline_demand']].copy()
    baseline_truth['store_baseline_demand'] = store_baseline
    baseline_truth['ecom_baseline_demand'] = ecom_baseline
    baseline_truth['store_unobserved_noise'] = store_noise
    baseline_truth['ecom_unobserved_noise'] = ecom_noise
    baseline_truth['unobserved_noise'] = store_noise + ecom_noise
    return model, long, truth, baseline_truth


def validate(model, long, truth, baseline_truth):
    by_week = long.groupby('week_start_date').spend.sum().sort_index()
    m = model.set_index('week_start_date').sort_index()
    assert len(m) >= 104, f'Expected at least 104 weeks, got {len(m)}'
    assert len(long) == len(m)*13
    assert long.groupby('week_start_date').subchannel.nunique().eq(13).all()
    assert np.allclose(m.store_sales+m.ecom_sales,m.total_sales,atol=.01)
    assert np.allclose(by_week.to_numpy(), m.planned_media_budget.to_numpy(), atol=.01)
    spend_ratio = by_week / m.total_sales
    assert spend_ratio.between(.035, .065).all(), 'Spend should stay near 5% of realized sales'
    assert (m.total_sales > 0).all() and (long.spend > 0).all()
    effects = truth.groupby('week_start_date').true_incremental_revenue.sum().sort_index()
    b = baseline_truth.set_index('week_start_date').sort_index()
    assert np.allclose(b.baseline_demand+b.unobserved_noise+effects,m.total_sales,atol=.01)
    store_effects = truth.groupby('week_start_date').true_store_incremental_revenue.sum().sort_index()
    ecom_effects = truth.groupby('week_start_date').true_ecom_incremental_revenue.sum().sort_index()
    assert np.allclose(b.store_baseline_demand+b.store_unobserved_noise+store_effects,m.store_sales,atol=.01)
    assert np.allclose(b.ecom_baseline_demand+b.ecom_unobserved_noise+ecom_effects,m.ecom_sales,atol=.01)
    assert np.isfinite(model.select_dtypes(include='number').to_numpy()).all()
    assert (m.black_friday.sum() >= 2 and m.christmas_week.sum() >= 2)
    assert (m.cyber_monday.sum() >= 2 and m.pre_christmas.sum() >= 4)
    assert np.allclose(m.planned_media_budget.to_numpy(), by_week.to_numpy())
    return {'weeks':len(m), 'subchannels':13, 'weekly_spend_ratio_min':float(spend_ratio.min()),
            'weekly_spend_ratio_max':float(spend_ratio.max()),
            'weekly_spend_ratio_mean':float(spend_ratio.mean()),
            'black_friday_weeks':int(m.black_friday.sum()), 'christmas_weeks':int(m.christmas_week.sum()),
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
    reconciliation = baseline.merge(model[['week_start_date','store_sales','ecom_sales','total_sales']], on='week_start_date')
    totals = truth.groupby('week_start_date',as_index=False)[['true_store_incremental_revenue','true_ecom_incremental_revenue','true_incremental_revenue']].sum()
    reconciliation = reconciliation.merge(totals,on='week_start_date')
    for prefix in ('store','ecom'):
        reconciliation[f'{prefix}_reconciled_sales'] = (reconciliation[f'{prefix}_baseline_demand'] +
            reconciliation[f'{prefix}_unobserved_noise'] + reconciliation[f'true_{prefix}_incremental_revenue'])
        reconciliation[f'{prefix}_reconciliation_error'] = reconciliation[f'{prefix}_sales'] - reconciliation[f'{prefix}_reconciled_sales']
    reconciliation['total_reconciliation_error'] = (reconciliation.total_sales -
        (reconciliation.baseline_demand + reconciliation.unobserved_noise + reconciliation.true_incremental_revenue))
    assert reconciliation[['store_reconciliation_error','ecom_reconciliation_error','total_reconciliation_error']].abs().to_numpy().max() < .01
    reconciliation.to_csv(out/'GROUND_TRUTH_WEEKLY_RECONCILIATION_DO_NOT_TRAIN.csv',index=False)
    return stats

if __name__ == '__main__':
    import argparse, json
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='output/mmm')
    p.add_argument('--seed',type=int,default=20261028)
    a=p.parse_args()
    print(json.dumps(export(a.output,a.seed),indent=2))
