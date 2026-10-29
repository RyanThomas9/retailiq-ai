"""Prepare separate store and e-commerce Meridian outcomes and validate simulated ROI.

Training runs outside Streamlit. Ground truth is for validation only.
"""
import numpy as np
import pandas as pd

CONTROLS = [
    'promotion', 'severe_weather', 'season_sin', 'season_cos', 'time_index',
    'black_friday', 'christmas_week', 'pre_christmas',
    'new_year_week', 'back_to_school',
]

MEDIA_COLUMNS = [
    'Affiliate- Conversion',
    'Display-Awareness',
    'Display-Consideration',
    'Display-Conversion',
    'Email- Conversion',
    'Linear TV- Awareness',
    'Paid Search- Conversion',
    'Paid Social- Awareness',
    'Paid Social- Consideration',
    'Paid Social- Conversion',
    'Video -Awareness',
    'Video -Consideration',
    'Video -Conversion',
]

OUTCOMES = {'stores': 'store_sales', 'ecommerce': 'ecom_sales'}


def prepare_frame(csv_path, outcome):
    if outcome not in OUTCOMES:
        raise ValueError(f'Unknown outcome: {outcome}')
    raw = pd.read_csv(csv_path)
    required = {'week_start_date', OUTCOMES[outcome], 'planned_media_budget'} | set(CONTROLS) | set(MEDIA_COLUMNS)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    raw['week_start_date'] = pd.to_datetime(raw['week_start_date'])
    raw = raw.sort_values('week_start_date').reset_index(drop=True)
    if raw.week_start_date.duplicated().any():
        raise ValueError('Duplicate weeks')
    if not (raw.week_start_date.diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError('Weeks must be consecutive')
    if len(raw) != 156:
        raise ValueError(f'Expected 156 weeks, got {len(raw)}')

    media = MEDIA_COLUMNS
    numeric = media + [OUTCOMES[outcome], 'planned_media_budget'] + CONTROLS
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors='raise')
    if raw[numeric].isna().any().any():
        raise ValueError('Missing input values')
    if (raw[media] < 0).any().any():
        raise ValueError('Negative spend')
    if (raw[OUTCOMES[outcome]] <= 0).any():
        raise ValueError('Non-positive revenue')
    if not np.allclose(raw[media].sum(axis=1), raw.planned_media_budget, atol=.01):
        raise ValueError('Subchannel spend does not reconcile to planned media budget')

    prepared = pd.DataFrame({
        'time': raw.week_start_date.dt.strftime('%Y-%m-%d'),
        'kpi': raw[OUTCOMES[outcome]].astype(float),
    })
    for i, name in enumerate(media):
        prepared[f'media_{i:02d}'] = raw[name].astype(float)
        prepared[f'spend_{i:02d}'] = raw[name].astype(float)
    for name in CONTROLS:
        prepared[name] = raw[name].astype(float)
    mapping = {f'media_{i:02d}': name for i, name in enumerate(media)}
    spend_mapping = {f'spend_{i:02d}': name for i, name in enumerate(media)}
    return prepared, mapping, spend_mapping, raw


def truth_roi(truth_csv, spend_csv, outcome, weekly_csv):
    """Calculate outcome-specific synthetic ROI, never used as model inputs/priors.

    weekly_csv is retained for compatibility with the existing trainer.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f'Unknown outcome: {outcome}')
    truth = pd.read_csv(truth_csv, parse_dates=['week_start_date'])
    spend = pd.read_csv(spend_csv, parse_dates=['week_start_date'])
    required_truth = {'week_start_date', 'subchannel', 'true_store_incremental_revenue', 'true_ecom_incremental_revenue'}
    required_spend = {'week_start_date', 'subchannel', 'spend'}
    if not required_truth.issubset(truth.columns):
        raise ValueError(f'Missing truth columns: {sorted(required_truth - set(truth.columns))}')
    if not required_spend.issubset(spend.columns):
        raise ValueError(f'Missing spend columns: {sorted(required_spend - set(spend.columns))}')
    if truth.duplicated(['week_start_date', 'subchannel']).any() or spend.duplicated(['week_start_date', 'subchannel']).any():
        raise ValueError('Duplicate week/subchannel in truth or spend')
    if set(map(tuple, truth[['week_start_date', 'subchannel']].to_numpy())) != set(map(tuple, spend[['week_start_date', 'subchannel']].to_numpy())):
        raise ValueError('Truth and spend do not cover identical weeks and subchannels')
    value_col = 'true_store_incremental_revenue' if outcome == 'stores' else 'true_ecom_incremental_revenue'
    revenue = truth.groupby('subchannel')[value_col].sum()
    costs = spend.groupby('subchannel')['spend'].sum()
    if (costs <= 0).any():
        raise ValueError('Non-positive subchannel spend')
    result = (revenue / costs).rename('true_roi').reset_index()
    if result.true_roi.isna().any():
        raise ValueError('Missing ROI values')
    return result


def save_metrics_table(summary_dataset, output_csv):
    df = summary_dataset.to_dataframe().reset_index()
    df.to_csv(output_csv, index=False)
    return df
