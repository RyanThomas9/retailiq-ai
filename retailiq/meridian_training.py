"""Shared preparation and result exports for separate national Meridian outcomes.

No training is executed on Streamlit. Run the notebook in Colab.
"""
from pathlib import Path
import json
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
EXCLUDE = {
    'week_start_date',
    'store_sales',
    'ecom_sales',
    'total_sales',
    'planned_media_budget',
    'holiday',
    'thanksgiving_week',
    'cyber_monday',
} | set(CONTROLS)


def prepare_frame(csv_path, outcome):
    if outcome not in OUTCOMES:
        raise ValueError(f'Unknown outcome {outcome}')

    raw = pd.read_csv(csv_path)

    required = (
        {'week_start_date', OUTCOMES[outcome]}
        | set(CONTROLS)
        | set(MEDIA_COLUMNS)
    )
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    raw['week_start_date'] = pd.to_datetime(raw['week_start_date'])
    raw = raw.sort_values('week_start_date').reset_index(drop=True)

    if raw.week_start_date.duplicated().any():
        raise ValueError('Duplicate weeks')

    if not (
        raw.week_start_date.diff().dropna()
        == pd.Timedelta(days=7)
    ).all():
        raise ValueError('Weeks must be consecutive')

    # Use only the 13 explicitly defined marketing subchannels.
    media = MEDIA_COLUMNS

    if len(media) != 13:
        raise ValueError(
            f'Expected 13 subchannels, found {len(media)}'
        )

    # Validate numeric marketing inputs.
    for name in media:
        raw[name] = pd.to_numeric(raw[name], errors='raise')

    if raw[media + [OUTCOMES[outcome]] + CONTROLS].isna().any().any():
        raise ValueError('Missing input values')

    if (raw[media] < 0).any().any():
        raise ValueError('Negative spend')

    if (raw[OUTCOMES[outcome]] <= 0).any():
        raise ValueError('Non-positive revenue')

    if len(raw) != 156:
        raise ValueError(f'Expected 156 weeks, got {len(raw)}')

    prepared = pd.DataFrame({
        'time': raw.week_start_date.dt.strftime('%Y-%m-%d'),
        'kpi': raw[OUTCOMES[outcome]]
    })

    for i, name in enumerate(media):
        prepared[f'media_{i:02d}'] = raw[name].astype(float)
        prepared[f'spend_{i:02d}'] = raw[name].astype(float)

    for name in CONTROLS:
        prepared[name] = raw[name].astype(float)

    mapping = {
        f'media_{i:02d}': name
        for i, name in enumerate(media)
    }

    spend_mapping = {
        f'spend_{i:02d}': name
        for i, name in enumerate(media)
    }

    assert np.allclose(
        raw[media].sum(axis=1),
        raw.planned_media_budget,
        atol=.01
    )

    return prepared, mapping, spend_mapping, raw


def truth_roi(truth_csv, spend_csv, outcome, weekly_csv):
    """Diagnostic only; NEVER used to construct Meridian input or priors.

    Existing generator allocates a single national media effect to outcomes
    using each week's e-commerce share; it does NOT simulate separate
    channel-specific store and e-commerce response curves.
    """
    truth = pd.read_csv(truth_csv, parse_dates=['week_start_date'])
    spend = pd.read_csv(spend_csv, parse_dates=['week_start_date'])
    weekly = pd.read_csv(weekly_csv, parse_dates=['week_start_date'])
    weekly['share'] = weekly.ecom_sales / weekly.total_sales
    if outcome == 'stores':
        weekly['share'] = 1 - weekly['share']
    t = truth.merge(weekly[['week_start_date','share']], on='week_start_date', validate='many_to_one')
    t['outcome_incremental'] = t.true_incremental_revenue * t.share
    numerator = t.groupby('subchannel').outcome_incremental.sum()
    denominator = spend.groupby('subchannel').spend.sum()
    return (numerator / denominator).rename('true_roi').reset_index()


def save_metrics_table(summary_dataset, output_csv):
    """Export Meridian Analyzer.summary_metrics xarray result without guessing values."""
    df = summary_dataset.to_dataframe().reset_index()
    df.to_csv(output_csv, index=False)
    return df
