"""Export actual time-resolved Meridian posterior counterfactuals.

Run ONLY after model fitting, outside Streamlit. This module does not use
synthetic ground truth and does not manufacture weekly ROI from full-period ROI.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def export_weekly(analyzer_obj, raw, media_map, folder, outcome, batch_size=100):
    """Export fitted weekly contributions and joint posterior draws.

    Expected and media counterfactuals are computed using Meridian's Analyzer.
    Baseline is fitted expected revenue less the sum of channel contributions;
    it includes seasonality, holidays, promotion, weather and other controls.
    It is NOT a separately identified holiday/promotion decomposition.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    channels = [media_map[f'media_{i:02d}'] for i in range(len(media_map))]
    dates = pd.to_datetime(raw['week_start_date']).dt.strftime('%Y-%m-%d').to_numpy()
    n_weeks = len(dates)
    expected = np.asarray(analyzer_obj.expected_outcome(
        use_posterior=True, aggregate_geos=True, aggregate_times=False,
        batch_size=batch_size,
    ), dtype=np.float64)
    incremental = np.asarray(analyzer_obj.incremental_outcome(
        use_posterior=True, aggregate_geos=True, aggregate_times=False,
        include_non_paid_channels=False, batch_size=batch_size,
    ), dtype=np.float64)
    if expected.ndim != 3 or expected.shape[-1] != n_weeks:
        raise ValueError(f'Unexpected expected outcome shape {expected.shape}; expected (chains, draws, {n_weeks})')
    if incremental.ndim != 4 or incremental.shape[-2:] != (n_weeks, len(channels)):
        raise ValueError(f'Unexpected incremental outcome shape {incremental.shape}; expected (chains, draws, {n_weeks}, {len(channels)})')
    if expected.shape[:2] != incremental.shape[:2]:
        raise ValueError('Posterior draw dimensions do not match')
    if not np.isfinite(expected).all() or not np.isfinite(incremental).all():
        raise ValueError('Non-finite posterior predictions')
    # Channel contributions use the same historical-zero counterfactual and
    # Meridian's channel additive decomposition convention.
    fitted = expected.mean(axis=(0, 1))
    contrib = incremental.mean(axis=(0, 1))
    media = contrib.sum(axis=1)
    actual = raw['store_sales' if outcome == 'stores' else 'ecom_sales'].to_numpy(dtype=float)
    baseline = fitted - media
    residual = actual - fitted
    fit = pd.DataFrame({
        'week_start_date': dates, 'actual_sales': actual, 'fitted_sales': fitted,
        'baseline_and_controls': baseline, 'marketing_contribution': media,
        'residual': residual,
    })
    if not np.allclose(fit['baseline_and_controls'] + fit['marketing_contribution'] + fit['residual'], fit['actual_sales'], atol=0.01):
        raise AssertionError('Weekly sales reconciliation failed')
    fit.to_csv(folder / 'weekly_model_fit.csv', index=False)
    weekly = pd.DataFrame({
        'week_start_date': np.repeat(dates, len(channels)),
        'subchannel': np.tile(channels, n_weeks),
        'incremental_revenue': contrib.reshape(-1),
    })
    spend = raw[channels].to_numpy(dtype=float)
    weekly['spend'] = spend.reshape(-1)
    weekly.to_csv(folder / 'weekly_media_contribution.csv', index=False)
    # Keep JOINT draws, not independent weekly credible bounds. This permits
    # exact period aggregation of correlated posterior contributions.
    np.savez_compressed(folder / 'weekly_posterior_draws.npz',
        incremental=incremental.reshape(-1, n_weeks, len(channels)).astype('float32'),
        expected=expected.reshape(-1, n_weeks).astype('float32'),
        dates=dates, channels=np.array(channels, dtype=str),
    )
    (folder / 'weekly_export_metadata.json').write_text(json.dumps({
        'outcome': outcome, 'weeks': n_weeks, 'channels': len(channels),
        'draws': int(np.prod(expected.shape[:2])),
        'baseline_definition': 'fitted expected revenue minus summed media contributions; includes controls',
        'residual_definition': 'observed minus fitted revenue',
        'warning': 'Historical fitted decomposition, not holdout accuracy. Calibrated using synthetic true ROI priors.',
    }, indent=2))
    return fit, weekly
