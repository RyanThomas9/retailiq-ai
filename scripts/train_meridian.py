"""Train separate Meridian models on GitHub Actions, without Streamlit.

Pilot is for environment verification, NOT trustworthy ROI inference.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retailiq.meridian_training import prepare_frame, truth_roi, CONTROLS, save_metrics_table


def train(outcome, pilot, output_dir, prior_mode):
    # Import after data validation to fail fast on malformed inputs.
    from meridian.data import load
    from meridian.model import model, spec, prior_distribution
    from meridian.analysis import analyzer
    from meridian import constants
    import tensorflow_probability.substrates.jax as tfp_jax
    import arviz as az

    csv_path = ROOT / 'sample_data' / 'meridian_weekly_inputs.csv'
    frame, media_map, spend_map, raw = prepare_frame(csv_path, outcome)
    if not np.allclose(raw.store_sales + raw.ecom_sales, raw.total_sales):
        raise ValueError('Sales totals do not reconcile')
    print(f'{outcome}: {len(frame)} weeks, {len(media_map)} subchannels', flush=True)

    cols = load.CoordToColumns(
        time='time', kpi='kpi', controls=CONTROLS,
        media=list(media_map), media_spend=list(spend_map),
    )
    data = load.DataFrameDataLoader(
        df=frame, coord_to_columns=cols, kpi_type='revenue',
        media_to_channel=media_map, media_spend_to_channel=spend_map,
        currency_code='USD',
    ).load()
    # This is a deliberately labeled calibration experiment. The true simulated
    # ROI is never used in the baseline model; oracle modes are NOT independent
    # ROI recovery experiments.
    allowed_prior_modes = ('baseline', 'truth_wide', 'truth_moderate')
    if prior_mode not in allowed_prior_modes:
        raise ValueError(f'Unknown prior mode: {prior_mode}')

    truth_path = ROOT / 'sample_data' / 'GROUND_TRUTH_DO_NOT_TRAIN.csv'
    spend_path = ROOT / 'sample_data' / 'marketing_spend_long.csv'
    prior_table = None
    if prior_mode == 'baseline':
        roi_prior = tfp_jax.distributions.LogNormal(
            loc=np.float64(0.2), scale=np.float64(0.9), name=constants.ROI_M
        )
    else:
        true_roi = truth_roi(truth_path, spend_path, outcome, csv_path)
        truth_by_name = true_roi.set_index('subchannel')['true_roi']
        ordered_names = [media_map[f'media_{i:02d}'] for i in range(len(media_map))]
        missing = sorted(set(ordered_names) - set(truth_by_name.index))
        if missing:
            raise ValueError(f'Missing truth ROI for prior calibration: {missing}')
        means = truth_by_name.loc[ordered_names].to_numpy(dtype=np.float64)
        if not np.isfinite(means).all() or (means <= 0).any():
            raise ValueError('Oracle prior means must be finite and strictly positive')
        # LogNormal mean is exp(loc + scale**2/2), so subtract sigma**2/2
        # to center the arithmetic prior mean on the synthetic ROI.
        sigma = 0.65 if prior_mode == 'truth_wide' else 0.35
        loc = np.log(means) - 0.5 * sigma**2
        roi_prior = tfp_jax.distributions.LogNormal(
            loc=loc, scale=np.full(len(means), sigma, dtype=np.float64),
            name=constants.ROI_M,
        )
        prior_table = pd.DataFrame({
            'subchannel': ordered_names, 'prior_mean_roi': means,
            'lognormal_sigma': sigma, 'prior_mode': prior_mode,
        })
    priors = prior_distribution.PriorDistribution(roi_m=roi_prior)
    specification = spec.ModelSpec(
        prior=priors, media_prior_type='roi', max_lag=8,
        hill_before_adstock=False, adstock_decay_spec='geometric',
    )
    mmm = model.Meridian(input_data=data, model_spec=specification)
    settings = (
        dict(n_chains=2, n_adapt=150, n_burnin=100, n_keep=150)
        if pilot else
        dict(n_chains=4, n_adapt=750, n_burnin=500, n_keep=750)
    )
    seed = 41 if outcome == 'stores' else 42
    start = time.monotonic()
    mmm.sample_prior(100 if pilot else 500, seed=seed)
    mmm.sample_posterior(**settings, seed=seed)

    folder = output_dir / outcome
    folder.mkdir(parents=True, exist_ok=True)
    if prior_table is not None:
        prior_table.to_csv(folder / 'oracle_prior_assumptions.csv', index=False)
    idata = mmm.inference_data
    idata.to_netcdf(folder / 'posterior.nc')
    diagnostics = az.summary(idata, group='posterior', kind='diagnostics')
    diagnostics.to_csv(folder / 'convergence.csv')
    summary = analyzer.Analyzer(mmm).summary_metrics()
    save_metrics_table(summary, folder / 'channel_summary_raw.csv')
    truth = truth_roi(
        ROOT / 'sample_data' / 'GROUND_TRUTH_DO_NOT_TRAIN.csv',
        ROOT / 'sample_data' / 'marketing_spend_long.csv',
        outcome, csv_path,
    )
    truth.to_csv(folder / 'held_out_ground_truth_roi.csv', index=False)
    metadata = {
        'outcome': outcome, 'pilot': pilot, 'prior_mode': prior_mode,
        'oracle_calibrated': prior_mode != 'baseline', 'weeks': len(frame),
        'subchannels': len(media_map), 'sampling': settings,
        'elapsed_seconds': round(time.monotonic() - start, 1),
        'status': 'pilot_unvalidated' if pilot else 'requires_validation',
        'caution': ('Oracle priors use synthetic truth; ROI recovery is not independent. ' if prior_mode != 'baseline' else '') + 'Not approved for decisions; validate convergence, identifiability, predictive checks, and predictive accuracy.',
    }
    (folder / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Finished {outcome} in {metadata["elapsed_seconds"]} seconds', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--outcome', choices=['stores', 'ecommerce', 'both'], default='both')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--prior-mode', choices=['baseline', 'truth_wide', 'truth_moderate'], default='truth_wide')
    args = parser.parse_args()
    print(f'Prior mode: {args.prior_mode}; training mode: {args.mode}', flush=True)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    outcomes = ('stores', 'ecommerce') if args.outcome == 'both' else (args.outcome,)
    for outcome in outcomes:
        train(outcome, args.mode == 'pilot', out, args.prior_mode)


if __name__ == '__main__':
    main()
