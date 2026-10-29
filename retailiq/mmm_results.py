"""Read-only display of exported Meridian model results; never trains in Streamlit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

RESULTS = Path(__file__).resolve().parents[1] / 'model_results' / 'meridian_truth_moderate'
OUTCOMES = ('stores', 'ecommerce')

@st.cache_data(show_spinner=False)
def load_results():
    results = {}
    for outcome in OUTCOMES:
        folder = RESULTS / outcome
        if not (folder / 'channel_summary_raw.csv').exists():
            return None
        summary = pd.read_csv(folder / 'channel_summary_raw.csv')
        summary = summary[summary['channel'] != 'All Channels'].copy()
        required = {'channel', 'metric', 'distribution', 'spend', 'incremental_outcome', 'roi', 'mroi'}
        if not required.issubset(summary.columns):
            raise ValueError(f'{outcome}: missing summary columns {required - set(summary.columns)}')
        if summary.channel.nunique() != 13:
            raise ValueError(f'{outcome}: expected 13 subchannels, found {summary.channel.nunique()}')
        diagnostics = pd.read_csv(folder / 'convergence.csv')
        metadata = json.loads((folder / 'run_metadata.json').read_text())
        truth = pd.read_csv(folder / 'held_out_ground_truth_roi.csv')
        priors = pd.read_csv(folder / 'oracle_prior_assumptions.csv')
        results[outcome] = dict(summary=summary, diagnostics=diagnostics, metadata=metadata,
                                truth=truth, priors=priors)
    return results

def _rows(frame, metric='mean', distribution='posterior'):
    selected = frame[(frame.metric == metric) & (frame.distribution == distribution)].copy()
    if selected.channel.duplicated().any() or len(selected) != 13:
        raise ValueError('Expected one summary row per channel')
    return selected.set_index('channel').sort_index()

def _combined(results, metric='mean'):
    stores = _rows(results['stores']['summary'], metric)
    ecommerce = _rows(results['ecommerce']['summary'], metric)
    if not stores.index.equals(ecommerce.index):
        raise ValueError('Store and e-commerce subchannels do not match')
    if not np.allclose(stores.spend, ecommerce.spend, rtol=1e-6, atol=1e-2):
        raise ValueError('Store and e-commerce spend denominators differ')
    combined = stores.copy()
    combined['incremental_outcome'] = stores.incremental_outcome + ecommerce.incremental_outcome
    combined['roi'] = combined.incremental_outcome / combined.spend
    # Combined marginal ROI is not derived here: separate models need consistent counterfactual increments.
    combined['mroi'] = np.nan
    return combined

def _money(value):
    return f'${value / 1e6:,.2f}M'

def render_mmm_results():
    st.markdown('<div class="ri-section">Meridian MMM — Fitted Results</div>', unsafe_allow_html=True)
    st.caption('156-week synthetic training scenario · Separate Store and E-commerce models · Truth-informed priors (moderate). Results are calibrated, not independently recovered ROI.')
    try:
        results = load_results()
    except Exception as exc:
        st.error(f'Could not load model results: {exc}')
        return
    if results is None:
        st.info('Fitted Meridian results have not been published yet. Upload the model_results folder from the integration package.')
        return
    from retailiq.mmm_weekly_view import render_weekly_contributions
    render_weekly_contributions()
    st.divider()
    st.markdown('#### Full-history Meridian summary')
    view = st.radio('Business outcome', ['Total Business', 'Stores', 'E-commerce'], horizontal=True, key='mmm_results_outcome')
    if view == 'Total Business':
        frame = _combined(results)
    else:
        frame = _rows(results['stores' if view == 'Stores' else 'ecommerce']['summary'])
    spend = float(frame.spend.sum())
    incremental = float(frame.incremental_outcome.sum())
    cols = st.columns(3)
    cols[0].metric('Modeled incremental revenue', _money(incremental))
    cols[1].metric('Marketing spend', _money(spend))
    cols[2].metric('Modeled iROAS', f'{incremental / spend:.2f}x')
    st.caption('Total Business uses Store + E-commerce incremental revenue divided by ONE shared media-spend denominator. It does not add the two spend amounts.')
    plot = frame.reset_index().sort_values('incremental_outcome')
    fig = px.bar(plot, x='incremental_outcome', y='channel', orientation='h',
                 color_discrete_sequence=['#A78BFA'], title='Modeled incremental revenue by subchannel')
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=510, margin=dict(l=5, r=5, t=45, b=5), xaxis_title='Incremental revenue ($)', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    table = frame.reset_index()[['channel','spend','incremental_outcome','roi','mroi']].copy()
    table.columns = ['Subchannel','Spend','Incremental revenue','iROAS','mROAS']
    st.dataframe(table.style.format({'Spend':'${:,.0f}', 'Incremental revenue':'${:,.0f}',
                                     'iROAS':'{:.2f}x','mROAS':'{:.2f}x'}, na_rep='—'),
                 hide_index=True, use_container_width=True)
    if view != 'Total Business':
        key = 'stores' if view == 'Stores' else 'ecommerce'
        source = results[key]['summary']
        lo = _rows(source, 'ci_lo')[['roi']].rename(columns={'roi':'lower'})
        hi = _rows(source, 'ci_hi')[['roi']].rename(columns={'roi':'upper'})
        intervals = frame[['roi']].join(lo).join(hi).reset_index()
        with st.expander('ROI posterior intervals'):
            st.dataframe(intervals.style.format({'roi':'{:.2f}x','lower':'{:.2f}x','upper':'{:.2f}x'}),
                         hide_index=True, use_container_width=True)
    else:
        st.caption('Combined credible intervals and combined mROAS are not shown: they require combining joint posterior draws or consistent counterfactual curves, not adding interval endpoints.')
    st.caption('Predictive holdout validation and response curves remain pending. Weekly model fit is in-sample.')

def render_mmm_model_validation():
    st.markdown('<div class="ri-eyebrow">RETAILIQ AI / MODEL VALIDATION</div>', unsafe_allow_html=True)
    st.title('Marketing Mix Modeling (MMM)')
    st.caption('Actual diagnostics from full Meridian sampling. Truth-informed prior calibration is not independent ROI validation.')
    try:
        results = load_results()
    except Exception as exc:
        st.error(f'Could not load model diagnostics: {exc}')
        return
    if results is None:
        st.info('Model results have not been published yet.')
        return
    rows = []
    for outcome in OUTCOMES:
        r = results[outcome]
        diag = r['diagnostics']
        rhat_col = next((x for x in diag.columns if x.lower() in ('r_hat', 'rhat')), None)
        ess_col = next((x for x in diag.columns if x.lower() in ('ess_bulk','bulk_ess')), None)
        summary = _rows(r['summary'])[['roi']].reset_index()
        truth = r['truth']
        merged = summary.merge(truth[['subchannel','true_roi']], left_on='channel', right_on='subchannel', validate='one_to_one')
        mae = float(np.mean(np.abs(merged.roi - merged.true_roi)))
        rows.append({'Outcome': outcome.title(), 'Weeks': r['metadata'].get('weeks'),
                     'Chains': r['metadata'].get('sampling', {}).get('n_chains'),
                     'Max R-hat': float(diag[rhat_col].max()) if rhat_col else np.nan,
                     'Min bulk ESS': float(diag[ess_col].min()) if ess_col else np.nan,
                     'ROI absolute error (x)': mae,
                     'Prior mode': r['metadata'].get('prior_mode', 'Unknown')})
    st.dataframe(pd.DataFrame(rows).style.format({'Max R-hat':'{:.3f}', 'Min bulk ESS':'{:,.0f}',
                                                 'ROI absolute error (x)':'{:.3f}x'}),
                 hide_index=True, use_container_width=True)
    st.warning('These models were calibrated with synthetic ground-truth ROI as priors. The ROI recovery metric is therefore not an independent validation. Holdout WAPE and posterior predictive checks have not yet been computed.')
    chosen = st.selectbox('Inspect outcome', ['stores','ecommerce'], key='mmm_validation_outcome')
    r = results[chosen]
    means = _rows(r['summary'])[['roi']].reset_index()
    compare = means.merge(r['truth'][['subchannel','true_roi']], left_on='channel', right_on='subchannel', validate='one_to_one')
    compare = compare[['channel','true_roi','roi']].rename(columns={'channel':'Subchannel','true_roi':'Synthetic truth','roi':'Posterior mean'})
    st.dataframe(compare.style.format({'Synthetic truth':'{:.2f}x','Posterior mean':'{:.2f}x'}), hide_index=True, use_container_width=True)
    with st.expander('Raw MCMC diagnostics'):
        st.dataframe(r['diagnostics'], use_container_width=True)
