# RetailIQ AI Marketing Intelligence V1

## Install into the existing GitHub repository

1. Replace root `app.py` with the included `app.py`.
2. Add `retailiq/marketing.py` to the `retailiq` folder.
3. Commit both changes. Streamlit Community Cloud should redeploy automatically.

All other existing files remain unchanged.

## What it does

- Generates deterministic synthetic marketing spend, impressions, clicks, GA4 sessions, transactions and GA4 attributed revenue, anchored to the existing retail sales date range and demand levels.
- Five time-window discovery tabs, ranked by absolute changes in attributed GA4 revenue vs the preceding equivalent period, with a $25,000 minimum.
- Three driver types (sessions, conversion, order value), direction-specific investigation suggestions, and a channel spend chart.

## Interpretation and limitations

- Existing store sales are the business source of truth. GA4 attributed revenue is a separate, overlapping attribution proxy and must NOT be added to business sales.
- Simulated marketing spend was generated after sales. Correlations are not causal effects; MMM, iROAS and budget optimization are intentionally excluded.
- Financial changes in discovery tiles are changes in GA4-attributed revenue, not incremental business revenue.
- For a 52-week tab, the preceding equivalent period must exist; the existing 1000-day sales history is sufficient.

## Requirements

The existing `requirements.txt` already needs `streamlit`, `numpy`, `pandas`, and `plotly` for the original application.
