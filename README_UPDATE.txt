RetailIQ AI - Discovery Engine V2

Only upload these three files to the same locations in your existing GitHub repository:
  app.py                       REPLACE
  retailiq/executive_v4.py     REPLACE
  retailiq/discovery_v2.py     ADD

Optional: tests/test_discovery_v2.py

Leave all other files as they are. This is a patch, not a complete application.

V2 scans 1-, 4-, 13-, and 52-week windows with aligned 364-day year-over-year comparisons, comparing store groups with the other stores. It calculates peer-growth dollar differences, group-vs-peer percentage point gaps, store-level Welch tests, Benjamini-Hochberg FDR q-values, weekly persistence, and trend slopes. It ranks and deduplicates discoveries before showing three full-size tiles.

Important limitations:
- These are observational, descriptive peer comparisons, NOT causal sales impacts.
- FDR correction addresses multiple candidate comparisons but does not remove all selection and dependence issues.
- No daily history, genuine 'new today' status, or nightly job yet.
- Synthetic data is fixed through 2026-10-17.
- The existing four executive KPI cards still use the current scored-data window and are NOT recalculated by Discovery V2.
- The full history is used only for peer-growth comparisons, not for out-of-sample 52-week forecasts.
