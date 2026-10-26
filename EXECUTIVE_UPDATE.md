# Executive Intelligence refresh

This version keeps the existing model and modules but simplifies the home page to:

1. Four actual-sales cards (Yesterday, 1 Week, 13 Weeks, 52 Weeks).
2. Top AI Discoveries Found Yesterday: one featured and two compact store-group findings.
3. AI Market Watch: real RSS headlines scored by transparent keyword relevance, with links and timestamps.

## Accuracy and limitations

- The synthetic dataset is static until the data generator is updated; **Found Yesterday** is a requested section title, not evidence that a nightly job ran. The UI explicitly labels the findings as the latest available model run.
- Discoveries are exploratory, based on brand-relative store-group residuals. They are not causal sales impacts.
- AI Market Watch is free rules-based headline analysis, **not** an LLM or a news-based sales impact model. It fetches Google News RSS with a six-hour cache and fails gracefully when offline.
- Weather only appears on the executive page if a representative-city public forecast crosses the severe-weather or snow thresholds; no weather card appears in normal conditions.
- For daily automated publication, add a scheduled GitHub Actions pipeline in a later release. The current app refreshes its RSS cache while active.
- Actual and expected 52-week sales include historical fitted values, not a fully out-of-sample forecast.

## Deploy

Upload `app.py`, `retailiq/executive.py`, `retailiq/market_watch.py`, and `retailiq/theme.py` to their corresponding paths in the GitHub repository. Existing dependencies are sufficient. Streamlit Community Cloud should redeploy from the main branch.
