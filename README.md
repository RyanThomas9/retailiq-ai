# RetailIQ — AI-Powered Retail Intelligence

An **independently implemented portfolio demo** built with Python, Streamlit, LightGBM and fictional synthetic retail data. No employer code, credentials, or confidential datasets are included.

## What works in v0.1
- Deterministic synthetic retail dataset (80 fictional stores, 1,000 days)
- LightGBM daily sales prediction with prior-date features and a 90-day chronological holdout
- Brand-adjusted financial discovery of two-attribute store groups
- Evidence-based executive briefing and drilldown
- Basic experiment difference-in-differences calculator
- Model performance view

Marketing, Customer, Merchandise, and Macro Intelligence are **planned**, not implemented ML modules yet.

## Run locally

```bash
python -m venv .venv
# activate your virtual environment
pip install -r requirements.txt
streamlit run app.py
```

## Publish
1. Create a **private** GitHub repository named `retailiq-ai`.
2. Upload these files (including the hidden `.streamlit/config.toml` file).
3. Test locally and review content for publication.
4. Make the repository public only when ready.
5. In Streamlit Community Cloud, deploy from your GitHub repository with `app.py` as the entrypoint.

## Methodological limitations
- Synthetic data is designed for demonstration, not representative of real retail performance.
- Daily holdout features use prior observed days in the test period: one-day-ahead operational simulation, not a fixed 90-day forecast.
- Discovery scans are exploratory; they have not undergone multiple-testing correction or prospective validation.
- Financial deviations are not causal effects or guaranteed recoverable revenue.
- The executive narrative is deterministic and grounded in generated model outputs, not an LLM or live economic data.
- The testing calculator is illustrative and does not establish causal identification by itself.

## Executive Intelligence upgrade

The home page now displays Yesterday, 1 Week, 13 Weeks, and 52 Weeks, comparing synthetic actual sales to model-fitted expected sales, previous periods, and 364-day-aligned prior-year periods. The last 90 days form the chronological validation holdout; earlier model expectations are in-sample fitted values and **must not** be interpreted as independently validated forecasts. The macro section optionally fetches public FRED CSV series (`UNRATE`, `CPIAUCSL`, `RSAFS`); if internet access fails, it displays no invented data. No external signals are yet included in the forecasting model or used to claim causality.

To deploy, upload the contents of this ZIP to the root of the GitHub repository preserving the `retailiq/` directory, and Streamlit Cloud will redeploy from `app.py`.
