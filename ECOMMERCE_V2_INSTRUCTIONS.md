# RetailIQ Marketing V2: e-commerce reconciliation

Update only these files in GitHub:

- `retailiq/ecommerce.py` — new synthetic online sales generator
- `retailiq/marketing.py` — replace Marketing V1 with reconciled version

Your existing `app.py` already imports `render_marketing`, so it does not need to change.

## Accounting
- Existing `data.py` store sales remain unchanged.
- `ecom_sales` is a new, non-rounded daily synthetic revenue series.
- `total_business_sales = store_sales + ecom_sales`.
- GA4 attributed revenue is 91–97% of the e-commerce sales for each date, allocated once across subchannels.
- Platform metrics remain synthetic and are not evidence of incrementality.
- The executive KPI cards currently remain store-only until you explicitly decide to include e-commerce there.

## Important limitation
The online business is generated after store sales. MMM cannot use this to validate causal marketing contribution. Attribution coverage is an explicit synthetic assumption.
