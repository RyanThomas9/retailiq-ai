# RetailIQ Discovery V3.1

Replace only:
- `retailiq/discovery_v3.py`
- `retailiq/executive_v4.py`

No change to `app.py`, sidebar, model training, five tabs, or AI Market Watch.

Changes:
- $25,000 minimum absolute peer-relative financial deviation applied before ranking and again at display.
- Positive and negative findings eligible, ranked together by absolute impact within each tab.
- Headline identifies the largest calculated traffic/conversion/order-size contribution.
- Recommendations distinguish opportunities to replicate positive drivers from investigations into negative drivers.
- Descriptive estimates remain explicitly noncausal and retain FDR evidence labels.

Note: 'Last Month' remains 28 days, 'Last 6 Months' 182 days, and 'Last Year' 364 days, matching V3.
