RetailIQ AI Discovery Engine V3 — Financial Impact Tabs

Replace app.py and retailiq/executive_v4.py; add retailiq/discovery_v3.py.
Five tabs: Yesterday (1 day), Last 7 Days (7), Last Month (28), Last 6 Months (182), Last Year (364).
Top 3 per tab ranked strictly by absolute descriptive peer-rate dollar gap.
Traffic, conversion, order size dollar allocation is an exact log-change allocation of that gap, NOT causal attribution.
The current sales dataset must include date, store, region, format, size, sales, traffic, orders.
No synthetic daily refresh or historical discovery memory is added.
