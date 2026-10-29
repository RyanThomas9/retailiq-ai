import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retailiq.mmm_synthetic import generate, validate

def test_outcome_reconciliation():
    model, spend, truth, baseline = generate()
    validate(model, spend, truth, baseline)
    w = model.set_index('week_start_date')
    b = baseline.set_index('week_start_date')
    t = truth.groupby('week_start_date')[['true_store_incremental_revenue','true_ecom_incremental_revenue']].sum()
    assert np.allclose(w.store_sales, b.store_baseline_demand + b.store_unobserved_noise + t.true_store_incremental_revenue)
    assert np.allclose(w.ecom_sales, b.ecom_baseline_demand + b.ecom_unobserved_noise + t.true_ecom_incremental_revenue)
    assert np.allclose(w.store_sales+w.ecom_sales,w.total_sales)
    assert not np.allclose(t.true_store_incremental_revenue/w.store_sales, t.true_ecom_incremental_revenue/w.ecom_sales)

def test_spend_and_truth():
    model, spend, truth, baseline = generate()
    totals = spend.groupby('week_start_date').spend.sum().to_numpy()
    assert np.allclose(totals, model.planned_media_budget.to_numpy())
    assert truth.groupby('week_start_date').subchannel.nunique().eq(13).all()
    assert np.allclose(truth.true_incremental_revenue, truth.true_store_incremental_revenue+truth.true_ecom_incremental_revenue)
