import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retailiq.mmm_synthetic import generate, validate

def test_reconciliation():
    model, long, truth, baseline = generate()
    stats = validate(model, long, truth, baseline)
    assert stats['subchannels'] == 13
    assert .045 < stats['weekly_spend_ratio_mean'] < .055
    assert stats['black_friday_weeks'] == 3
    assert stats['christmas_weeks'] == 3
    assert 'planned_demand' not in model.columns

def test_reproducible():
    a = generate(seed=123)
    b = generate(seed=123)
    assert np.allclose(a[0].total_sales, b[0].total_sales)
    assert np.allclose(a[1].spend, b[1].spend)

def test_spend_not_exact_sales_ratio():
    model, long, _, _ = generate()
    ratio = long.groupby('week_start_date').spend.sum().to_numpy() / model.total_sales.to_numpy()
    assert np.std(ratio) > .001
