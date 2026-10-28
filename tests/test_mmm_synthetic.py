import numpy as np
from retailiq.mmm_synthetic import generate, validate, CHANNELS

def test_reconciliation():
    model,long,truth,baseline=generate()
    stats=validate(model,long,truth,baseline)
    assert stats['subchannels']==13
    assert np.isclose(stats['weekly_spend_ratio_min'], .05)
    assert len(model)==156
    assert long.groupby(['week_start_date','channel']).spend.sum().sum() == long.spend.sum()

def test_reproducibility():
    a=generate(seed=77)[0]
    b=generate(seed=77)[0]
    assert a.equals(b)

def test_truth_not_in_training_inputs():
    model,*_=generate()
    assert not any('true_' in x or 'baseline_demand' in x for x in model.columns)
