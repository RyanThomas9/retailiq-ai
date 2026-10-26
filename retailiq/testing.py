import numpy as np
from scipy import stats


def difference_in_differences(treatment_before, treatment_after, control_before, control_after):
    """Descriptive DID point estimate. Inference requires unit-level observations and design checks."""
    return (treatment_after - treatment_before) - (control_after - control_before)


def randomized_test(treatment, control):
    """Illustrative Welch test; use unit-level outcomes, not aggregated totals."""
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    if len(t) < 2 or len(c) < 2:
        raise ValueError('Need at least two observations per group')
    result = stats.ttest_ind(t, c, equal_var=False)
    return {'difference': float(t.mean() - c.mean()), 'p_value': float(result.pvalue),
            'treatment_mean': float(t.mean()), 'control_mean': float(c.mean())}
