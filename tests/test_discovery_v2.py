import numpy as np
from retailiq.discovery_v2 import discover_v2, _bh_adjust
from data_tmp import generate

def test_bh_bounds_and_order():
    q = _bh_adjust([.001, .04, .8])
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] <= q[1] <= q[2]

def test_multiwindow_and_deterministic():
    data = generate(stores=80, days=1000)
    a = discover_v2(data)
    b = discover_v2(data)
    assert not a.empty
    assert a.equals(b)
    assert set(a.period).issubset({'1 Week','4 Weeks','13 Weeks','52 Weeks'})
    assert a[['attribute','group']].duplicated().sum() == 0
    assert a.q_value.between(0, 1).all()
