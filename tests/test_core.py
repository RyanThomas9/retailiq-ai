from retailiq.data import generate
from retailiq.testing import difference_in_differences

def test_data_keys():
    d = generate(stores=3, days=40)
    assert len(d) == 120
    assert not d.duplicated(['store','date']).any()

def test_did():
    assert difference_in_differences(100,115,100,105) == 10
