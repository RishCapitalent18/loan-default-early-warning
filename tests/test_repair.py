"""Unit tests for the data-repair layer: the rebuild has to be correct before
any model result can be trusted."""
import numpy as np
from src.ews.pipeline import money, pct, emp, fico

def test_money_parses_dollars_and_commas():
    assert money("$31,917.63") == 31917.63
    assert money("40000") == 40000.0

def test_money_rejects_unusable():
    assert np.isnan(money("N/A"))
    assert np.isnan(money("0"))
    assert np.isnan(money("-1"))

def test_pct_strips_sign():
    assert pct("45.5%") == 45.5
    assert np.isnan(pct(""))

def test_emp_freetext():
    assert emp("< 1 year") == 0.0
    assert emp("10+ years") == 10.0
    assert emp("6 years") == 6.0
    assert np.isnan(emp("n/a"))

def test_fico_sentinel_dropped():
    assert np.isnan(fico(9999))
    assert fico(720) == 720
