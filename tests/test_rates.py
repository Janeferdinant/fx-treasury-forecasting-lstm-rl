"""The constant-maturity par-bond return built from Treasury yields."""

import numpy as np
import pandas as pd
import pytest

from stocklab import rates


def test_par_bond_prices_at_par_when_coupon_equals_yield():
    assert rates.bond_price(coupon_pct=4.0, yield_pct=4.0, maturity=10.0) == pytest.approx(100.0)


def test_price_falls_when_yield_rises_and_is_convex():
    base = rates.bond_price(4.0, 4.0, 10.0)
    up = rates.bond_price(4.0, 5.0, 10.0)
    down = rates.bond_price(4.0, 3.0, 10.0)
    assert up < base < down
    assert (down - base) > (base - up)  # convexity: a fall in yield gains more than a rise loses


def test_ten_year_duration_is_near_textbook_value():
    # modified duration of a 4% par 10-year bond with semiannual coupons is about 8.1 years
    dp = (rates.bond_price(4.0, 3.99, 10.0) - rates.bond_price(4.0, 4.01, 10.0)) / (0.02 / 100)
    assert dp / 100.0 == pytest.approx(8.1, abs=0.15)


def test_unchanged_yield_earns_only_carry():
    r = rates.par_bond_returns(pd.Series([4.0, 4.0, 4.0]), maturity=10.0)
    assert len(r) == 2
    np.testing.assert_allclose(r.to_numpy(), 4.0 / 100 / rates.DAYS_PER_YEAR)


def test_yield_rise_gives_loss_and_fall_gives_gain():
    r = rates.par_bond_returns(pd.Series([4.0, 4.10, 3.90]), maturity=10.0)
    assert r.iloc[0] < 0 < r.iloc[1]
    # about -duration * 10 bp = -0.8% plus a few bp of carry
    assert r.iloc[0] == pytest.approx(-0.0081 + 4.0 / 100 / rates.DAYS_PER_YEAR, abs=0.0004)


def test_index_is_a_compounded_total_return_and_stays_positive():
    y = pd.Series([4.0, 4.2, 3.8, 3.9], index=pd.bdate_range("2000-01-03", periods=4))
    idx = rates.par_bond_index(y, maturity=10.0)
    assert len(idx) == 3 and (idx > 0).all()
    r = rates.par_bond_returns(y, maturity=10.0)
    assert idx.iloc[-1] == pytest.approx(100 * np.prod(1 + r.to_numpy()))
    assert idx.index.equals(y.index[1:])


def test_rejects_non_positive_yields_and_bad_maturity():
    with pytest.raises(ValueError):
        rates.par_bond_returns(pd.Series([4.0, 0.0]), maturity=10.0)
    with pytest.raises(ValueError):
        rates.bond_price(4.0, 4.0, 0.0)
