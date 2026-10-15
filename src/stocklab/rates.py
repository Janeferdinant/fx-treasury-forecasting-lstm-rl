"""A constant-maturity par-bond total-return series built from Treasury constant-maturity yields.

The yield series (Federal Reserve H.15 constant-maturity yields, for example ``RIFLGFCY10_N.B``) is
not a price. To get a return series that a position can earn, each day we revalue a par bond:
yesterday it had a coupon equal to yesterday's yield and exactly ``maturity`` years left; today it is
repriced at today's yield with the same maturity (so roll-down is ignored) and earns one trading day
of coupon carry. This is a standard duration-and-convexity style approximation, not an actual
Treasury index and not a tradable product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252


def bond_price(coupon_pct, yield_pct, maturity: float, freq: int = 2):
    """Price per 100 of face of a bullet bond; coupon and yield are annual percentages.

    Works on scalars or arrays. ``maturity`` in years must be a positive multiple of 1/freq.
    """
    n = round(maturity * freq)
    if maturity <= 0 or n < 1 or abs(n - maturity * freq) > 1e-9:
        raise ValueError("maturity must be a positive multiple of the coupon period")
    c = np.asarray(coupon_pct, dtype=float) / 100.0 / freq * 100.0
    i = np.asarray(yield_pct, dtype=float) / 100.0 / freq
    annuity = (1.0 - (1.0 + i) ** (-n)) / i
    price = c * annuity + 100.0 * (1.0 + i) ** (-n)
    return float(price) if price.ndim == 0 else price


def par_bond_returns(yields_pct: pd.Series, maturity: float) -> pd.Series:
    """One-day total returns (decimals) for days 2..N of a yield series (percent)."""
    y = pd.Series(yields_pct, dtype=float)
    if (y <= 0).any() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("yields must be positive and finite")
    prev, new = y.to_numpy()[:-1], y.to_numpy()[1:]
    price_new = bond_price(prev, new, maturity)
    ret = price_new / 100.0 - 1.0 + prev / 100.0 / DAYS_PER_YEAR
    return pd.Series(ret, index=y.index[1:], name="ret")


def par_bond_index(yields_pct: pd.Series, maturity: float, base: float = 100.0) -> pd.Series:
    """Compounded total-return index (value after each day, starting from ``base`` before day 2)."""
    r = par_bond_returns(yields_pct, maturity)
    return (base * (1.0 + r).cumprod()).rename("price")
