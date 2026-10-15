"""Rule-based strategies: buy-and-hold and moving-average crossover."""

import numpy as np

from stocklab.strategies import buy_and_hold_positions, ma_crossover_positions


def test_buy_and_hold_is_constant_long():
    np.testing.assert_array_equal(buy_and_hold_positions(5), np.ones(5))


def test_ma_crossover_follows_trend_direction():
    up = np.linspace(100, 200, 300)
    pos = ma_crossover_positions(up, start=150, end=300, fast=5, slow=20)
    assert (pos == 1.0).all()
    down = np.linspace(200, 100, 300)
    pos = ma_crossover_positions(down, start=150, end=300, fast=5, slow=20)
    assert (pos == -1.0).all()


def test_ma_crossover_is_flat_before_slow_window_is_available():
    prices = np.linspace(100, 200, 100)
    pos = ma_crossover_positions(prices, start=1, end=30, fast=5, slow=20)
    # decisions at t = start-1 .. ; slow MA needs 20 observations (t >= 19)
    assert (pos[: 19 - 0] == 0.0).all()


def test_ma_crossover_has_no_look_ahead():
    rng = np.random.default_rng(3)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    base = ma_crossover_positions(p, start=100, end=400, fast=10, slow=50)
    p2 = p.copy()
    p2[251:] *= 3.0  # perturb the future after day 250
    alt = ma_crossover_positions(p2, start=100, end=400, fast=10, slow=50)
    # position for day d is decided at t=d-1, so days d <= 251 may only see prices <= 250
    np.testing.assert_array_equal(base[: 251 - 100 + 1], alt[: 251 - 100 + 1])
    assert not np.array_equal(base, alt)
