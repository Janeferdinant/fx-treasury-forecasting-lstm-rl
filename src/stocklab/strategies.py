"""Rule-based benchmark strategies. Positions are in {-1, 0, +1}."""

from __future__ import annotations

import numpy as np


def buy_and_hold_positions(n_days: int) -> np.ndarray:
    return np.ones(n_days)


def ma_crossover_positions(
    prices: np.ndarray, start: int, end: int, fast: int = 20, slow: int = 100
) -> np.ndarray:
    """Position held on each day d in [start, end), decided at the close of day d-1.

    Long (+1) when the ``fast`` simple moving average of closes up to d-1 is above the ``slow`` one,
    short (-1) when below, flat (0) until ``slow`` closes exist or on a tie. Only prices up to d-1
    are used, so the history before ``start`` may be used as a warm-up but nothing after d-1.
    """
    p = np.asarray(prices, dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(p)])
    pos = np.zeros(end - start)
    for k, d in enumerate(range(start, end)):
        t = d - 1
        if t + 1 < slow:
            continue
        ma_fast = (csum[t + 1] - csum[t + 1 - fast]) / fast
        ma_slow = (csum[t + 1] - csum[t + 1 - slow]) / slow
        pos[k] = float(np.sign(ma_fast - ma_slow))
    return pos
