"""Supervised windows with a strict next-day label, and train-only standardisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_supervised(returns: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a return series r[0..N-1] into (X, y, target_day).

    Sample i uses the days r[i .. i+window-1] as features and the NEXT day r[i+window] as the
    label, so nothing at or after the label day ever appears in a feature row.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r) - window
    if window < 1 or n < 1:
        raise ValueError("series too short for this window")
    idx = np.arange(n)[:, None] + np.arange(window)[None, :]
    x = r[idx]
    target_day = np.arange(n) + window
    y = r[target_day]
    return x, y, target_day


@dataclass(frozen=True)
class Standardiser:
    """Scalar mean/std fitted on training data only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, train: np.ndarray) -> Standardiser:
        a = np.asarray(train, dtype=float)
        sd = float(a.std())
        return cls(mean=float(a.mean()), std=sd if sd > 1e-12 else 1.0)

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, dtype=float) - self.mean) / self.std
