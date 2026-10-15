"""Confidence intervals and significance tests."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _stationary_bootstrap_indices(
    n: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths with mean ``block``."""
    p = 1.0 / block
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def block_bootstrap_ci(
    x: np.ndarray, block: int = 10, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI for the mean of a (possibly autocorrelated) series."""
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(len(a), block, n_boot, rng)
    means = a[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_block_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, block: int = 10, n_boot: int = 1000, seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for mean(a) - mean(b), resampling days jointly."""
    return block_bootstrap_ci(
        np.asarray(a, dtype=float) - np.asarray(b, dtype=float), block, n_boot, seed, alpha
    )


def bootstrap_stat_ci(
    x: np.ndarray, stat, block: int = 10, n_boot: int = 500, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI of an arbitrary statistic (for example Sharpe) under the stationary bootstrap."""
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(len(a), block, n_boot, rng)
    vals = np.array([stat(a[i]) for i in idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray) -> tuple[float, float]:
    """One-sided Diebold-Mariano test on squared-error loss, one-step horizon.

    H1: the model has LOWER expected squared error than the benchmark. Uses a Newey-West variance
    with lag floor(n^(1/3)) and the Harvey-Leybourne-Newbold small-sample correction.
    Returns (statistic, p_value); a negative statistic favours the model.
    """
    d = np.asarray(e_model, dtype=float) ** 2 - np.asarray(e_bench, dtype=float) ** 2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lag = int(np.floor(n ** (1 / 3)))
    gamma0 = float(np.dot(dc, dc) / n)
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        var += 2.0 * w * float(np.dot(dc[k:], dc[:-k]) / n)
    if var <= 1e-30:
        return 0.0, 1.0
    stat = dbar / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 + 0.0) / n)  # HLN correction for h = 1
    p = float(sps.t.cdf(stat, df=n - 1))
    return float(stat), p


def binomial_pvalue(successes: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial test."""
    return float(sps.binomtest(int(successes), int(n), p0).pvalue)


def holm_adjust(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
