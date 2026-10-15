"""Walk-forward (expanding window) split generator over sample indices."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Chronological train / validation / test index ranges. Validation sits right before test."""

    train: range
    val: range
    test: range


def walk_forward_folds(
    n_samples: int, min_train: int, test_size: int, val_size: int
) -> Iterator[Fold]:
    """Yield expanding-window folds.

    Fold k tests samples [min_train + k*test_size, +test_size). The model may use everything before
    that block: the last ``val_size`` samples before the block are held out for early stopping and
    the rest is training data. The final block is kept even if partial.
    """
    if test_size <= 0 or val_size <= 0 or min_train <= 0:
        raise ValueError("min_train, test_size and val_size must be positive")
    if val_size >= min_train:
        raise ValueError("val_size must be smaller than min_train")
    if min_train >= n_samples:
        raise ValueError("no room for a test block after min_train")
    start = min_train
    while start < n_samples:
        stop = min(start + test_size, n_samples)
        yield Fold(
            train=range(0, start - val_size),
            val=range(start - val_size, start),
            test=range(start, stop),
        )
        start = stop
