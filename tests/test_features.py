"""No look-ahead in features; shifted-label checks."""

import numpy as np
import pytest

from stocklab.features import Standardiser, build_supervised


def _returns(n=200, seed=0):
    return np.random.default_rng(seed).normal(0, 0.01, n)


def test_shapes_and_label_alignment():
    r = _returns()
    w = 10
    x, y, tidx = build_supervised(r, w)
    assert x.shape == (len(r) - w, w)
    # label of sample i is the return on the day AFTER the last feature day
    for i in (0, 5, 50, len(y) - 1):
        last_feature_day = i + w - 1
        assert tidx[i] == last_feature_day + 1
        assert y[i] == r[last_feature_day + 1]
        np.testing.assert_array_equal(x[i], r[i : i + w])


def test_features_ignore_the_future():
    """Changing any return after day t must not change feature rows that end at or before t."""
    r = _returns()
    w = 10
    x0, _, tidx = build_supervised(r, w)
    t = 120
    r2 = r.copy()
    r2[t + 1 :] = np.random.default_rng(99).normal(0, 0.05, len(r) - t - 1)
    x1, _, _ = build_supervised(r2, w)
    rows_ending_by_t = np.where(tidx - 1 <= t)[0]
    np.testing.assert_array_equal(x0[rows_ending_by_t], x1[rows_ending_by_t])
    # and rows that end after t do change (the check has teeth)
    assert not np.array_equal(x0[rows_ending_by_t[-1] + 1 :], x1[rows_ending_by_t[-1] + 1 :])


def test_no_feature_equals_its_own_label():
    """Shifted-label check: no column of X is the label or any later return."""
    r = _returns()
    x, y, _ = build_supervised(r, 15)
    for c in range(x.shape[1]):
        assert not np.allclose(x[:, c], y)


def test_leaky_pipeline_would_be_caught():
    """If a feature were built from the label day, the shifted-label check must flag it."""
    r = _returns()
    w = 5
    x, y, _ = build_supervised(r, w)
    leaky = np.column_stack([x, y])  # deliberately leaky
    assert any(np.allclose(leaky[:, c], y) for c in range(leaky.shape[1]))


def test_window_too_large_raises():
    with pytest.raises(ValueError):
        build_supervised(_returns(5), 10)


def test_standardiser_uses_training_statistics_only():
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    s = Standardiser.fit(train)
    assert s.mean == pytest.approx(2.0)
    far_future = np.array([[100.0, 100.0]])
    out = s.transform(far_future)
    # transform must not refit on the data it is given
    assert out[0, 0] == pytest.approx((100.0 - 2.0) / s.std)
    assert s.mean == pytest.approx(2.0)


def test_standardiser_zero_variance_is_safe():
    s = Standardiser.fit(np.zeros((4, 3)))
    assert np.isfinite(s.transform(np.ones((2, 3)))).all()
