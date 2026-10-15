"""Metrics on toy data with hand-computed answers."""

import numpy as np
import pytest

from stocklab import metrics as m


def test_equity_curve_compounds_from_one():
    eq = m.equity_curve(np.array([0.10, -0.10, 0.05]))
    assert eq.shape == (4,)
    assert eq[0] == 1.0
    assert eq[1] == pytest.approx(1.10)
    assert eq[2] == pytest.approx(0.99)
    assert eq[3] == pytest.approx(0.99 * 1.05)


def test_max_drawdown_toy_path():
    # peak 3.0 then trough 1.2 -> (1.2 - 3.0) / 3.0 = -0.6
    assert m.max_drawdown(np.array([1.0, 2.0, 1.0, 3.0, 1.2])) == pytest.approx(-0.6)


def test_max_drawdown_monotone_is_zero():
    assert m.max_drawdown(np.array([1.0, 1.1, 1.2, 1.3])) == 0.0


def test_drawdown_series_matches_definition():
    dd = m.drawdown_series(np.array([1.0, 2.0, 1.0, 3.0, 1.2]))
    np.testing.assert_allclose(dd, [0.0, 0.0, -0.5, 0.0, -0.6])


def test_sharpe_known_value():
    r = np.array([0.01, 0.03] * 50)  # mean 0.02, sample std ~ 0.01005
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert m.sharpe(r) == pytest.approx(expected)


def test_sharpe_zero_variance_returns_zero():
    assert m.sharpe(np.full(10, 0.001)) == 0.0


def test_sharpe_symmetric_returns_is_zero():
    assert m.sharpe(np.array([0.01, -0.01] * 20)) == pytest.approx(0.0, abs=1e-12)


def test_annualised_return_doubling_in_one_year():
    r = np.full(252, 2 ** (1 / 252) - 1)
    assert m.annualised_return(r) == pytest.approx(1.0, rel=1e-9)


def test_annualised_return_rejects_empty():
    with pytest.raises(ValueError):
        m.annualised_return(np.array([]))


def test_directional_accuracy_ignores_zero_truth():
    y = np.array([1.0, -1.0, 0.0, 2.0])
    p = np.array([0.5, 0.5, 1.0, 3.0])
    # days with y == 0 are dropped: correct, wrong, correct -> 2/3
    assert m.directional_accuracy(y, p) == pytest.approx(2 / 3)


def test_directional_accuracy_all_correct():
    y = np.array([0.1, -0.2, 0.3])
    assert m.directional_accuracy(y, np.sign(y)) == 1.0


def test_rmse_mae_known():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    p = np.array([1.0, -1.0, 1.0, -1.0])
    assert m.rmse(y, p) == pytest.approx(1.0)
    assert m.mae(y, p) == pytest.approx(1.0)


def test_mse_skill_against_benchmark():
    y = np.array([1.0, 2.0, 3.0])
    assert m.mse_skill(y, y, np.zeros(3)) == pytest.approx(1.0)  # perfect model
    assert m.mse_skill(y, np.zeros(3), np.zeros(3)) == pytest.approx(0.0)  # equals benchmark
