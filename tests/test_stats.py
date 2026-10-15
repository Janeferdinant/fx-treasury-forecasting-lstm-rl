"""Bootstrap intervals, Diebold-Mariano test, multiple-testing correction."""

import numpy as np

from stocklab import stats


def test_block_bootstrap_ci_covers_true_mean_of_iid_data():
    rng = np.random.default_rng(0)
    x = rng.binomial(1, 0.5, 2000).astype(float)
    lo, hi = stats.block_bootstrap_ci(x, block=5, n_boot=500, seed=1)
    assert lo < 0.5 < hi
    assert hi - lo < 0.1


def test_block_bootstrap_is_seeded():
    x = np.random.default_rng(0).normal(size=300)
    a = stats.block_bootstrap_ci(x, block=5, n_boot=200, seed=7)
    b = stats.block_bootstrap_ci(x, block=5, n_boot=200, seed=7)
    assert a == b


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(1)
    a = (rng.random(3000) < 0.60).astype(float)
    b = (rng.random(3000) < 0.50).astype(float)
    lo, hi = stats.paired_block_bootstrap_ci(a, b, block=5, n_boot=500, seed=2)
    assert lo > 0.0


def test_paired_bootstrap_includes_zero_when_no_difference():
    rng = np.random.default_rng(2)
    a = (rng.random(1500) < 0.5).astype(float)
    b = (rng.random(1500) < 0.5).astype(float)
    lo, hi = stats.paired_block_bootstrap_ci(a, b, block=5, n_boot=500, seed=3)
    assert lo < 0.0 < hi


def test_diebold_mariano_identical_errors_gives_p_one():
    e = np.random.default_rng(0).normal(size=500)
    stat, p = stats.diebold_mariano(e, e)
    assert p == 1.0 and stat == 0.0


def test_diebold_mariano_detects_better_forecaster():
    rng = np.random.default_rng(0)
    e_bad = rng.normal(0, 2.0, 1500)
    e_good = rng.normal(0, 1.0, 1500)
    stat, p = stats.diebold_mariano(e_good, e_bad)  # alternative: first model has lower loss
    assert stat < 0
    assert p < 0.001


def test_diebold_mariano_not_significant_for_equal_quality():
    rng = np.random.default_rng(5)
    stat, p = stats.diebold_mariano(rng.normal(size=1000), rng.normal(size=1000))
    assert p > 0.05


def test_binomial_p_value_against_half():
    assert stats.binomial_pvalue(520, 1000) > 0.05
    assert stats.binomial_pvalue(600, 1000) < 1e-6


def test_holm_adjustment_known_values():
    adj = stats.holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_order():
    adj = stats.holm_adjust([0.9, 0.8])
    assert (adj <= 1.0).all()
    assert adj[0] >= adj[1]
