import numpy as np
import pytest

from stocklab import analysis as an
from stocklab.env import simulate_positions

COSTS = (0.0, 2.0, 5.0, 10.0, 20.0)


def _raw(n=400, seeds=(0, 1), seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 0.01, n)
    y = y - y.mean() + 0.002  # exact sample mean, so a constant 0.002 forecast must beat zero
    raw = {"day": np.arange(100, 100 + n), "y": y, "zero": np.zeros(n), "last": np.roll(y, 1),
           "drift": np.full(n, 0.002), "ses": np.full(n, 0.0002), "arima": np.full(n, 0.0003)}
    for s in seeds:
        raw[f"lstm_ret_s{s}"] = np.full(n, 0.0004) + 0.0001 * s
        raw[f"lstm_prob_s{s}"] = np.full(n, 0.55)
    return raw


def test_concat_blocks_keeps_order_and_keys():
    out = an.concat_blocks([{"a": np.array([1, 2])}, {"a": np.array([3])}])
    assert out["a"].tolist() == [1, 2, 3]


def test_lstm_ensemble_is_mean_over_seeds():
    ret, prob, seeds = an.lstm_ensemble(_raw())
    assert seeds == [0, 1]
    assert ret[0] == pytest.approx(0.00045)
    assert prob[0] == pytest.approx(0.55)


def test_forecast_report_zero_is_reference_and_constant_beats_it_when_mean_positive():
    rep = an.forecast_report(_raw())
    assert rep["models"]["zero"]["mse_skill_vs_zero"] == 0.0
    assert "dm_p_vs_zero" not in rep["models"]["zero"]
    assert rep["models"]["drift"]["mse_skill_vs_zero"] > 0
    lo, hi = rep["models"]["drift"]["mse_gain_ci"]
    assert lo <= hi
    assert 0.0 <= rep["models"]["lstm"]["dm_p_vs_zero"] <= 1.0
    assert "dm_p_vs_drift" in rep["models"]["lstm"]
    assert "dm_p_vs_drift" not in rep["models"]["drift"]
    assert rep["n_test"] == 400


def test_directional_bar_is_the_always_up_rate():
    raw = _raw()
    rep = an.forecast_report(raw)
    assert rep["share_up_days"] == pytest.approx(np.mean(raw["y"] > 0))
    assert 0.0 <= rep["models"]["lstm"]["directional_p_vs_always_up"] <= 1.0


def test_holm_over_assets_is_bounded_and_never_below_raw():
    per = {a: an.forecast_report(_raw(seed=i)) for i, a in enumerate("ABC")}
    adj = an.holm_over_assets(per)
    for a in per:
        raw_p = per[a]["models"]["lstm"]["dm_p_vs_zero"]
        assert raw_p - 1e-12 <= adj[a] <= 1.0


def test_net_over_blocks_starts_each_block_flat():
    r = np.full(300, 0.001)
    days = np.concatenate([np.arange(10, 20), np.arange(50, 60)])  # two blocks
    pos = np.ones(len(days))
    net, turn = an._net_over_blocks(pos, r, days, 10.0)
    assert turn[0] == 1.0 and turn[10] == 1.0  # entering at the start of each block
    assert turn.sum() == 2.0
    n0, _, _ = simulate_positions(np.ones(10), np.full(10, 0.001), 10.0)
    assert net[:10] == pytest.approx(n0)


def test_signal_strategies_shapes_and_values():
    n = 400
    r = np.random.default_rng(1).normal(0, 0.01, 700)
    prices = 100 * np.cumprod(1 + r)
    raw = _raw(n)
    raw["day"] = np.arange(150, 150 + n)
    pos = an.signal_strategies(raw, r, prices)
    assert set(pos) == {"buy_and_hold", "ma_crossover", "lstm_direction", "lstm_return_sign", "flat"}
    assert all(len(v) == n for v in pos.values())
    assert set(np.unique(pos["lstm_direction"])) == {1.0}  # prob 0.55 > 0.5
    assert not pos["flat"].any()


def test_strategy_report_pairs_against_benchmark():
    rng = np.random.default_rng(2)
    bench = rng.normal(0.0005, 0.01, 500)
    rep = an.strategy_report({"buy_and_hold": bench, "other": bench + 0.001})
    assert "daily_excess_vs_benchmark" not in rep["buy_and_hold"]
    assert rep["other"]["daily_excess_vs_benchmark"] == pytest.approx(0.001)
    lo, hi = rep["other"]["daily_excess_ci"]
    assert lo == pytest.approx(0.001) and hi == pytest.approx(0.001)  # constant difference, no spread
    assert rep["other"]["sharpe_ci"][0] <= rep["other"]["sharpe_ci"][1]


def test_dqn_report_ensemble_is_equal_weight_average():
    n = 60
    base = {"day": np.arange(n), "positions": np.ones(n)}

    def make(v):
        return {**base, **{f"net_{c:g}": np.full(n, v) for c in COSTS}}

    rep = an.dqn_report({0: make(0.001), 1: make(0.003)})
    assert rep["per_cost"]["5"]["ensemble"]["final_equity"] == pytest.approx(1.002**n)
    assert rep["share_long"] == 1.0 and rep["mean_abs_turnover_per_day"] == 0.0


def test_sample_curves_are_indexed_to_100_and_keep_the_worst_drawdown():
    rng = np.random.default_rng(3)
    net = rng.normal(0.0, 0.01, 503)
    idx, dd = an.sample_curves(net, every=5)
    sample_ix = an.sample_indices(len(net) + 1, 5)
    assert len(idx) == len(dd) == len(sample_ix)
    assert idx[0] == 100.0 and dd[0] == 0.0
    assert sample_ix[-1] == len(net)  # the final point is always kept
    eq = 100 * np.concatenate([[1.0], np.cumprod(1 + net)])
    assert idx[-1] == pytest.approx(round(float(eq[-1]), 2))
    full = eq / np.maximum.accumulate(eq) - 1
    assert min(dd) == pytest.approx(round(100 * float(full.min()), 2), abs=0.011)  # bucket minimum is kept
    assert all(d <= 0 for d in dd)


def test_sample_indices_cover_start_and_end():
    assert an.sample_indices(11, 5) == [0, 5, 10]
    assert an.sample_indices(12, 5) == [0, 5, 10, 11]
    assert an.sample_indices(1, 5) == [0]
