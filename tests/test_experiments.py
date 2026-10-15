import numpy as np
import pytest

from stocklab import experiments as ex
from stocklab.dqn import DQNConfig
from stocklab.lstm import LSTMConfig

CFG = ex.ForecastConfig(window=5, min_train=300, test_size=100, val_size=50, seeds=(0,),
                        lstm=LSTMConfig(hidden=4, epochs=1))
KEYS = ("zero", "last", "drift", "ses", "arima", "lstm_ret_s0", "lstm_prob_s0")


def _returns(n=520, seed=0):
    return np.random.default_rng(seed).normal(0.0003, 0.01, n)


def test_forecast_and_trade_blocks_cover_the_same_days():
    n = 700
    folds = ex.forecast_folds(n, CFG)
    blocks = ex.trade_blocks(n, CFG)
    assert len(folds) == len(blocks)
    for f, (a, b) in zip(folds, blocks, strict=True):
        assert (a, b) == (f.test[0] + CFG.window, f.test[-1] + 1 + CFG.window)
    assert blocks[-1][1] == n  # the last block ends at the last return


@pytest.mark.slow
def test_forecast_block_has_all_models_and_aligned_targets():
    r = _returns()
    out = ex.forecast_block(r, 0, CFG)
    n = len(out["y"])
    for key in KEYS:
        assert len(out[key]) == n
    assert np.array_equal(out["y"], r[out["day"]])
    assert np.array_equal(out["last"], r[out["day"] - 1])
    assert out["drift"][0] == pytest.approx(r[: out["day"][0]].mean())


@pytest.mark.slow
def test_forecast_for_first_test_day_ignores_that_day_and_everything_after():
    r = _returns()
    out = ex.forecast_block(r, 0, CFG)
    first = int(out["day"][0])
    r2 = r.copy()
    r2[first:] = np.random.default_rng(9).normal(0, 0.05, len(r) - first)  # rewrite the future
    out2 = ex.forecast_block(r2, 0, CFG)
    for key in KEYS:
        assert out2[key][0] == pytest.approx(out[key][0], rel=1e-5, abs=1e-8), key


@pytest.mark.slow
def test_trade_block_outputs_for_every_cost_and_valid_positions():
    r = np.expm1(_returns(700))
    tcfg = ex.TradeConfig(window=5, train_len=200, val_len=50, seeds=(0,),
                          dqn=DQNConfig(epochs=1, hidden=8, learn_start=50, batch_size=16))
    block = ex.trade_blocks(len(r), CFG)[1]
    out = ex.trade_block(r, block, 0, tcfg)
    n = block[1] - block[0]
    assert len(out["positions"]) == n and set(np.unique(out["positions"])) <= {-1.0, 0.0, 1.0}
    for c in ex.COSTS_BPS:
        assert len(out[f"net_{c:g}"]) == n
    # higher costs can only lower the total net return for the same positions
    assert out["net_0"].sum() >= out["net_20"].sum()
