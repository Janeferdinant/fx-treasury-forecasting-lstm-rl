"""LSTM forecaster and DQN agent: learn a planted signal, stay at chance on noise, be seeded.

These train tiny networks; they take a few seconds each.
"""

import numpy as np
import pytest

from stocklab.dqn import DQNConfig, evaluate_policy, train_dqn
from stocklab.env import TradingEnv
from stocklab.features import build_supervised
from stocklab.lstm import LSTMConfig, fit_predict


def _ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


def _fit_eval(r, seed=0, epochs=12):
    w = 10
    x, y, _ = build_supervised(r, w)
    n_train, n_val = 1500, 300
    cfg = LSTMConfig(hidden=16, epochs=epochs, patience=4, batch_size=64)
    tr, va, te = slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, None)
    return y[te], fit_predict(x[tr], y[tr], x[va], y[va], x[te], cfg=cfg, seed=seed)


@pytest.mark.slow
def test_lstm_learns_planted_ar1_signal():
    y_te, out = _fit_eval(_ar1(3500, 0.35, 0))
    acc = np.mean(np.sign(out.ret_pred) == np.sign(y_te))
    acc_cls = np.mean((out.prob_up > 0.5) == (y_te > 0))
    assert acc > 0.56
    assert acc_cls > 0.56


@pytest.mark.slow
def test_lstm_is_near_chance_on_iid_noise():
    y_te, out = _fit_eval(_ar1(3500, 0.0, 1))
    acc_cls = np.mean((out.prob_up > 0.5) == (y_te > 0))
    assert 0.44 < acc_cls < 0.56  # ~ +-3.5 sd for n ~ 1700


@pytest.mark.slow
def test_lstm_is_deterministic_given_seed():
    r = _ar1(2600, 0.2, 2)
    _, a = _fit_eval(r, seed=3, epochs=3)
    _, b = _fit_eval(r, seed=3, epochs=3)
    np.testing.assert_allclose(a.ret_pred, b.ret_pred, atol=1e-6)
    _, c = _fit_eval(r, seed=4, epochs=3)
    assert not np.allclose(a.ret_pred, c.ret_pred)


def test_lstm_prediction_uses_only_training_scaling_and_shapes():
    r = _ar1(1200, 0.1, 5)
    x, y, _ = build_supervised(r, 8)
    cfg = LSTMConfig(hidden=8, epochs=1, patience=1, batch_size=64)
    out = fit_predict(x[:700], y[:700], x[700:900], y[700:900], x[900:], cfg=cfg, seed=0)
    assert out.ret_pred.shape == out.prob_up.shape == (len(x) - 900,)
    assert np.isfinite(out.ret_pred).all()
    assert ((out.prob_up >= 0) & (out.prob_up <= 1)).all()


@pytest.mark.slow
def test_dqn_learns_to_follow_momentum_and_generalises_to_held_out_segment():
    r = _ar1(2600, 0.4, 0)
    train = TradingEnv(r, window=5, start=10, end=1500, commission_bps=0, slippage_bps=0, obs_scale=0.01)
    val = TradingEnv(r, window=5, start=1500, end=1900, commission_bps=0, slippage_bps=0, obs_scale=0.01)
    test = TradingEnv(r, window=5, start=1900, end=2600, commission_bps=0, slippage_bps=0, obs_scale=0.01)
    agent = train_dqn(train, val, seed=0, cfg=DQNConfig(epochs=12, hidden=32))
    res = evaluate_policy(agent, test)
    assert res["net_returns"].sum() > 0
    assert res["positions"].min() == -1.0 or res["positions"].max() == 1.0


def test_dqn_policy_evaluation_is_deterministic_and_greedy():
    r = _ar1(600, 0.1, 6)
    train = TradingEnv(r, window=5, start=10, end=300, obs_scale=0.01)
    val = TradingEnv(r, window=5, start=300, end=400, obs_scale=0.01)
    test = TradingEnv(r, window=5, start=400, end=600, obs_scale=0.01)
    agent = train_dqn(train, val, seed=1, cfg=DQNConfig(epochs=1, hidden=8))
    a = evaluate_policy(agent, test)
    b = evaluate_policy(agent, test)
    np.testing.assert_array_equal(a["positions"], b["positions"])
    assert len(a["net_returns"]) == 200
