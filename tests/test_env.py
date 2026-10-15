"""Trading environment: step logic and cost accounting."""

import numpy as np
import pytest

from stocklab.env import TradingEnv, simulate_positions

R = np.array([0.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.03, 0.00, 0.01, -0.01])  # simple returns


def make_env(**kw):
    defaults = {"window": 3, "commission_bps": 0.0, "slippage_bps": 0.0}
    defaults.update(kw)
    return TradingEnv(R, **defaults)


def test_spaces_and_reset():
    env = make_env(start=3, end=10)
    obs, info = env.reset(seed=0)
    assert env.action_space.n == 3
    assert obs.shape == env.observation_space.shape == (3 + 1,)
    assert info["position"] == 0.0
    assert info["equity"] == 1.0


def test_reward_is_position_times_next_day_return_no_costs():
    env = make_env(start=4, end=8)
    env.reset()
    # action 2 = long (+1). First earned day is day 4 (return 0.01).
    _, r, term, _, info = env.step(2)
    assert r == pytest.approx(0.01)
    assert not term
    # action 0 = short (-1). Day 5 return is -0.02 -> +0.02
    _, r, _, _, info = env.step(0)
    assert r == pytest.approx(0.02)
    assert info["position"] == -1.0


def test_flat_position_earns_nothing_and_pays_nothing():
    env = make_env(start=4, end=8, commission_bps=50, slippage_bps=50)
    env.reset()
    _, r, _, _, info = env.step(1)  # flat
    assert r == 0.0
    assert info["cost"] == 0.0


def test_cost_charged_on_turnover_only():
    env = make_env(start=4, end=9, commission_bps=10.0, slippage_bps=5.0)  # 15 bps one-way
    env.reset()
    _, r1, _, _, i1 = env.step(2)  # 0 -> +1 : turnover 1
    assert i1["turnover"] == 1.0
    assert i1["cost"] == pytest.approx(15e-4)
    assert r1 == pytest.approx(0.01 - 15e-4)
    _, r2, _, _, i2 = env.step(2)  # hold +1 : no turnover, no cost
    assert i2["turnover"] == 0.0
    assert r2 == pytest.approx(-0.02)
    _, _, _, _, i3 = env.step(0)  # +1 -> -1 : turnover 2
    assert i3["turnover"] == 2.0
    assert i3["cost"] == pytest.approx(2 * 15e-4)


def test_episode_terminates_and_length_matches_segment():
    env = make_env(start=4, end=9)
    env.reset()
    steps, term = 0, False
    while not term:
        _, _, term, trunc, _ = env.step(1)
        steps += 1
        assert not trunc
    assert steps == 9 - 4


def test_equity_matches_compounded_net_rewards():
    env = make_env(start=4, end=10, commission_bps=7.0, slippage_bps=3.0)
    env.reset()
    rewards, actions = [], [2, 2, 0, 0, 1, 2]
    for a in actions:
        _, r, _, _, info = env.step(a)
        rewards.append(r)
    assert info["equity"] == pytest.approx(np.prod(1 + np.array(rewards)))


def test_observation_contains_only_past_returns():
    """The observation at decision time t must be built from r[t-window+1..t] and the position."""
    env = make_env(start=4, end=10)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs[:3], R[1:4])  # decision at t=3, earns day 4
    obs, *_ = env.step(2)
    np.testing.assert_allclose(obs[:3], R[2:5])
    assert obs[3] == 1.0  # position is the last element


def test_future_returns_do_not_leak_into_observations():
    r2 = R.copy()
    r2[7:] = 9.99  # change the far future
    a = TradingEnv(R, window=3, start=4, end=10, commission_bps=0, slippage_bps=0)
    b = TradingEnv(r2, window=3, start=4, end=10, commission_bps=0, slippage_bps=0)
    oa, _ = a.reset()
    ob, _ = b.reset()
    for _ in range(2):  # decisions at t=3,4,5 only look at r[<=5]
        np.testing.assert_array_equal(oa, ob)
        oa, *_ = a.step(2)
        ob, *_ = b.step(2)


def test_env_matches_pure_simulator():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 60)
    acts = rng.integers(0, 3, size=40)
    env = TradingEnv(r, window=5, start=10, end=50, commission_bps=4.0, slippage_bps=6.0)
    env.reset()
    got = []
    for a in acts[:40]:
        _, rew, term, _, _ = env.step(int(a))
        got.append(rew)
        if term:
            break
    pos = np.array([(-1.0, 0.0, 1.0)[a] for a in acts[: len(got)]])
    net, _, _ = simulate_positions(pos, r[10:50][: len(got)], cost_bps=10.0)
    np.testing.assert_allclose(got, net)


def test_simulator_buy_and_hold_pays_entry_cost_once():
    r = np.array([0.01, 0.02, -0.01])
    net, turnover, cost = simulate_positions(np.ones(3), r, cost_bps=100.0)
    assert turnover.sum() == 1.0
    assert cost.sum() == pytest.approx(0.01)
    assert net[0] == pytest.approx(0.01 - 0.01)
    assert net[1] == pytest.approx(0.02)


def test_validation_of_segment_bounds():
    with pytest.raises(ValueError):
        TradingEnv(R, window=3, start=2, end=8)  # not enough history for the first window
    with pytest.raises(ValueError):
        TradingEnv(R, window=3, start=4, end=11)  # beyond data
    with pytest.raises(ValueError):
        TradingEnv(R, window=3, commission_bps=-1.0)
