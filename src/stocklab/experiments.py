"""The two experiments: walk-forward one-day-ahead forecasting, and walk-forward DQN trading.

Both use the same chronological test blocks per asset, so forecasting and trading results refer to
the same days. Per-block work is a pure function of (asset data, block, seed), which is what the
process pool in ``stocklab.cli`` parallelises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import baselines as bl
from .data_sources import load_asset
from .dqn import DQNConfig, evaluate_policy, train_dqn
from .env import TradingEnv
from .features import build_supervised
from .lstm import LSTMConfig, fit_predict
from .splits import walk_forward_folds

COSTS_BPS = (0.0, 2.0, 5.0, 10.0, 20.0)
BASE_COST_BPS = 5.0  # 2 bps commission + 3 bps slippage per unit of turnover


@dataclass(frozen=True)
class ForecastConfig:
    window: int = 20
    min_train: int = 2000
    test_size: int = 500
    val_size: int = 250
    seeds: tuple[int, ...] = (0, 1, 2)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)


@dataclass(frozen=True)
class TradeConfig:
    window: int = 20
    train_len: int = 1500  # rolling training window, in days
    val_len: int = 250
    seeds: tuple[int, ...] = (0, 1, 2)
    dqn: DQNConfig = field(default_factory=lambda: DQNConfig(epochs=20))
    cost_bps: float = BASE_COST_BPS


def load_returns(data_dir: Path, asset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(log returns, simple returns, prices aligned with the returns) for one asset."""
    df = load_asset(Path(data_dir) / "processed", asset)
    log_r = df["log_ret"].to_numpy()
    return log_r, np.expm1(log_r), df["price"].to_numpy()


def forecast_folds(n_returns: int, cfg: ForecastConfig):
    n_samples = n_returns - cfg.window
    return list(walk_forward_folds(n_samples, cfg.min_train, cfg.test_size, cfg.val_size))


def forecast_block(log_r: np.ndarray, fold_idx: int, cfg: ForecastConfig) -> dict[str, np.ndarray]:
    """All forecasters for one test block. Keys: day, y, zero, last, drift, ses, arima,
    lstm_ret_s{seed}, lstm_prob_s{seed}."""
    x, y, target_day = build_supervised(log_r, cfg.window)
    fold = forecast_folds(len(log_r), cfg)[fold_idx]
    te = np.array(fold.test)
    days = target_day[te]
    fit_end = int(days[0])  # every day before the first test target is available
    out: dict[str, np.ndarray] = {"day": days, "y": y[te]}
    out["zero"] = bl.zero_forecast(len(te))
    out["last"] = log_r[days - 1]
    out["drift"] = np.cumsum(log_r)[days - 1] / days
    alpha = bl.fit_ses_alpha(log_r[:fit_end])
    out["ses"] = bl.ses_forecast(log_r[: int(days[-1]) + 1], alpha)[days - 1]
    out["arima"] = bl.arima_forecast(log_r, fit_end)[days - fit_end]
    tr, va = np.array(fold.train), np.array(fold.val)
    for s in cfg.seeds:
        o = fit_predict(x[tr], y[tr], x[va], y[va], x[te], cfg.lstm, seed=s)
        out[f"lstm_ret_s{s}"] = o.ret_pred
        out[f"lstm_prob_s{s}"] = o.prob_up
    return out


def trade_blocks(n_returns: int, cfg: ForecastConfig) -> list[tuple[int, int]]:
    """The same test blocks as the forecasting experiment, as [start_day, end_day) return indices."""
    blocks = []
    for f in forecast_folds(n_returns, cfg):
        blocks.append((f.test[0] + cfg.window, f.test[-1] + 1 + cfg.window))
    return blocks


def trade_block(
    simple_r: np.ndarray, block: tuple[int, int], seed: int, cfg: TradeConfig
) -> dict[str, np.ndarray]:
    """Train a DQN on the rolling window before ``block`` and run it greedily on the block."""
    t0, t1 = block
    v0 = t0 - cfg.val_len
    tr0 = max(cfg.window, v0 - cfg.train_len)
    seg = simple_r[tr0:v0]
    mean, scale = float(seg.mean()), float(seg.std()) or 1.0
    kw = {"window": cfg.window, "commission_bps": cfg.cost_bps, "slippage_bps": 0.0,
          "obs_mean": mean, "obs_scale": scale}
    train_env = TradingEnv(simple_r, start=tr0, end=v0, **kw)
    val_env = TradingEnv(simple_r, start=v0, end=t0, **kw)
    agent = train_dqn(train_env, val_env, seed=seed, cfg=cfg.dqn)
    test_env = TradingEnv(simple_r, start=t0, end=t1, **kw)
    base = evaluate_policy(agent, test_env)
    out = {"day": np.arange(t0, t1), "positions": base["positions"]}
    for c in COSTS_BPS:
        out[f"net_{c:g}"] = evaluate_policy(agent, test_env.with_cost_bps(c))["net_returns"]
    return out
