"""A small Gymnasium-style single-asset trading environment with transaction costs and slippage.

Timing: at the close of day t the agent sees the last ``window`` returns r[t-window+1 .. t] and its
current position, and picks a target position. That position is held over day t+1 and earns r[t+1].
Costs are charged on turnover |new - old| at ``commission_bps + slippage_bps`` basis points per unit.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DEFAULT_POSITIONS = (-1.0, 0.0, 1.0)  # sell/short, hold flat, buy/long


def simulate_positions(
    positions: np.ndarray, simple_returns: np.ndarray, cost_bps: float, initial_position: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Net returns for a fixed sequence of daily positions (position[k] is held on day k).

    Returns (net_returns, turnover, cost) per day. This is the same accounting as the environment.
    """
    pos = np.asarray(positions, dtype=float)
    r = np.asarray(simple_returns, dtype=float)
    prev = np.concatenate([[initial_position], pos[:-1]])
    turnover = np.abs(pos - prev)
    cost = turnover * cost_bps / 1e4
    return pos * r - cost, turnover, cost


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,
        window: int = 20,
        positions: tuple[float, ...] = DEFAULT_POSITIONS,
        commission_bps: float = 2.0,
        slippage_bps: float = 3.0,
        start: int | None = None,
        end: int | None = None,
        obs_mean: float = 0.0,
        obs_scale: float = 1.0,
    ) -> None:
        super().__init__()
        r = np.asarray(returns, dtype=float)
        self.start = window if start is None else start
        self.end = len(r) if end is None else end
        if self.start < window:
            raise ValueError("start must leave `window` days of history before the first decision")
        if self.end > len(r) or self.end <= self.start:
            raise ValueError("segment [start, end) must lie inside the data and be non-empty")
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("costs must be non-negative")
        self.returns = r
        self.window = window
        self.positions = tuple(float(p) for p in positions)
        self.cost_rate = (commission_bps + slippage_bps) / 1e4
        self.obs_mean = obs_mean
        self.obs_scale = obs_scale
        self.action_space = spaces.Discrete(len(self.positions))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(window + 1,), dtype=np.float32)
        self._t = self.start - 1
        self._pos = 0.0
        self._equity = 1.0

    def with_cost_bps(self, total_bps: float) -> TradingEnv:
        """A copy of this environment charging ``total_bps`` per unit turnover (for sweeps)."""
        return TradingEnv(
            self.returns, self.window, self.positions, total_bps, 0.0, self.start, self.end,
            self.obs_mean, self.obs_scale,
        )

    @property
    def n_steps(self) -> int:
        return self.end - self.start

    def _obs(self) -> np.ndarray:
        w = self.returns[self._t - self.window + 1 : self._t + 1]
        feats = (w - self.obs_mean) / self.obs_scale
        return np.concatenate([feats, [self._pos]]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = self.start - 1
        self._pos = 0.0
        self._equity = 1.0
        return self._obs(), {"position": 0.0, "equity": 1.0}

    def step(self, action: int):
        new_pos = self.positions[int(action)]
        turnover = abs(new_pos - self._pos)
        cost = turnover * self.cost_rate
        reward = new_pos * float(self.returns[self._t + 1]) - cost
        self._t += 1
        self._pos = new_pos
        self._equity *= 1.0 + reward
        terminated = self._t + 1 >= self.end
        info = {
            "position": new_pos,
            "turnover": turnover,
            "cost": cost,
            "equity": self._equity,
        }
        return self._obs(), reward, terminated, False, info
