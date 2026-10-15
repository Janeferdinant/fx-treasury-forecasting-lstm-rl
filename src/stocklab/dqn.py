"""A small CPU DQN (double-DQN target, Huber loss) for the trading environment."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .env import TradingEnv
from .metrics import sharpe


@dataclass(frozen=True)
class DQNConfig:
    hidden: int = 64
    gamma: float = 0.9
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 20_000
    epochs: int = 30  # passes over the training segment
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6  # fraction of all steps over which epsilon decays linearly
    target_every: int = 250
    train_every: int = 4
    learn_start: int = 500
    reward_scale: float = 100.0  # rewards are daily returns; scale to percent for the network


class QAgent:
    """Greedy policy wrapper around a trained Q-network."""

    def __init__(self, net: nn.Module) -> None:
        self.net = net.eval()

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(self.net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(1))


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_out),
    )


def evaluate_policy(agent: QAgent, env: TradingEnv) -> dict[str, np.ndarray]:
    """Run the greedy policy once over the environment's segment."""
    obs, _ = env.reset(seed=0)
    rewards, positions, turnover, cost = [], [], [], []
    done = False
    while not done:
        obs, r, done, _, info = env.step(agent.act(obs))
        rewards.append(r)
        positions.append(info["position"])
        turnover.append(info["turnover"])
        cost.append(info["cost"])
    return {
        "net_returns": np.array(rewards),
        "positions": np.array(positions),
        "turnover": np.array(turnover),
        "cost": np.array(cost),
    }


def train_dqn(
    train_env: TradingEnv, val_env: TradingEnv, seed: int = 0, cfg: DQNConfig | None = None
) -> QAgent:
    """Train on ``train_env``; after every epoch score the greedy policy on ``val_env`` and keep
    the weights with the best validation Sharpe. The test segment is never touched here."""
    cfg = cfg or DQNConfig()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_obs = train_env.observation_space.shape[0]
    n_act = train_env.action_space.n
    q = _mlp(n_obs, n_act, cfg.hidden)
    q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    cap = cfg.buffer_size
    buf_s = np.zeros((cap, n_obs), dtype=np.float32)
    buf_s2 = np.zeros((cap, n_obs), dtype=np.float32)
    buf_a = np.zeros(cap, dtype=np.int64)
    buf_r = np.zeros(cap, dtype=np.float32)
    buf_d = np.zeros(cap, dtype=np.float32)
    size = ptr = step_count = 0
    total_steps = cfg.epochs * train_env.n_steps
    decay_steps = max(1, int(cfg.eps_decay_frac * total_steps))

    best_score, best_state = -np.inf, copy.deepcopy(q.state_dict())
    for epoch in range(cfg.epochs):
        obs, _ = train_env.reset(seed=seed + epoch)
        done = False
        while not done:
            eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step_count / decay_steps)
            if rng.random() < eps:
                a = int(rng.integers(n_act))
            else:
                with torch.no_grad():
                    a = int(q(torch.as_tensor(obs).unsqueeze(0)).argmax(1))
            obs2, r, done, _, _ = train_env.step(a)
            buf_s[ptr], buf_a[ptr], buf_r[ptr], buf_s2[ptr], buf_d[ptr] = (
                obs, a, r * cfg.reward_scale, obs2, float(done),
            )
            ptr = (ptr + 1) % cap
            size = min(size + 1, cap)
            obs = obs2
            step_count += 1
            if size >= cfg.learn_start and step_count % cfg.train_every == 0:
                idx = rng.integers(0, size, cfg.batch_size)
                s = torch.as_tensor(buf_s[idx])
                s2 = torch.as_tensor(buf_s2[idx])
                aa = torch.as_tensor(buf_a[idx])
                rr = torch.as_tensor(buf_r[idx])
                dd = torch.as_tensor(buf_d[idx])
                with torch.no_grad():
                    a2 = q(s2).argmax(1, keepdim=True)
                    target = rr + cfg.gamma * (1 - dd) * q_target(s2).gather(1, a2).squeeze(1)
                pred = q(s).gather(1, aa.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step_count % cfg.target_every == 0:
                q_target.load_state_dict(q.state_dict())
        score = sharpe(evaluate_policy(QAgent(q), val_env)["net_returns"])
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(q.state_dict())
    q.load_state_dict(best_state)
    return QAgent(q)
