"""Small CPU LSTM forecasters for next-day return (regression) and direction (classification)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .features import Standardiser


@dataclass(frozen=True)
class LSTMConfig:
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 25
    patience: int = 4
    batch_size: int = 64


@dataclass(frozen=True)
class ForecastOutput:
    ret_pred: np.ndarray  # predicted next-day return, original units
    prob_up: np.ndarray  # predicted probability that the next-day return is positive


class _Net(nn.Module):
    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            1, cfg.hidden, cfg.layers, batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def _train_one(
    x_tr: torch.Tensor, y_tr: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
    classify: bool, cfg: LSTMConfig, seed: int,
) -> _Net:
    torch.manual_seed(seed)
    net = _Net(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if classify else nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    best_loss, best_state, bad = float("inf"), copy.deepcopy(net.state_dict()), 0
    n = len(x_tr)
    for _ in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            b = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(net(x_tr[b]), y_tr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(x_val), y_val))
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def fit_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_te: np.ndarray,
    cfg: LSTMConfig | None = None, seed: int = 0,
) -> ForecastOutput:
    """Fit on the training window, early-stop on the validation window, predict the test window.

    Inputs are standardised with statistics from the training rows only. The regression target is
    scaled by the training standard deviation and un-scaled again for the output.
    """
    cfg = cfg or LSTMConfig()
    torch.set_num_threads(1)
    sx = Standardiser.fit(x_tr)
    y_sd = float(np.std(y_tr)) or 1.0

    def to_t(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(sx.transform(a), dtype=torch.float32).unsqueeze(-1)

    xt, xv, xe = to_t(x_tr), to_t(x_val), to_t(x_te)
    reg = _train_one(
        xt, torch.tensor(y_tr / y_sd, dtype=torch.float32),
        xv, torch.tensor(y_val / y_sd, dtype=torch.float32), False, cfg, seed,
    )
    cls = _train_one(
        xt, torch.tensor(y_tr > 0, dtype=torch.float32),
        xv, torch.tensor(y_val > 0, dtype=torch.float32), True, cfg, seed + 10_000,
    )
    with torch.no_grad():
        ret = reg(xe).numpy().astype(float) * y_sd
        prob = torch.sigmoid(cls(xe)).numpy().astype(float)
    return ForecastOutput(ret_pred=ret, prob_up=prob)
