"""Performance and forecast-error metrics. All return series are simple (arithmetic) returns."""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def equity_curve(simple_returns: np.ndarray) -> np.ndarray:
    """Compounded equity starting at 1.0; length is len(returns) + 1."""
    r = np.asarray(simple_returns, dtype=float)
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Drawdown from the running peak, as non-positive fractions."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    return eq / peak - 1.0


def max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough fall as a non-positive fraction (-0.6 means a 60% fall)."""
    return float(drawdown_series(equity).min())


def sharpe(simple_returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate (mean / sample std * sqrt(periods))."""
    r = np.asarray(simple_returns, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def annualised_return(simple_returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Geometric annualised return."""
    r = np.asarray(simple_returns, dtype=float)
    if len(r) == 0:
        raise ValueError("annualised_return needs at least one return")
    growth = float(np.prod(1.0 + r))
    return growth ** (periods / len(r)) - 1.0


def annualised_vol(simple_returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(simple_returns, dtype=float)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else 0.0


def summarise_strategy(net_returns: np.ndarray) -> dict[str, float]:
    """Headline numbers for one strategy run on one test segment."""
    eq = equity_curve(net_returns)
    return {
        "ann_return": annualised_return(net_returns),
        "sharpe": sharpe(net_returns),
        "max_drawdown": max_drawdown(eq),
        "final_equity": float(eq[-1]),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of days where sign(pred) == sign(truth); days with truth == 0 are dropped."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = y != 0
    if not keep.any():
        raise ValueError("no non-zero targets")
    return float(np.mean(np.sign(p[keep]) == np.sign(y[keep])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(d)))


def mse_skill(y_true: np.ndarray, y_pred: np.ndarray, y_bench: np.ndarray) -> float:
    """1 - MSE(model)/MSE(benchmark): positive means better than the benchmark."""
    y = np.asarray(y_true, dtype=float)
    mse_m = np.mean((y - np.asarray(y_pred, dtype=float)) ** 2)
    mse_b = np.mean((y - np.asarray(y_bench, dtype=float)) ** 2)
    return float(1.0 - mse_m / mse_b)
