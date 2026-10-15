"""Baseline one-step-ahead return forecasters.

Convention: for a return series r[0..N-1], ``forecasts[i]`` is the prediction of r[i+1] made with
information up to and including day i (except ``arima_forecast``, see its docstring).
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def zero_forecast(n: int) -> np.ndarray:
    """Random walk on the price level: the best guess for the next return is zero."""
    return np.zeros(n)


def last_value_forecast(r: np.ndarray) -> np.ndarray:
    """Next return equals today's return (naive persistence in return space)."""
    return np.asarray(r, dtype=float)[:-1].copy()


def drift_forecast(r: np.ndarray) -> np.ndarray:
    """Next return equals the expanding mean of all returns seen so far (random walk with drift)."""
    a = np.asarray(r, dtype=float)
    return (np.cumsum(a) / np.arange(1, len(a) + 1))[:-1]


def ses_forecast(r: np.ndarray, alpha: float) -> np.ndarray:
    """ETS(A,N,N), i.e. simple exponential smoothing of returns; level initialised at r[0]."""
    a = np.asarray(r, dtype=float)
    out = np.empty(len(a) - 1)
    level = a[0]
    out[0] = level
    for t in range(1, len(a) - 1):
        level = alpha * a[t] + (1.0 - alpha) * level
        out[t] = level
    return out


def fit_ses_alpha(r: np.ndarray) -> float:
    """Pick alpha on a grid by minimising one-step squared error on the given (pre-test) data."""
    a = np.asarray(r, dtype=float)
    best, best_sse = 0.01, np.inf
    for alpha in np.concatenate([np.arange(0.01, 0.10, 0.01), np.arange(0.1, 1.0001, 0.05)]):
        f = ses_forecast(a, float(alpha))
        sse = float(np.sum((a[1:] - f) ** 2))
        if sse < best_sse:
            best, best_sse = float(alpha), sse
    return best


def arima_forecast(
    r: np.ndarray, fit_end: int, order: tuple[int, int, int] = (1, 0, 1)
) -> np.ndarray:
    """One-step-ahead ARIMA forecasts for days fit_end .. N-1.

    Parameters are estimated once on r[:fit_end]. The model is then applied to the whole series
    with those parameters frozen, so the forecast for day t is a filter output that depends only on
    days before t. Element i of the result targets day ``fit_end + i``.
    (ARIMA(1,1,1) on log prices is ARMA(1,1) with a constant on returns, which is what is fitted.)
    """
    a = np.asarray(r, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(a[:fit_end], order=order, trend="c").fit()
        full = res.apply(a, refit=False)
        pred = full.get_prediction(start=fit_end, end=len(a) - 1, dynamic=False).predicted_mean
    return np.asarray(pred, dtype=float)
