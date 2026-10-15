"""Baseline forecasters: alignment and absence of look-ahead."""

import numpy as np
import pytest

from stocklab import baselines as b


def _r(n=400, seed=0, phi=0.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


def test_zero_forecast_is_all_zeros():
    assert (b.zero_forecast(10) == 0).all()


def test_last_value_forecast_is_previous_return():
    r = np.array([0.1, -0.2, 0.3, 0.4])
    # forecast for day t+1 is r[t]; forecasts[i] targets day i+1
    np.testing.assert_array_equal(b.last_value_forecast(r), r[:-1])


def test_drift_is_expanding_mean_with_no_look_ahead():
    r = _r()
    f = b.drift_forecast(r)
    assert f.shape == (len(r) - 1,)
    assert f[10] == pytest.approx(r[:11].mean())
    r2 = r.copy()
    r2[200:] += 1.0
    f2 = b.drift_forecast(r2)
    np.testing.assert_allclose(f[:199], f2[:199])


def test_ses_alpha_fit_prefers_high_alpha_for_persistent_series():
    r = _r(n=3000, phi=0.8)
    assert b.fit_ses_alpha(r) > b.fit_ses_alpha(_r(n=3000, phi=0.0))


def test_ses_forecast_uses_only_past_and_matches_recursion():
    r = _r()
    alpha = 0.3
    f = b.ses_forecast(r, alpha)
    level = r[0]
    manual = []
    for t in range(len(r) - 1):
        level = alpha * r[t] + (1 - alpha) * level if t > 0 else r[0]
        manual.append(level)
    np.testing.assert_allclose(f, manual)
    r2 = r.copy()
    r2[250:] = 5.0
    np.testing.assert_allclose(b.ses_forecast(r2, alpha)[:249], f[:249])


def test_arima_forecast_recovers_ar1_signal_out_of_sample():
    r = _r(n=3000, phi=0.5, seed=4)
    train, test_start = r[:2000], 2000
    f = b.arima_forecast(r, fit_end=2000, order=(1, 0, 0))
    assert f.shape == (len(r) - test_start,)
    corr = np.corrcoef(f, r[test_start:])[0, 1]
    assert corr > 0.3
    assert len(train) == 2000


def test_arima_forecast_has_no_look_ahead_in_test_block():
    r = _r(n=700, phi=0.3, seed=9)
    f = b.arima_forecast(r, fit_end=500, order=(1, 0, 1))
    r2 = r.copy()
    r2[600:] += 0.5
    f2 = b.arima_forecast(r2, fit_end=500, order=(1, 0, 1))
    # forecast i targets day fit_end+i; it may only use days < fit_end+i
    np.testing.assert_allclose(f[: 600 - 500], f2[: 600 - 500], atol=1e-10)
    assert not np.allclose(f, f2)


def test_arima_params_depend_only_on_fit_window():
    r = _r(n=700, phi=0.3, seed=9)
    r2 = r.copy()
    r2[500:] = np.random.default_rng(1).normal(0, 0.2, 200)
    f = b.arima_forecast(r, fit_end=500, order=(1, 0, 0))
    f2 = b.arima_forecast(r2, fit_end=500, order=(1, 0, 0))
    assert f[0] == pytest.approx(f2[0])  # first forecast uses only data before the block
