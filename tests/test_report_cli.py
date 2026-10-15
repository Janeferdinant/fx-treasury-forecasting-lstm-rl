import json

import numpy as np
import pandas as pd
import pytest

from stocklab import cli, fetch, report


def _setup(tmp_path, seed=0):
    n_days = 900
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    rng = np.random.default_rng(seed)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days)), index=idx, name="price")
    fetch.write_processed({"EURUSD": prices}, tmp_path / "processed")
    days = np.arange(600, 880)
    y = np.diff(np.log(prices.to_numpy()))[days - 1]
    raw = {"day": days, "y": y, "zero": np.zeros(len(days)), "last": np.roll(y, 1),
           "drift": np.full(len(days), 0.0003), "ses": np.full(len(days), 0.0002),
           "arima": np.full(len(days), 0.0001), "lstm_ret_s0": np.full(len(days), 0.0002),
           "lstm_prob_s0": np.full(len(days), 0.52)}
    report.save_npz(tmp_path / "raw" / "forecast_EURUSD.npz", raw)
    return prices, days


def _add_trades(tmp_path, days):
    trade = {"0/day": days, "0/positions": np.ones(len(days))}
    for c in (0, 2, 5, 10, 20):
        trade[f"0/net_{c}"] = np.full(len(days), 0.0001)
    report.save_npz(tmp_path / "raw" / "trade_EURUSD.npz", trade)


def test_save_npz_roundtrip_types(tmp_path):
    arrays = {"day": np.arange(5), "y": np.linspace(-0.01, 0.01, 5), "0/day": np.arange(5)}
    path = tmp_path / "x.npz"
    report.save_npz(path, arrays)
    with np.load(path) as z:
        assert z["day"].dtype == np.int64 and z["0/day"].dtype == np.int64
        assert z["y"].dtype == np.float32
        assert np.allclose(z["y"], arrays["y"], atol=1e-7)


def test_smoke_uses_only_the_last_blocks():
    assert cli._block_ids(10, smoke=True) == [8, 9]
    assert cli._block_ids(10, smoke=False) == list(range(10))
    assert cli._block_ids(1, smoke=True) == [0]


def test_report_without_results_exits(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["report", "--raw-dir", str(tmp_path), "--data-dir", str(tmp_path)])


def test_report_build_end_to_end_on_synthetic_data(tmp_path):
    _, days = _setup(tmp_path)
    summary = report.build(tmp_path, tmp_path / "raw", tmp_path / "summary.json", tmp_path / "data.json")
    entry = summary["assets"]["EURUSD"]
    assert entry["n_test_days"] == len(days)
    assert set(entry["strategies"]) == {"buy_and_hold", "ma_crossover", "lstm_direction",
                                        "lstm_return_sign", "flat"}
    assert entry["share_long_days"]["buy_and_hold"] == 1.0
    data = json.loads((tmp_path / "data.json").read_text())
    a = data["assets"]["EURUSD"]
    assert len(a["dates"]) == len(a["equity"]["buy_and_hold"]) == len(a["drawdown"]["buy_and_hold"])
    assert a["equity"]["flat"][0] == 100.0 and a["equity"]["flat"][-1] == 100.0
    assert a["drawdown"]["flat"] == [0.0] * len(a["dates"])


def test_explorer_data_has_no_raw_prices_or_underlying_returns(tmp_path):
    _, days = _setup(tmp_path)
    report.build(tmp_path, tmp_path / "raw", tmp_path / "summary.json", tmp_path / "data.json")
    text = (tmp_path / "data.json").read_text()
    data = json.loads(text)["assets"]["EURUSD"]
    assert set(data) == {"label", "group", "dates", "equity", "drawdown", "strategies", "forecast",
                         "dqn_cost", "share_long_days", "share_short_days", "turnover", "signal_cost_sweep"}
    for forbidden in ('"price', '"prices', '"returns', '"y"', '"log_ret'):
        assert forbidden not in text
    # every curve starts from the index base of 100, never from a price level
    assert all(curve[0] == 100.0 for curve in data["equity"].values())
    # curves are thinned (weekly), so the daily return series cannot be read back from them
    assert all(len(curve) < len(days) for curve in data["equity"].values())


def test_tables_file_is_named_after_the_summary_and_never_clobbers(tmp_path):
    _, days = _setup(tmp_path, seed=1)
    _add_trades(tmp_path, days)
    report.build(tmp_path, tmp_path / "raw", tmp_path / "summary.json", None)
    assert (tmp_path / "tables.md").exists()
    first = (tmp_path / "tables.md").read_text()
    report.build(tmp_path, tmp_path / "raw", tmp_path / "smoke_summary.json", None)
    assert (tmp_path / "smoke_summary_tables.md").exists()
    assert (tmp_path / "tables.md").read_text() == first
