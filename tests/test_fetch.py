import io
import json

import numpy as np
import pandas as pd
import pytest

from stocklab import data_sources as ds
from stocklab import fetch


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fred_text(sid: str, n: int = 300, seed: int = 0, level: float = 1.0) -> str:
    idx = pd.bdate_range("2000-01-03", periods=n)
    r = np.random.default_rng(seed).normal(0, 0.003, n)
    vals = level * np.cumprod(1 + r)
    lines = [f"observation_date,{sid}"] + [f"{d:%Y-%m-%d},{v:.4f}" for d, v in zip(idx, vals, strict=True)]
    lines[5] = f"{idx[4]:%Y-%m-%d},."  # one missing observation
    return "\n".join(lines) + "\n"


def test_download_retries_with_backoff_then_succeeds(tmp_path):
    calls, sleeps = [], []

    def opener(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("closed")
        return _Resp(b"ok")

    blob = fetch.download("http://x", tmp_path, "f.csv", retries=4, opener=opener, sleeper=sleeps.append)
    assert blob == b"ok" and len(calls) == 3
    assert sleeps == sorted(sleeps) and len(sleeps) == 2  # waits grow between attempts
    assert (tmp_path / "f.csv").read_bytes() == b"ok"


def test_download_uses_cache_and_gives_up_with_a_clear_error(tmp_path):
    (tmp_path / "f.csv").write_bytes(b"cached")

    def boom(req, timeout):
        raise OSError("blocked")

    assert fetch.download("http://x", tmp_path, "f.csv", opener=boom) == b"cached"
    with pytest.raises(RuntimeError, match="could not download"):
        fetch.download("http://x", tmp_path, "g.csv", retries=2, opener=boom, sleeper=lambda s: None)


def test_build_all_makes_every_asset_from_fred_files(tmp_path):
    def fake_fetch(url, cache, name, refresh=False, **kw):
        sid = name.removeprefix("fred_").removesuffix(".csv")
        level = 4.0 if ds.ASSETS_BY_FRED[sid].kind == "yield" else 1.0
        return _fred_text(sid, level=level).encode()

    series = fetch.build_all(tmp_path, fetcher=fake_fetch, start="2000-01-03", end="2025-12-31", min_obs=100)
    assert set(series) == set(ds.ASSETS)
    for name, s in series.items():
        assert (s > 0).all() and s.index.is_monotonic_increasing, name
    # the yield asset is a total-return index built from yields, so it starts near 100
    assert 50 < series["UST10Y"].iloc[0] < 150


def test_write_processed_checksums_match_files(tmp_path):
    idx = pd.bdate_range("2000-01-03", periods=150)
    s = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 150)), index=idx)
    manifest = fetch.write_processed({"A": s}, tmp_path)
    assert ds.verify_manifest(tmp_path, manifest) == []
    (tmp_path / "A.csv").write_text("date,price\n2000-01-03,1.0\n", encoding="utf-8")
    assert ds.verify_manifest(tmp_path, manifest) == ["A"]


def test_run_detects_changed_upstream_data(tmp_path, monkeypatch):
    def make(seed):
        idx = pd.bdate_range("2000-01-03", periods=150)
        s = 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0, 0.01, 150))
        return {"A": pd.Series(s, index=idx, name="price")}

    monkeypatch.setattr(fetch, "build_all", lambda raw_dir, refresh=False, **kw: make(1))
    info = fetch.run(tmp_path, write_manifest=True)
    assert json.loads((tmp_path / "manifest.json").read_text())["A"]["sha256"] == info["A"]["sha256"]
    fetch.run(tmp_path)
    monkeypatch.setattr(fetch, "build_all", lambda raw_dir, refresh=False, **kw: make(2))
    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch.run(tmp_path)
