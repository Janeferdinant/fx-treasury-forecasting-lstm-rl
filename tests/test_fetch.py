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


def _ddp_text(ids: list[str], level: dict[str, float], n: int = 300, seed: int = 0) -> str:
    idx = pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(seed)
    cols = {i: level[i] * np.cumprod(1 + rng.normal(0, 0.003, n)) for i in ids}
    head = [
        '"Series Description",' + ",".join(f'"{i}"' for i in ids),
        '"Unit:",' + ",".join('"x"' for _ in ids),
        '"Multiplier:",' + ",".join('"1"' for _ in ids),
        '"Currency:",' + ",".join('"NA"' for _ in ids),
        '"Unique Identifier: ",' + ",".join(f'"H10/H10/{i}"' for i in ids),
        '"Time Period",' + ",".join(f'"{i}"' for i in ids),
    ]
    rows = []
    for k, d in enumerate(idx):
        cells = ["ND" if k == 7 else f"{cols[i][k]:.4f}" for i in ids]
        rows.append(f"{d:%Y-%m-%d}," + ",".join(cells))
    return "\n".join(head + rows) + "\n"


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


def test_build_all_makes_every_asset_from_the_federal_reserve_files(tmp_path):
    h10_ids = [a.source_key for a in ds.ASSETS.values() if a.provider == "h10"]
    h15_ids = [a.source_key for a in ds.ASSETS.values() if a.provider == "h15"]
    texts = {
        fetch.H10_URL: _ddp_text(h10_ids, dict.fromkeys(h10_ids, 1.2)).encode(),
        fetch.H15_URL: _ddp_text(h15_ids, dict.fromkeys(h15_ids, 4.0)).encode(),
    }

    def fake_fetch(url, cache, name, refresh=False, **kw):
        return texts[url]

    series = fetch.build_all(tmp_path, fetcher=fake_fetch, start="2000-01-03", end="2025-12-31", min_obs=100)
    assert set(series) == set(ds.ASSETS)
    for name, s in series.items():
        assert (s > 0).all() and s.index.is_monotonic_increasing, name
    assert 50 < series["UST10Y"].iloc[0] < 150  # a total-return index built from yields


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
