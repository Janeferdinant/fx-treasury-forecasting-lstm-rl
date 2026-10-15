"""Download the public-domain FRED series, build the processed price series and check them against
the committed manifest.

Raw files go to ``data/raw`` and processed series to ``data/processed`` (both git-ignored). The
committed ``data/manifest.json`` holds the SHA-256 of each processed series, so a changed or
revised upstream file is detected instead of silently producing different results.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from . import data_sources as ds
from . import rates

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2000-01-01&coed=2025-12-31"
UA = {"User-Agent": "Mozilla/5.0 (stocklab research code; see the repository README)"}


def download(
    url: str, cache: Path, name: str, refresh: bool = False, retries: int = 4,
    opener=urllib.request.urlopen, sleeper=time.sleep,
) -> bytes:
    """Fetch ``url`` to ``cache/name`` (cached), retrying with growing waits; FRED throttles bursts."""
    path = Path(cache) / name
    if path.exists() and not refresh:
        return path.read_bytes()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with opener(urllib.request.Request(url, headers=UA), timeout=60) as resp:
                blob = resp.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            return blob
        except Exception as exc:  # network errors of several kinds
            last = exc
            if attempt < retries - 1:
                sleeper(5 * 3**attempt)
    raise RuntimeError(f"could not download {name} after {retries} attempts: {last}")


def build_all(
    raw_dir: Path, refresh: bool = False, fetcher=download, start: str = ds.START, end: str = ds.END,
    min_obs: int = 1000,
) -> dict[str, pd.Series]:
    """Download (or read from the cache) every series and return the processed price series."""
    series: dict[str, pd.Series] = {}
    for name, spec in ds.ASSETS.items():
        url = FRED_CSV.format(sid=spec.fred_id)
        blob = fetcher(url, Path(raw_dir), f"fred_{spec.fred_id}.csv", refresh)
        s = ds.clip_range(ds.parse_fred_csv(blob.decode("utf-8")), start, end)
        if spec.kind == "yield":
            s = rates.par_bond_index(s, spec.maturity)
        ds.validate_prices(s, name, min_obs=min_obs)
        series[name] = s.rename("price")
    return series


def write_processed(series: dict[str, pd.Series], out_dir: Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, s in series.items():
        text = ds.canonical_csv(s)
        (out_dir / f"{name}.csv").write_text(text, encoding="utf-8", newline="\n")
        manifest[name] = ds.sha256_text(text)
    return manifest


def run(data_dir: Path, write_manifest: bool = False, refresh: bool = False) -> dict[str, dict]:
    """Fetch, build, write and verify. Returns per-asset info (rows, first/last date, sha256)."""
    data_dir = Path(data_dir)
    series = build_all(data_dir / "raw", refresh)
    manifest = write_processed(series, data_dir / "processed")
    info = {
        n: {"fred_id": ds.ASSETS[n].fred_id if n in ds.ASSETS else None, "rows": int(len(s)),
            "first": f"{s.index[0]:%Y-%m-%d}", "last": f"{s.index[-1]:%Y-%m-%d}", "sha256": manifest[n]}
        for n, s in series.items()
    }
    path = data_dir / "manifest.json"
    if write_manifest:
        path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        expected = json.loads(path.read_text(encoding="utf-8"))
        bad = [n for n in expected if info.get(n, {}).get("sha256") != expected[n]["sha256"]]
        if bad:
            raise SystemExit(
                f"checksum mismatch for {bad}: the upstream data differs from the committed manifest "
                "(possible revision). Results will differ; see docs/runs for the original checksums."
            )
    return info
