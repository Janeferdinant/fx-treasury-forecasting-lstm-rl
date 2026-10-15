"""Download the Federal Reserve source files, build the processed price series and check them against
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

DDP = "https://www.federalreserve.gov/datadownload/Output.aspx?rel={rel}&series={series}"
H10_URL = DDP.format(rel="H10", series="60f32914ab61dfab590e0e470153e3ae") + (
    "&lastobs=&from=01/01/2000&to=12/31/2025&filetype=csv&label=include&layout=seriescolumn&type=package"
)
H15_URL = DDP.format(rel="H15", series="bf17364827e38702b42a58cf8eaa3f78") + (
    "&lastobs=&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package"  # all observations
)
UA = {"User-Agent": "Mozilla/5.0 (stocklab research code; see the repository README)"}


def download(
    url: str, cache: Path, name: str, refresh: bool = False, retries: int = 4,
    opener=urllib.request.urlopen, sleeper=time.sleep,
) -> bytes:
    """Fetch ``url`` to ``cache/name`` (cached), retrying with growing waits if the server throttles."""
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
    """Download (or read from the cache) every source file and return the processed price series."""
    raw_dir = Path(raw_dir)
    h10 = fetcher(H10_URL, raw_dir, "fed_h10_daily_rates.csv", refresh).decode("utf-8")
    h15 = fetcher(H15_URL, raw_dir, "fed_h15_treasury_constant_maturities.csv", refresh).decode("utf-8")
    series: dict[str, pd.Series] = {}
    for name, spec in ds.ASSETS.items():
        s = ds.parse_fed_ddp(h10 if spec.provider == "h10" else h15, spec.source_key)
        s = ds.clip_range(s, start, end)
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
        n: {"source": ds.ASSETS[n].source if n in ds.ASSETS else None, "rows": int(len(s)),
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
