"""Asset registry, FRED CSV parsing and checksum helpers for the public-domain daily series.

Every series comes from the FRED CSV endpoint (no key). Their original publishers are U.S.
government agencies; the terms checked at source are recorded in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

START = "2000-01-03"
END = "2025-12-31"


@dataclass(frozen=True)
class AssetSpec:
    name: str
    label: str
    group: str  # fx | commodity | rates
    fred_id: str
    source: str
    kind: str = "level"  # level: the series is a price; yield: build a par-bond return index
    maturity: float | None = None


ASSETS: dict[str, AssetSpec] = {
    a.name: a
    for a in [
        AssetSpec("EURUSD", "EUR/USD (USD per EUR)", "fx", "DEXUSEU", "Board of Governors, H.10"),
        AssetSpec("USDJPY", "USD/JPY (JPY per USD)", "fx", "DEXJPUS", "Board of Governors, H.10"),
        AssetSpec("GBPUSD", "GBP/USD (USD per GBP)", "fx", "DEXUSUK", "Board of Governors, H.10"),
        AssetSpec("USDCAD", "USD/CAD (CAD per USD)", "fx", "DEXCAUS", "Board of Governors, H.10"),
        AssetSpec("BRENT", "Brent crude spot (USD per barrel)", "commodity", "DCOILBRENTEU", "U.S. EIA"),
        AssetSpec("HENRY_HUB", "Henry Hub natural gas spot (USD per MMBtu)", "commodity", "DHHNGSP", "U.S. EIA"),
        AssetSpec("UST10Y", "10-year Treasury par-bond total return (from yields)", "rates", "DGS10",
                  "Board of Governors, H.15", kind="yield", maturity=10.0),
    ]
}
ASSETS_BY_FRED: dict[str, AssetSpec] = {a.fred_id: a for a in ASSETS.values()}


def parse_fred_csv(text: str) -> pd.Series:
    """FRED ``fredgraph.csv``: a date column and one value column; '.' marks a missing observation."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] < 2:
        raise ValueError("expected a date column and a value column")
    s = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
        index=pd.to_datetime(df.iloc[:, 0]),
        name="price",
    ).dropna()
    if s.index.has_duplicates:
        raise ValueError("duplicate dates in the series")
    return s.sort_index()


def returns_to_index(decimal_returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Compound decimal daily returns into a total-return index (value after each day)."""
    return (base * (1.0 + decimal_returns).cumprod()).rename("price")


def clip_range(s: pd.Series, start: str = START, end: str = END) -> pd.Series:
    s = s.sort_index()
    return s.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def validate_prices(s: pd.Series, name: str, min_obs: int = 100) -> None:
    if len(s) < min_obs:
        raise ValueError(f"{name}: too few observations ({len(s)})")
    if not np.isfinite(s.to_numpy()).all() or (s.to_numpy() <= 0).any():
        raise ValueError(f"{name}: prices must be positive and finite")
    if not s.index.is_monotonic_increasing or s.index.has_duplicates:
        raise ValueError(f"{name}: dates must be unique and increasing")


def canonical_csv(s: pd.Series) -> str:
    """Deterministic text form used for the committed checksums."""
    lines = ["date,price"]
    lines += [f"{d:%Y-%m-%d},{v:.6f}" for d, v in zip(s.index, s.to_numpy(), strict=True)]
    return "\n".join(lines) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_manifest(directory: Path, manifest: dict[str, str]) -> list[str]:
    """Return the asset names whose processed CSV is missing or does not match its checksum."""
    bad = []
    for name, digest in manifest.items():
        p = Path(directory) / f"{name}.csv"
        if not p.exists() or sha256_text(p.read_text(encoding="utf-8")) != digest:
            bad.append(name)
    return bad


def load_asset(directory: Path, name: str) -> pd.DataFrame:
    """Processed price file -> DataFrame with columns price and log_ret (first day dropped)."""
    df = pd.read_csv(Path(directory) / f"{name}.csv", parse_dates=["date"], index_col="date")
    df["log_ret"] = np.log(df["price"]).diff()
    return df.dropna()
