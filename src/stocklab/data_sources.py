"""Asset registry, parser for the Federal Reserve source files, and checksum helpers.

Every series is hosted by the Federal Reserve Board's Data Download Program:
  * H.10 daily exchange rates (noon buying rates in New York, certified by the New York Fed);
  * H.15 Treasury constant-maturity yields (interpolated by the U.S. Treasury from its yield curve).
The terms checked at source are recorded in THIRD_PARTY_NOTICES.md. EIA oil and gas spot prices
were considered and left out: EIA documents their source as a commercial data vendor.
"""

from __future__ import annotations

import csv
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
    group: str  # fx | rates
    provider: str  # h10 | h15
    source_key: str  # Fed Data Download series identifier
    source: str
    kind: str = "level"  # level: the series is a price; yield: build a par-bond return index
    maturity: float | None = None


ASSETS: dict[str, AssetSpec] = {
    a.name: a
    for a in [
        AssetSpec("EURUSD", "EUR/USD (USD per EUR)", "fx", "h10", "RXI$US_N.B.EU", "Federal Reserve H.10"),
        AssetSpec("USDJPY", "USD/JPY (JPY per USD)", "fx", "h10", "RXI_N.B.JA", "Federal Reserve H.10"),
        AssetSpec("GBPUSD", "GBP/USD (USD per GBP)", "fx", "h10", "RXI$US_N.B.UK", "Federal Reserve H.10"),
        AssetSpec("USDCAD", "USD/CAD (CAD per USD)", "fx", "h10", "RXI_N.B.CA", "Federal Reserve H.10"),
        AssetSpec("AUDUSD", "AUD/USD (USD per AUD)", "fx", "h10", "RXI$US_N.B.AL", "Federal Reserve H.10"),
        AssetSpec("USDCHF", "USD/CHF (CHF per USD)", "fx", "h10", "RXI_N.B.SZ", "Federal Reserve H.10"),
        AssetSpec("UST2Y", "2-year Treasury par-bond total return (from yields)", "rates", "h15",
                  "RIFLGFCY02_N.B", "Federal Reserve H.15", kind="yield", maturity=2.0),
        AssetSpec("UST10Y", "10-year Treasury par-bond total return (from yields)", "rates", "h15",
                  "RIFLGFCY10_N.B", "Federal Reserve H.15", kind="yield", maturity=10.0),
    ]
}


def parse_fed_ddp(text: str, series_id: str) -> pd.Series:
    """One series from a Federal Reserve Data Download Program CSV (the 'seriescolumn' layout).

    The column is picked by its unique identifier (for example ``RXI_N.B.JA``). Cells marked ``ND``
    (no data, such as holidays) or left blank are dropped.
    """
    rows = list(csv.reader(io.StringIO(text)))
    uid = next((i for i, r in enumerate(rows) if r and r[0].strip() == "Unique Identifier:"), None)
    head = next((i for i, r in enumerate(rows) if r and r[0].strip() == "Time Period"), None)
    if uid is None or head is None:
        raise ValueError("not a Federal Reserve data-download file")
    ids = [c.strip().split("/")[-1] for c in rows[uid][1:]]
    if series_id not in ids:
        raise KeyError(series_id)
    col = ids.index(series_id) + 1
    dates, vals = [], []
    for r in rows[head + 1 :]:
        if len(r) > col and r[0].strip():
            dates.append(r[0].strip())
            vals.append(r[col].strip())
    s = pd.Series(pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(),
                  index=pd.to_datetime(pd.Series(dates)), name="price").dropna()
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
