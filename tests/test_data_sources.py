"""Parsers, asset registry and checksum manifest, on tiny inline fixtures (no network)."""

import numpy as np
import pandas as pd
import pytest

from stocklab import data_sources as ds

FRED = "observation_date,DEXUSEU\n2000-01-03,1.0155\n2000-01-04,.\n2000-01-05,1.0335\n"


def test_fred_parser_drops_missing_markers():
    s = ds.parse_fred_csv(FRED)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2000-01-03", "2000-01-05"]
    assert s.iloc[1] == pytest.approx(1.0335)


def test_fred_parser_rejects_files_without_two_columns():
    with pytest.raises(ValueError):
        ds.parse_fred_csv("observation_date\n2000-01-03\n")


def test_fred_parser_sorts_and_rejects_duplicate_dates():
    s = ds.parse_fred_csv("observation_date,X\n2000-01-05,2.0\n2000-01-03,1.0\n")
    assert list(s.values) == [1.0, 2.0]
    with pytest.raises(ValueError):
        ds.parse_fred_csv("observation_date,X\n2000-01-03,1.0\n2000-01-03,2.0\n")


def test_asset_registry_uses_only_documented_public_domain_series():
    assert len(ds.ASSETS) >= 6
    for spec in ds.ASSETS.values():
        assert spec.fred_id and spec.source and spec.group in {"fx", "commodity", "rates"}
        assert spec.kind in {"level", "yield"}
    assert ds.ASSETS["UST10Y"].kind == "yield" and ds.ASSETS["UST10Y"].maturity == 10.0


def test_returns_to_index_starts_at_100_and_compounds():
    idx = ds.returns_to_index(pd.Series([0.1, -0.1], index=pd.to_datetime(["2000-01-03", "2000-01-04"])))
    assert idx.iloc[0] == pytest.approx(110.0)
    assert idx.iloc[1] == pytest.approx(99.0)


def test_clip_range_is_inclusive_and_sorted():
    s = pd.Series([3.0, 1.0, 2.0], index=pd.to_datetime(["2000-01-05", "2000-01-03", "2000-01-04"]))
    out = ds.clip_range(s, "2000-01-04", "2000-01-05")
    assert list(out.values) == [2.0, 3.0]


def test_canonical_csv_and_hash_are_deterministic():
    s = pd.Series([1.0, 2.5], index=pd.to_datetime(["2000-01-03", "2000-01-04"]))
    a = ds.canonical_csv(s)
    assert a == ds.canonical_csv(s.copy())
    assert a.splitlines()[0] == "date,price"
    assert ds.sha256_text(a) != ds.sha256_text(a + " ")


def test_non_positive_prices_are_rejected():
    s = pd.Series([1.0, -0.5], index=pd.to_datetime(["2000-01-03", "2000-01-04"]))
    with pytest.raises(ValueError):
        ds.validate_prices(s, "X")


def test_too_short_or_unordered_series_are_rejected():
    idx = pd.bdate_range("2000-01-03", periods=120)
    with pytest.raises(ValueError):
        ds.validate_prices(pd.Series(np.ones(50), index=idx[:50]), "short")
    with pytest.raises(ValueError):
        ds.validate_prices(pd.Series(np.ones(120), index=idx[::-1]), "unordered")


def test_manifest_verification(tmp_path):
    (tmp_path / "A.csv").write_text("date,price\n2000-01-03,1.0\n")
    good = {"A": ds.sha256_text((tmp_path / "A.csv").read_text())}
    assert ds.verify_manifest(tmp_path, good) == []
    assert ds.verify_manifest(tmp_path, {"A": "0" * 64}) == ["A"]
    assert ds.verify_manifest(tmp_path, {"B": "0" * 64}) == ["B"]


def test_load_returns_from_processed_csv(tmp_path):
    (tmp_path / "X.csv").write_text("date,price\n2000-01-03,100.0\n2000-01-04,110.0\n2000-01-05,99.0\n")
    df = ds.load_asset(tmp_path, "X")
    np.testing.assert_allclose(df["log_ret"].to_numpy(), np.diff(np.log([100.0, 110.0, 99.0])))
    assert len(df) == 2
