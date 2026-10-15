"""Parsers, asset registry and checksum manifest, on tiny inline fixtures (no network)."""

import numpy as np
import pandas as pd
import pytest

from stocklab import data_sources as ds

H10 = '''"Series Description","Australian Dollar","Euro-Area Euro","Japanese Yen"
"Unit:","Currency:_Per_US$","Currency:_Per_US$","Currency:_Per_US$"
"Multiplier:","1","1","1"
"Currency:","AUD","EUR","JPY"
"Unique Identifier: ","H10/H10/RXI$US_N.B.AL","H10/H10/RXI$US_N.B.EU","H10/H10/RXI_N.B.JA"
"Time Period","RXI$US_N.B.AL","RXI$US_N.B.EU","RXI_N.B.JA"
2000-01-03,0.6591,1.0155,101.7
2000-01-04,0.6600,ND,103.09
2000-01-05,,1.0335,ND
'''

H15 = '''"Series Description","Market yield at 5-year","Market yield at 10-year"
"Unit:","Percent:_Per_Year","Percent:_Per_Year"
"Multiplier:","1","1"
"Currency:","NA","NA"
"Unique Identifier: ","H15/H15/RIFLGFCY05_N.B","H15/H15/RIFLGFCY10_N.B"
"Time Period","RIFLGFCY05_N.B","RIFLGFCY10_N.B"
1962-01-02,,
2000-01-03,6.50,6.58
2000-01-04,6.40,6.49
2000-01-05,ND,6.62
'''


def test_fed_parser_selects_column_by_identifier_and_drops_missing():
    eur = ds.parse_fed_ddp(H10, "RXI$US_N.B.EU")
    assert list(eur.index.strftime("%Y-%m-%d")) == ["2000-01-03", "2000-01-05"]
    assert eur.iloc[0] == pytest.approx(1.0155)
    jpy = ds.parse_fed_ddp(H10, "RXI_N.B.JA")
    assert list(jpy.values) == [101.7, 103.09]


def test_fed_parser_handles_h15_blank_and_nd_cells():
    s = ds.parse_fed_ddp(H15, "RIFLGFCY10_N.B")
    assert list(s.values) == [6.58, 6.49, 6.62]  # the 1962 blank row is dropped
    assert list(ds.parse_fed_ddp(H15, "RIFLGFCY05_N.B").values) == [6.5, 6.4]


def test_fed_parser_rejects_unknown_identifier_and_malformed_file():
    with pytest.raises(KeyError):
        ds.parse_fed_ddp(H10, "RXI_N.B.XX")
    with pytest.raises(ValueError):
        ds.parse_fed_ddp("just,one,line\n", "RXI_N.B.JA")


def test_eia_parser_reads_date_and_price_columns():
    raw = pd.DataFrame([["Sourcekey", "RBRTE"], ["Date", "Price"], [pd.Timestamp("2000-01-04"), 23.95],
                        [pd.Timestamp("2000-01-05"), 23.72], [pd.Timestamp("2000-01-06"), None]])
    s = ds.parse_eia_daily(raw)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2000-01-04", "2000-01-05"]
    assert list(s.values) == [23.95, 23.72]


def test_asset_registry_lists_only_documented_public_domain_series():
    assert len(ds.ASSETS) >= 6
    providers = {a.provider for a in ds.ASSETS.values()}
    assert providers <= {"h10", "h15", "eia"}
    for spec in ds.ASSETS.values():
        assert spec.source_key and spec.group in {"fx", "commodity", "rates"}
        assert spec.kind in {"level", "yield"}
    assert ds.ASSETS["UST10Y"].kind == "yield" and ds.ASSETS["UST10Y"].maturity == 10.0
    assert not any("S&P" in a.label or "Fama" in a.label for a in ds.ASSETS.values())


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
