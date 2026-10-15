# Third-party notices

Original work in this repository is (c) Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`). No third-party data or code is copied into the repository. The data below is downloaded at run time by `stocklab fetch` and is not committed; only SHA-256 checksums of the processed series (`data/manifest.json`), model-performance results and strategy equity curves derived from the data are.

## Data used

All eight series are published by the Federal Reserve Board through its Data Download Program (https://www.federalreserve.gov/datadownload/). The Board is a U.S. government agency; the underlying Treasury yields are produced by the U.S. Treasury. Terms and sources were read at source on 2026-10-15 (the passages below are quoted from those pages):

| Series (as used) | Fed identifier | Publisher and what the source says |
|---|---|---|
| EUR/USD, USD/JPY, GBP/USD, USD/CAD, AUD/USD, USD/CHF daily exchange rates | H.10 `RXI$US_N.B.EU`, `RXI_N.B.JA`, `RXI$US_N.B.UK`, `RXI_N.B.CA`, `RXI$US_N.B.AL`, `RXI_N.B.SZ` | Federal Reserve Board, release H.10. https://www.federalreserve.gov/releases/h10/about.htm: "The data are noon buying rates in New York for cable transfers payable in the listed currencies. The rates have been certified by the Federal Reserve Bank of New York for customs purposes as required by section 522 of the amended Tariff Act of 1930." |
| 2-year and 10-year Treasury constant-maturity yields, used to build par-bond total-return series | H.15 `RIFLGFCY02_N.B`, `RIFLGFCY10_N.B` | Federal Reserve Board, release H.15; the yields come from the U.S. Treasury. https://www.federalreserve.gov/releases/h15/default.htm: yields on Treasury nominal securities at "constant maturity" are "interpolated by the U.S. Treasury from the daily yield curve for non-inflation-indexed Treasury securities" and are marked "Source: U.S. Treasury". |

**Licence evidence.** https://www.federalreserve.gov/disclaimer.htm ("Copyright/trademark"): "Unless otherwise indicated, information on Board's website is in the public domain and may be copied and distributed without permission. Please cite to the Board as the source of the information." The H.10 and H.15 release pages and the data-download pages carry no copyright, trademark or third-party notice for these series. Works of the U.S. government are not subject to copyright (17 U.S.C. 105). Attribution requested and given here: Board of Governors of the Federal Reserve System (U.S.), Data Download Program, releases H.10 and H.15 (retrieved 2026-10-15); yields: U.S. Treasury.

Only the six H.10 pairs and the two H.15 maturities listed are used. The commercially sourced H.15 series (commercial paper, from the Depository Trust Company) and every other H.10 series are not used.

**Treasury par-bond returns** are this project's own construction from the yields (see `src/stocklab/rates.py`); they are an approximation, not an index published by the Treasury or the Federal Reserve.

## Series considered and not used

| Series | Why not |
|---|---|
| Fama/French research factors and industry portfolios (Kenneth R. French Data Library) | Page carries "Copyright Eugene F. Fama and Kenneth R. French" and no redistribution licence. Not used. |
| S&P 500, NASDAQ, Wilshire and other index series republished by FRED | Third-party copyrighted or restricted. Not used. |
| EIA Brent, WTI and Henry Hub daily spot prices | EIA states that "U.S. government publications are in the public domain" (https://www.eia.gov/about/copyrights_reuse.php) but the same page warns that some material is "contributed or licensed by private individuals, companies, or organizations", and EIA's table notes give the source of the spot prices as a commercial vendor: "Sources: Refinitiv, an LSEG business" (https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp) and, for Henry Hub, "Spot Price: Refinitiv, an LSEG business" (https://www.eia.gov/dnav/ng/TblDefs/ng_pri_fut_tbldef2.asp). Whether that vendor data may be redistributed in derived results is unclear, so it is not used. WTI also had a negative price on 2020-04-20, which a log-return pipeline cannot handle. |
| FRED (fred.stlouisfed.org) copies of the Federal Reserve series | FRED's series notes could not be read: the St. Louis Fed servers timed out from the build machine on 2026-10-15, so the data was taken directly from the Board's own data-download files. For the three pairs EUR/USD, USD/JPY and GBP/USD, the processed series is byte-identical to the one built earlier from the FRED CSV copies. |

## Software (installed from PyPI, not copied)

| Package | License |
|---|---|
| NumPy, pandas, SciPy, statsmodels, patsy | BSD-3-Clause |
| PyTorch (CPU build) | BSD-3-Clause |
| Gymnasium | MIT |
| pytest, ruff | MIT |

No pretrained models are used. All networks are trained from scratch on CPU and none are committed.
