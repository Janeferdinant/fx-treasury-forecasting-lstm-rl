# Run log 2026-10-15: walk-forward LSTM forecasting and DQN trading, eight Federal Reserve series

Status: full runs of `stocklab forecast` and `stocklab trade`, then `stocklab report`. These are the numbers in the README; nothing is quoted from outside this repository. History note: this repository was rebuilt on 2026-10-15 from earlier private experiments, with a new git history, after the data set was changed to series whose terms are clearly public domain (the earlier Fama-French, equity-index and EIA-based results and files are not part of this repository).

> Research code on public historical data. Not investment advice. The results say nothing about future returns.

## Environment

Python 3.11.9, Windows 11, 16 logical CPUs, CPU only (no CUDA). PyTorch 2.14.1+cpu, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, statsmodels 0.15.0, Gymnasium 1.4.0, pytest 9.1.1, ruff 0.17.0 (`requirements.txt`). Each worker process uses one PyTorch thread; 6 worker processes ran in parallel.

## Commands and wall time (UTC)

```
stocklab fetch --write-manifest   # once, 2026-10-15; later runs verify against data/manifest.json
stocklab forecast --jobs 6        # 2026-10-15T17:29:11Z to 17:33:23Z, 243 s for 72 blocks (8 assets x 9 blocks)
stocklab trade --jobs 6           # 2026-10-15T17:33:23Z to 17:45:48Z, 739 s for 216 DQN fits (72 blocks x 3 seeds)
stocklab report                   # writes results/summary.json, results/tables.md, docs/data.json
```

Each run stayed under the 25-minute limit. Progress logs: `results/logs/forecast.log`, `results/logs/trade.log`.

## Configuration

| | |
|---|---|
| Window | 20 days of returns |
| Walk-forward | first test sample 2,000 (about 8 years), blocks of 500 samples, last block partial; 250 samples before each block held out for early stopping |
| LSTM | 1 layer, 32 hidden, dropout 0.1, Adam lr 1e-3, weight decay 1e-5, up to 25 epochs, patience 4, batch 64; regression and direction networks; seeds 0, 1, 2 |
| Baselines | zero; yesterday's return; expanding mean; exponential smoothing (alpha on a grid, fitted on pre-test data); ARIMA(1,0,1) on returns fitted once on pre-block data |
| Trading environment | actions short / flat / long; 2 bp commission + 3 bp slippage = 5 bp per unit of turnover; every block starts flat; cost sweep 0, 2, 5, 10, 20 bp |
| DQN | hidden 64 x 2, double-DQN target, gamma 0.9, lr 1e-3, batch 64, buffer 20,000, 20 passes, epsilon 1.0 to 0.05 over 60% of steps, rolling 1,500-day training window, 250-day validation window, best-validation-Sharpe weights kept; seeds 0, 1, 2 |
| Treasury series | constant-maturity par bond revalued daily from H.15 yields plus one day of coupon carry; 2-year and 10-year |
| Statistics | stationary bootstrap, mean block length 10, 1,000 resamples (500 for Sharpe); one-sided Diebold-Mariano with Newey-West and HLN; Holm across 8 assets |

## Data and test periods

| Asset | Test days | Test period | Rows | SHA-256 of processed series (first 16 hex) | DQN Sharpe at 5 bp per seed (0 / 1 / 2) |
|---|---|---|---|---|---|
| EURUSD | 4497 | 2008-01-14 to 2025-12-31 | 6518 | `d7a554d06be3278e...` | -0.92 / -1.06 / -1.54 |
| USDJPY | 4497 | 2008-01-14 to 2025-12-31 | 6518 | `8a9f704f344d91ec...` | -0.77 / -0.53 / -1.31 |
| GBPUSD | 4497 | 2008-01-14 to 2025-12-31 | 6518 | `6d786212ea01506d...` | -1.28 / -1.43 / -1.17 |
| USDCAD | 4497 | 2008-01-14 to 2025-12-31 | 6518 | `d9936614f576631f...` | -0.84 / -1.76 / -1.29 |
| AUDUSD | 4497 | 2008-01-14 to 2025-12-31 | 6518 | `2018e372a4d7c2b3...` | -1.01 / -0.97 / -0.83 |
| USDCHF | 4497 | 2008-01-14 to 2025-12-31 | 6518 | `93ffcb6ac86f185c...` | -0.69 / -0.93 / -0.60 |
| UST2Y | 4480 | 2008-02-04 to 2025-12-31 | 6501 | `c295057c0eed028b...` | -1.60 / -1.48 / -1.15 |
| UST10Y | 4480 | 2008-02-04 to 2025-12-31 | 6501 | `7b1fac54af82bf1f...` | -1.54 / -1.69 / -1.63 |

Full checksums: `data/manifest.json`. Source files were downloaded on 2026-10-15 from the Federal Reserve Board Data Download Program (see THIRD_PARTY_NOTICES.md for the exact series and licence evidence).

## Results

The complete tables are in `results/tables.md` and the README; per-asset detail (all forecasters, all strategies, cost sweeps, per-seed numbers, bootstrap intervals) is in `results/summary.json`. Aggregate:

- LSTM squared error below the zero forecast on 1 of 8 assets (GBPUSD, +0.02%); range -0.25% to +0.02%.
- LSTM versus zero, Diebold-Mariano one-sided: smallest raw p-value 0.39 (GBPUSD); **0 of 8 significant**; every Holm-adjusted p-value is 1.00.
- DQN equal-weight Sharpe above buy and hold: **0 of 8** assets at 5 bp; the 95% interval of its daily excess return over buy and hold is below zero on all 8.
- LSTM direction strategy is long on 100% of days on both Treasury series (so identical to buy and hold there); on EUR/USD and USD/JPY its paired daily excess return over buy and hold has a 95% interval below zero.

## Checks

- `pytest -q`: 108 passed (101 with `-m "not slow"`, as in CI). `ruff check .`: all checks passed.
- Leak tests: the first test day's forecast from every model is unchanged when all returns from that day onward are rewritten (`tests/test_experiments.py`); features contain only past days (`tests/test_features.py`); the fold generator covers each test sample once (`tests/test_splits.py`).
- Explorer data: `tests/test_report_cli.py` checks that `docs/data.json` holds only strategy curves (indexed to 100), drawdowns and metrics, with no price or return series.
- Playwright (Chromium) on `docs/index.html` served locally, at 390 px and 1280 px: aggregate list, asset selector (8 assets), equity chart and drawdown chart with 6 lines each, three tables of 6 rows, legend toggles, hover read-out, 0 px horizontal page overflow, 0 console errors or warnings, no requests to other hosts. Screenshots: `docs/explorer-390.png`, `docs/explorer-1280.png`. Tables scroll sideways inside their own box on narrow screens. (An overflow caused by a long asset name in the selector was found and fixed.)
- Same-data cross-check: EUR/USD, USD/JPY and GBP/USD, which were also processed in earlier private work from FRED copies of the same Federal Reserve series, give identical processed-series checksums and identical forecasting and trading tables here.
