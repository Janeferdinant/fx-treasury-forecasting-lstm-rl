"""Command line: fetch data, run the experiments (in parallel), build the report.

    stocklab fetch [--write-manifest]
    stocklab forecast [--assets EURUSD ...] [--jobs 4] [--smoke]
    stocklab trade    [--assets EURUSD ...] [--jobs 4] [--smoke]
    stocklab report
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import experiments as ex
from . import fetch, report
from .data_sources import ASSETS
from .dqn import DQNConfig
from .lstm import LSTMConfig

SMOKE_ASSETS = ["EURUSD"]
SMOKE_LAST_BLOCKS = 2


def _forecast_task(args):
    data_dir, asset, fold_idx, cfg = args
    log_r, _, _ = ex.load_returns(data_dir, asset)
    return asset, fold_idx, ex.forecast_block(log_r, fold_idx, cfg)


def _trade_task(args):
    data_dir, asset, block_idx, seed, cfg, fcfg = args
    _, simple_r, _ = ex.load_returns(data_dir, asset)
    block = ex.trade_blocks(len(simple_r), fcfg)[block_idx]
    return asset, block_idx, seed, ex.trade_block(simple_r, block, seed, cfg)


def _configs(args) -> tuple[ex.ForecastConfig, ex.TradeConfig, list[str]]:
    fcfg = ex.ForecastConfig(seeds=tuple(args.seeds))
    tcfg = ex.TradeConfig(seeds=tuple(args.seeds))
    assets = args.assets or list(ASSETS)
    if args.smoke:
        fcfg = replace(fcfg, seeds=(0,), lstm=LSTMConfig(epochs=3))
        tcfg = replace(tcfg, seeds=(0,), dqn=DQNConfig(epochs=2, learn_start=200))
        assets = args.assets or SMOKE_ASSETS
    return fcfg, tcfg, assets


def _block_ids(n_blocks: int, smoke: bool) -> list[int]:
    ids = list(range(n_blocks))
    return ids[-SMOKE_LAST_BLOCKS:] if smoke else ids


def cmd_forecast(args) -> None:
    fcfg, _, assets = _configs(args)
    tasks = []
    for a in assets:
        n = len(ex.load_returns(args.data_dir, a)[0])
        ids = _block_ids(len(ex.forecast_folds(n, fcfg)), args.smoke)
        tasks += [(args.data_dir, a, i, fcfg) for i in ids]
    t0 = time.time()
    done: dict[str, dict[int, dict]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for asset, i, out in pool.map(_forecast_task, tasks):
            done.setdefault(asset, {})[i] = out
            print(f"forecast {asset} block {i} done ({time.time() - t0:.0f}s)", flush=True)
    from .analysis import concat_blocks

    for asset, blocks in done.items():
        merged = concat_blocks([blocks[i] for i in sorted(blocks)])
        report.save_npz(Path(args.raw_dir) / f"forecast_{asset}.npz", merged)
    print(f"forecast finished in {time.time() - t0:.0f}s for {len(tasks)} blocks")


def cmd_trade(args) -> None:
    fcfg, tcfg, assets = _configs(args)
    tasks = []
    for a in assets:
        n = len(ex.load_returns(args.data_dir, a)[0])
        for i in _block_ids(len(ex.trade_blocks(n, fcfg)), args.smoke):
            tasks += [(args.data_dir, a, i, s, tcfg, fcfg) for s in tcfg.seeds]
    t0 = time.time()
    done: dict[tuple[str, int], dict[int, dict]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for asset, i, seed, out in pool.map(_trade_task, tasks):
            done.setdefault((asset, seed), {})[i] = out
            print(f"trade {asset} block {i} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    by_asset: dict[str, dict] = {}
    from .analysis import concat_blocks

    for (asset, seed), blocks in done.items():
        merged = concat_blocks([blocks[i] for i in sorted(blocks)])
        for k, v in merged.items():
            by_asset.setdefault(asset, {})[f"{seed}/{k}"] = v
    for asset, arrays in by_asset.items():
        report.save_npz(Path(args.raw_dir) / f"trade_{asset}.npz", arrays)
    print(f"trade finished in {time.time() - t0:.0f}s for {len(tasks)} fits")


def cmd_fetch(args) -> None:
    info = fetch.run(Path(args.data_dir), write_manifest=args.write_manifest, refresh=args.refresh)
    print(json.dumps(info, indent=1))


def cmd_report(args) -> None:
    s = report.build(Path(args.data_dir), Path(args.raw_dir), Path(args.out), Path(args.explorer))
    print(json.dumps(s["aggregate"], indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="stocklab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, runs: bool) -> None:
        sp.add_argument("--data-dir", default="data")
        sp.add_argument("--raw-dir", default="results/raw")
        if runs:
            sp.add_argument("--assets", nargs="*", choices=list(ASSETS))
            sp.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
            sp.add_argument("--jobs", type=int, default=4)
            sp.add_argument("--smoke", action="store_true",
                            help="tiny run for checking the pipeline (EURUSD, last 2 blocks, 1 seed)")

    f = sub.add_parser("fetch", help="download sources and verify checksums")
    f.add_argument("--data-dir", default="data")
    f.add_argument("--write-manifest", action="store_true")
    f.add_argument("--refresh", action="store_true", help="re-download even if cached")
    f.set_defaults(fn=cmd_fetch)
    for name, fn in (("forecast", cmd_forecast), ("trade", cmd_trade)):
        sp = sub.add_parser(name)
        common(sp, True)
        sp.set_defaults(fn=fn)
    r = sub.add_parser("report", help="build results/summary.json and the explorer data")
    common(r, False)
    r.add_argument("--out", default="results/summary.json")
    r.add_argument("--explorer", default="docs/data.json")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
