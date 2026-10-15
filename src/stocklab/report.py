"""Build the summary JSON and the static explorer's data file from the raw per-asset outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import analysis as an
from .data_sources import ASSETS, load_asset
from .experiments import BASE_COST_BPS


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    typed = {k: np.asarray(v, dtype=np.int64 if k.endswith("day") else np.float32) for k, v in arrays.items()}
    np.savez_compressed(path, **typed)


def build(data_dir: Path, raw_dir: Path, out_json: Path, explorer_json: Path | None, every: int = 5):
    """Read results/raw/*.npz, compute everything, write ``summary.json`` and the explorer data."""
    data_dir, raw_dir = Path(data_dir), Path(raw_dir)
    assets = [a for a in ASSETS if (raw_dir / f"forecast_{a}.npz").exists()]
    if not assets:
        raise SystemExit("no forecast results found; run `stocklab forecast` first")
    summary: dict = {"assets": {}, "base_cost_bps": BASE_COST_BPS}
    explorer: dict = {"assets": {}, "base_cost_bps": BASE_COST_BPS}
    forecast_by_asset = {}
    for a in assets:
        raw = _load_npz(raw_dir / f"forecast_{a}.npz")
        log_df = load_asset(Path(data_dir) / "processed", a)
        simple_r = np.expm1(log_df["log_ret"].to_numpy())
        prices = log_df["price"].to_numpy()
        days = raw["day"].astype(int)
        fr = an.forecast_report({k: v.astype(float) for k, v in raw.items()})
        forecast_by_asset[a] = fr
        positions = an.signal_strategies(raw, simple_r, prices)
        nets, turns = an.trading_nets(positions, simple_r, days)
        entry = {"label": ASSETS[a].label, "group": ASSETS[a].group, "forecast": fr}

        dqn_file = raw_dir / f"trade_{a}.npz"
        dqn_nets = None
        if dqn_file.exists():
            z = _load_npz(dqn_file)
            per_seed: dict[int, dict[str, np.ndarray]] = {}
            for k, v in z.items():
                s, key = k.split("/", 1)
                per_seed.setdefault(int(s), {})[key] = v
            if not all(np.array_equal(per_seed[s]["day"].astype(int), days) for s in per_seed):
                raise SystemExit(f"{a}: DQN test days differ from the forecast test days")
            entry["dqn"] = an.dqn_report(per_seed)
            dqn_nets = np.mean([per_seed[s][f"net_{BASE_COST_BPS:g}"] for s in sorted(per_seed)], axis=0)
            nets["dqn_ensemble"] = dqn_nets
        entry["strategies"] = an.strategy_report(nets)
        entry["mean_turnover_per_day"] = {k: float(np.mean(v)) for k, v in turns.items()}
        entry["share_long_days"] = {k: float(np.mean(v > 0)) for k, v in positions.items()}
        entry["share_short_days"] = {k: float(np.mean(v < 0)) for k, v in positions.items()}
        if "dqn" in entry:
            entry["share_long_days"]["dqn_ensemble"] = entry["dqn"]["share_long"]
            entry["share_short_days"]["dqn_ensemble"] = entry["dqn"]["share_short"]
            entry["mean_turnover_per_day"]["dqn_ensemble"] = entry["dqn"]["mean_abs_turnover_per_day"]
        entry["signal_cost_sweep"] = an.signal_cost_sweep(positions, simple_r, days)
        entry["test_period"] = [f"{log_df.index[days[0]]:%Y-%m-%d}", f"{log_df.index[days[-1]]:%Y-%m-%d}"]
        entry["n_test_days"] = int(len(days))
        summary["assets"][a] = entry
        explorer["assets"][a] = {
            "label": ASSETS[a].label,
            "group": ASSETS[a].group,
            "dates": [],
            "equity": {},
            "drawdown": {},
            "strategies": entry["strategies"],
            "share_long_days": entry["share_long_days"],
            "share_short_days": entry["share_short_days"],
            "turnover": entry["mean_turnover_per_day"],
            "forecast": {m: {k: v for k, v in row.items()} for m, row in fr["models"].items()},
            "dqn_cost": entry.get("dqn", {}).get("per_cost"),
            "signal_cost_sweep": entry["signal_cost_sweep"],
        }
        # curves have len(days)+1 points (the start, then each test day); thin them and keep the dates aligned
        for k, v in nets.items():
            eq, dd = an.sample_curves(v, every)
            explorer["assets"][a]["equity"][k] = eq
            explorer["assets"][a]["drawdown"][k] = dd
        all_dates = [f"{log_df.index[days[0] - 1]:%Y-%m-%d}"] + [f"{d:%Y-%m-%d}" for d in log_df.index[days]]
        explorer["assets"][a]["dates"] = [all_dates[i] for i in an.sample_indices(len(all_dates), every)]

    holm = an.holm_over_assets(forecast_by_asset, "lstm")
    for a, p in holm.items():
        summary["assets"][a]["forecast"]["models"]["lstm"]["dm_p_holm_across_assets"] = p
    wins = [a for a in assets if forecast_by_asset[a]["models"]["lstm"]["mse_skill_vs_zero"] > 0]
    summary["aggregate"] = {
        "n_assets": len(assets),
        "lstm_mse_skill_positive": wins,
        "lstm_dm_significant_after_holm_5pct": [a for a, p in holm.items() if p < 0.05],
        "dqn_ensemble_sharpe_above_buy_and_hold": [
            a for a in assets
            if "dqn_ensemble" in summary["assets"][a]["strategies"]
            and summary["assets"][a]["strategies"]["dqn_ensemble"]["sharpe"]
            > summary["assets"][a]["strategies"]["buy_and_hold"]["sharpe"]
        ],
        "dqn_assets": [a for a in assets if "dqn" in summary["assets"][a]],
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    if all("dqn" in summary["assets"][a] for a in assets):
        # the committed run writes tables.md; any other output name (for example a smoke run) gets its own
        name = "tables.md" if Path(out_json).name == "summary.json" else f"{Path(out_json).stem}_tables.md"
        tables = Path(out_json).with_name(name)
        tables.write_text(tables_markdown(summary), encoding="utf-8", newline="\n")
    if explorer_json is not None:
        explorer["aggregate"] = summary["aggregate"]
        Path(explorer_json).parent.mkdir(parents=True, exist_ok=True)
        Path(explorer_json).write_text(
            json.dumps(explorer, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n"
        )
    return summary


def tables_markdown(summary: dict) -> str:
    """Markdown tables of the headline results, generated from the summary so they cannot drift."""
    lines = ["## Forecasting: next-day log return, pooled walk-forward test days", "",
             "| Asset | Test days | LSTM RMSE skill vs zero | LSTM DM p vs zero (Holm) | LSTM DM p vs drift "
             "| ARIMA skill vs zero | Drift skill vs zero | LSTM direction hit rate | Share of up days |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a, e in summary["assets"].items():
        m = e["forecast"]["models"]
        lines.append(
            f"| {a} | {e['n_test_days']} | {100 * m['lstm']['mse_skill_vs_zero']:.2f}% "
            f"| {m['lstm']['dm_p_vs_zero']:.3f} ({m['lstm']['dm_p_holm_across_assets']:.3f}) "
            f"| {m['lstm']['dm_p_vs_drift']:.3f} | {100 * m['arima']['mse_skill_vs_zero']:.2f}% "
            f"| {100 * m['drift']['mse_skill_vs_zero']:.2f}% "
            f"| {100 * m['lstm']['directional_accuracy']:.1f}% "
            f"| {100 * e['forecast']['share_up_days']:.1f}% |"
        )
    names = ["buy_and_hold", "ma_crossover", "lstm_direction", "lstm_return_sign", "dqn_ensemble"]
    lines += ["", f"## Trading at {summary['base_cost_bps']:g} bp per unit of turnover: Sharpe ratio "
              "[95% bootstrap CI], pooled test days", "",
              "| Asset | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a, e in summary["assets"].items():
        cells = []
        for n in names:
            s = e["strategies"][n]
            cells.append(f"{s['sharpe']:.2f} [{s['sharpe_ci'][0]:.2f}, {s['sharpe_ci'][1]:.2f}]")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", "## DQN (3 seeds, equal weight): Sharpe ratio by cost in bp, and trading behaviour", "",
              "| Asset | 0 bp | 2 bp | 5 bp | 10 bp | 20 bp | Turnover per day | Days long | Days short |",
              "|---|---|---|---|---|---|---|---|---|"]
    for a, e in summary["assets"].items():
        d = e["dqn"]
        sh = [f"{d['per_cost'][c]['ensemble']['sharpe']:.2f}" for c in ("0", "2", "5", "10", "20")]
        lines.append(f"| {a} | " + " | ".join(sh) + f" | {d['mean_abs_turnover_per_day']:.2f} "
                     f"| {100 * d['share_long']:.0f}% | {100 * d['share_short']:.0f}% |")
    lines += ["", "## Paired comparison with buy and hold: mean daily excess return in bp [95% CI]", "",
              "| Asset | " + " | ".join(n for n in names if n != "buy_and_hold") + " |",
              "|---|" + "---|" * (len(names) - 1)]
    for a, e in summary["assets"].items():
        cells = []
        for n in names[1:]:
            s = e["strategies"][n]
            cells.append(f"{1e4 * s['daily_excess_vs_benchmark']:.2f} "
                         f"[{1e4 * s['daily_excess_ci'][0]:.2f}, {1e4 * s['daily_excess_ci'][1]:.2f}]")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
