"""Turn raw per-block predictions and returns into the tables, intervals and tests that get reported."""

from __future__ import annotations

import numpy as np

from . import metrics as mt
from . import stats as st
from .env import simulate_positions
from .experiments import BASE_COST_BPS, COSTS_BPS
from .strategies import buy_and_hold_positions, ma_crossover_positions

BASELINES = ("zero", "last", "drift", "ses", "arima")


def concat_blocks(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-block dicts (same keys) in chronological order."""
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def lstm_ensemble(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Mean over seeds of the regression forecast and of the up-probability."""
    seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in raw if k.startswith("lstm_ret_s")})
    ret = np.mean([raw[f"lstm_ret_s{s}"] for s in seeds], axis=0)
    prob = np.mean([raw[f"lstm_prob_s{s}"] for s in seeds], axis=0)
    return ret, prob, seeds


def forecast_report(raw: dict[str, np.ndarray], seed: int = 0) -> dict:
    """Accuracy of every forecaster on the pooled test blocks of one asset."""
    y = raw["y"]
    lstm_ret, _, seeds = lstm_ensemble(raw)
    preds = {m: raw[m] for m in BASELINES}
    preds["lstm"] = lstm_ret
    for s in seeds:
        preds[f"lstm_s{s}"] = raw[f"lstm_ret_s{s}"]
    se_zero = (y - preds["zero"]) ** 2
    nonzero = y != 0
    always_up = float(np.mean(y[nonzero] > 0))
    out: dict = {"n_test": int(len(y)), "share_up_days": always_up, "models": {}}
    for name, p in preds.items():
        se = (y - p) ** 2
        row = {
            "rmse": mt.rmse(y, p),
            "mae": mt.mae(y, p),
            "mse_skill_vs_zero": mt.mse_skill(y, p, preds["zero"]),
        }
        if name != "zero":
            stat, pval = st.diebold_mariano(y - p, y - preds["zero"])
            lo, hi = st.paired_block_bootstrap_ci(se_zero, se, seed=seed)
            hits = int(np.sum(np.sign(p[nonzero]) == np.sign(y[nonzero])))
            if name != "drift":
                d_stat, d_p = st.diebold_mariano(y - p, y - preds["drift"])
                row.update({"dm_stat_vs_drift": d_stat, "dm_p_vs_drift": d_p})
            row.update({
                "dm_stat_vs_zero": stat,
                "dm_p_vs_zero": pval,
                "mse_gain_ci": [lo, hi],  # mean(se_zero - se_model); positive favours the model
                "directional_accuracy": hits / int(nonzero.sum()),
                "directional_p_vs_half": st.binomial_pvalue(hits, int(nonzero.sum())),
                # an "always up" rule scores share_up_days, so that is the fair bar for a direction call
                "directional_p_vs_always_up": st.binomial_pvalue(
                    hits, int(nonzero.sum()), min(max(always_up, 1e-9), 1 - 1e-9)
                ),
            })
        out["models"][name] = row
    return out


def holm_over_assets(per_asset: dict[str, dict], model: str = "lstm") -> dict[str, float]:
    """Holm-adjusted DM p-values for ``model`` versus the zero forecast across assets."""
    names = list(per_asset)
    p = [per_asset[n]["models"][model]["dm_p_vs_zero"] for n in names]
    return dict(zip(names, map(float, st.holm_adjust(p)), strict=True))


def _net_over_blocks(positions: np.ndarray, simple_r: np.ndarray, days: np.ndarray, cost_bps: float):
    """Apply the shared accounting block by block; every block starts flat."""
    net = np.empty(len(days))
    turn = np.empty(len(days))
    breaks = np.flatnonzero(np.diff(days) != 1) + 1
    for idx in np.split(np.arange(len(days)), breaks):
        n, t, _ = simulate_positions(positions[idx], simple_r[days[idx]], cost_bps)
        net[idx], turn[idx] = n, t
    return net, turn


def signal_strategies(raw: dict[str, np.ndarray], simple_r: np.ndarray, prices: np.ndarray) -> dict:
    """Positions for the rule and forecast-driven strategies on the pooled test days.

    ``raw['day']`` indexes the return array; blocks are contiguous so each block is handled alone.
    """
    days = raw["day"].astype(int)
    lstm_ret, prob, _ = lstm_ensemble(raw)
    breaks = np.flatnonzero(np.diff(days) != 1) + 1
    ma = np.concatenate([
        ma_crossover_positions(prices, int(days[i[0]]), int(days[i[-1]]) + 1)
        for i in np.split(np.arange(len(days)), breaks)
    ])
    return {
        "buy_and_hold": buy_and_hold_positions(len(days)),
        "ma_crossover": ma,
        "lstm_direction": np.where(prob > 0.5, 1.0, -1.0),
        "lstm_return_sign": np.where(lstm_ret > 0, 1.0, -1.0),
        "flat": np.zeros(len(days)),
    }


def strategy_report(
    net_by_name: dict[str, np.ndarray], benchmark: str = "buy_and_hold", seed: int = 0
) -> dict:
    """Headline numbers with bootstrap intervals; paired comparison with the benchmark."""
    bench = net_by_name[benchmark]
    out = {}
    for name, net in net_by_name.items():
        s = mt.summarise_strategy(net)
        lo, hi = st.bootstrap_stat_ci(net, mt.sharpe, seed=seed)
        s["sharpe_ci"] = [lo, hi]
        if name != benchmark:
            d_lo, d_hi = st.paired_block_bootstrap_ci(net, bench, seed=seed)
            s["daily_excess_vs_benchmark"] = float(np.mean(net - bench))
            s["daily_excess_ci"] = [d_lo, d_hi]
        out[name] = s
    return out


def signal_cost_sweep(positions: dict[str, np.ndarray], simple_r: np.ndarray, days: np.ndarray) -> dict:
    """Sharpe and final equity for each strategy at each cost level."""
    out: dict = {}
    for c in COSTS_BPS:
        nets = {k: _net_over_blocks(v, simple_r, days, c)[0] for k, v in positions.items()}
        out[f"{c:g}"] = {k: mt.summarise_strategy(v) for k, v in nets.items()}
    return out


def trading_nets(positions: dict[str, np.ndarray], simple_r: np.ndarray, days: np.ndarray,
                 cost_bps: float = BASE_COST_BPS):
    nets, turns = {}, {}
    for k, v in positions.items():
        nets[k], turns[k] = _net_over_blocks(v, simple_r, days, cost_bps)
    return nets, turns


def dqn_report(per_seed: dict[int, dict[str, np.ndarray]]) -> dict:
    """Summaries of the DQN agents. ``per_seed[seed]`` has day, positions and net_<cost> arrays."""
    seeds = sorted(per_seed)
    out: dict = {"seeds": seeds, "per_cost": {}, "per_seed_sharpe": {}}
    for c in COSTS_BPS:
        key = f"net_{c:g}"
        nets = [per_seed[s][key] for s in seeds]
        avg = np.mean(nets, axis=0)  # equal-weight portfolio of the seed agents
        out["per_cost"][f"{c:g}"] = {
            "ensemble": mt.summarise_strategy(avg),
            "per_seed": {str(s): mt.summarise_strategy(n) for s, n in zip(seeds, nets, strict=True)},
        }
    base = f"net_{BASE_COST_BPS:g}"
    out["per_seed_sharpe"] = {str(s): mt.sharpe(per_seed[s][base]) for s in seeds}
    pos = np.array([per_seed[s]["positions"] for s in seeds])
    out["mean_abs_turnover_per_day"] = float(np.mean(np.abs(np.diff(pos, axis=1))))
    out["share_long"] = float(np.mean(pos > 0))
    out["share_short"] = float(np.mean(pos < 0))
    out["share_flat"] = float(np.mean(pos == 0))
    return out


def sample_indices(n_points: int, every: int) -> list[int]:
    """Indices 0, every, 2*every, ... and always the last point."""
    idx = list(range(0, n_points, every))
    if idx[-1] != n_points - 1:
        idx.append(n_points - 1)
    return idx


def sample_curves(net_returns: np.ndarray, every: int) -> tuple[list[float], list[float]]:
    """Equity indexed to 100 and drawdown in percent, sampled every ``every`` days.

    Each sampled drawdown is the worst value since the previous sampled point, so thinning the
    curve never hides the deepest fall. Only strategy performance is returned, never a price.
    """
    eq = 100.0 * mt.equity_curve(net_returns)
    dd = mt.drawdown_series(eq)
    ix = sample_indices(len(eq), every)
    equity, drawdown, prev = [], [], 0
    for i in ix:
        worst = 0.0 if i == 0 else float(dd[prev + 1 : i + 1].min())
        equity.append(round(float(eq[i]), 2))
        drawdown.append(round(100.0 * worst, 2))
        prev = i
    return equity, drawdown
