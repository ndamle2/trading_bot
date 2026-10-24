"""Backtest ETF / basket arbitrage on the research prices (no fair values needed).

    ../.venv/bin/python etf_arb.py            # tune on 201-600, report on the holdout 1-200

Trade (per ETF, every `every` s): if the ETF bid beats the cost of buying its basket at the asks by more than
`min_edge` (fraction of the basket value) per unit, sell n ETF + buy n x basket, then CREATE n (give the basket,
get n ETF back) so every position returns to where it was. Mirror image when the ETF is cheap (buy ETF, sell
basket, REDEEM). n is the largest unit count whose marginal unit still clears min_edge, walking each book.
Costs: decision at t, fills at the house prices `latency` s later (the gap may have closed: leg risk is paid).
Actions: 1 + legs + 1 (create/redeem) per arb, within `actions_per_min`.
Rivals: when the gap seen `lag` s ago beat their threshold they took `share` of the depth and narrowed the gap
by `push`. `noise` removes that fraction of the measured gap as non-tradable quote noise (stress).
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import case, prices  # noqa: E402
from backtest import book_profile  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
ETFS = case()["etfs"]
LIMITS = case()["limits"]
TS = np.arange(0, 780, 2)


@dataclass(frozen=True)
class ArbParams:
    min_edge: float = 0.001       # per-unit profit needed, as a fraction of the basket value
    every: int = 2
    latency: int = 2
    actions_per_min: int = 45
    refill: float = 0.066
    lag: int = 4
    share: float = 0.3
    push: float = 0.3
    rival_edge: float = 0.001
    noise: float = 0.0
    max_units: int = 0            # 0 = ETF position limit


SCENARIOS = {
    "no rivals": dict(share=0.0, push=0.0),
    "base: rivals 4 s slower, 30%": dict(),
    "rivals 2 s slower, 50%": dict(lag=2, share=0.5, push=0.5),
    "rivals as fast, 50%": dict(lag=0, share=0.5, push=0.5),
    "base + 30% of gap is noise": dict(noise=0.3),
    "base + latency 4 s": dict(latency=4),
}


def load(seasons_sel):
    p = prices()
    p = p[seasons_sel(p.season)].sort_values(["season", "round", "t"])
    p = p[p.t < 780]
    ss = np.array(sorted(p.season.unique()))
    n = len(ss)
    arr = {s: p[s].to_numpy().reshape(n, 3 * len(TS)) for s in p.columns if s not in ("season", "round", "t", "time")}
    return ss, arr


def walk_cost(mid, off, depth, units, per_unit, side):
    """Marginal price of the `units`-th unit when taking `per_unit` shares per unit through levels (vectorised)."""
    need = units * per_unit
    cum = np.cumsum(depth, axis=-1)
    lvl = np.argmax(cum >= need[..., None], axis=-1)
    ok = cum[..., -1] >= need
    price = mid * (1 + side * np.take_along_axis(off, lvl[..., None], -1)[..., 0])
    return np.where(ok, price, np.nan)


def simulate(arr, prm: ArbParams, prof=None):
    prof = prof or book_profile()
    n_s = next(iter(arr.values())).shape[0]
    nstep = 3 * len(TS)
    treal = np.tile(TS, 3)
    rnd = np.repeat([1, 2, 3], len(TS))
    pnl = {e: np.zeros(n_s) for e in ETFS}
    arbs = {e: np.zeros(n_s) for e in ETFS}
    units_tot = {e: np.zeros(n_s) for e in ETFS}
    budget_hist = np.zeros((n_s, 0))
    acts = []  # (step, n_actions per season)
    lag_s, lat_s, ev_s = prm.lag // 2, prm.latency // 2, max(prm.every // 2, 1)
    avail = {}
    win = 30  # 60 s of 2 s steps
    used = np.zeros((n_s, nstep))
    for j in range(0, nstep):
        reg = "pre" if treal[j] < 60 else "storm"
        for e, spec in ETFS.items():
            syms = [e] + list(spec["basket"])
            for s in syms:
                off, full = prof[(s, reg)][:, 0], prof[(s, reg)][:, 1]
                key = (s, "b"), (s, "a")
                for k in key:
                    if k not in avail or treal[j] == 0:
                        avail[k] = np.broadcast_to(full, (n_s, len(full))).copy()
                    else:
                        avail[k] = np.minimum(full, avail[k] + prm.refill * full)
        if j % ev_s:
            continue
        je = j + lat_s if (j + lat_s < nstep and rnd[j + lat_s] == rnd[j]) else j
        jr = j - lag_s if (j - lag_s >= 0 and rnd[j - lag_s] == rnd[j]) else None
        for e, spec in ETFS.items():
            b = spec["basket"]
            nav = sum(q * arr[s][:, j] for s, q in b.items())
            gap = (arr[e][:, j] - nav) * (1 - prm.noise)
            # rivals: they saw the gap lag s ago
            if jr is not None:
                nav_r = sum(q * arr[s][:, jr] for s, q in b.items())
                gr = arr[e][:, jr] - nav_r
                active = np.abs(gr) > prm.rival_edge * nav_r
            else:
                active = np.zeros(n_s, bool)
            gap = np.where(active, gap * (1 - prm.push), gap)
            side = np.where(gap > 0, -1, 1)            # -1: ETF rich -> sell ETF, buy basket
            # effective mids for decision: basket at nav, ETF at nav + gap
            etf_mid = nav + gap
            reg = "pre" if treal[j] < 60 else "storm"
            # depth available (after rivals took their share)
            def depth(s, take_ask):
                d = avail[(s, "a" if take_ask else "b")]
                return np.where(active[:, None], d * (1 - prm.share), d)
            # ETF leg: rich -> we sell (hit bids); cheap -> we buy (lift asks)
            lim = prm.max_units or LIMITS[e]
            best_n = np.zeros(n_s)
            for u in range(1, lim + 1):
                etf_px = np.where(side < 0,
                                  walk_cost(etf_mid, np.broadcast_to(prof[(e, reg)][:, 0], (n_s, 5)), depth(e, False), np.full(n_s, u), 1, -1),
                                  walk_cost(etf_mid, np.broadcast_to(prof[(e, reg)][:, 0], (n_s, 5)), depth(e, True), np.full(n_s, u), 1, 1))
                bkt = np.zeros(n_s)
                for s, q in b.items():
                    off = np.broadcast_to(prof[(s, reg)][:, 0], (n_s, 5))
                    px_s = arr[s][:, j]
                    buy = walk_cost(px_s, off, depth(s, True), np.full(n_s, u), q, 1)
                    sell = walk_cost(px_s, off, depth(s, False), np.full(n_s, u), q, -1)
                    bkt = bkt + q * np.where(side < 0, buy, sell)
                marg = np.where(side < 0, etf_px - bkt, bkt - etf_px)
                ok = np.nan_to_num(marg, nan=-1e9) > prm.min_edge * nav
                if not ok.any():
                    break
                best_n = np.where(ok & (best_n == u - 1), u, best_n)
            # rate limit: actions in the last 60 s
            cost_actions = 2 + len(b)
            recent = used[:, max(0, j - win + 1): j + 1].sum(1)
            can = (recent + cost_actions) <= prm.actions_per_min
            n_units = np.where(can, best_n, 0)
            go = n_units > 0
            if not go.any():
                continue
            used[go, j] += cost_actions
            # execute at prices `latency` later (gap may have moved), walking the same depth
            nav_e = sum(q * arr[s][:, je] for s, q in b.items())
            gap_e = (arr[e][:, je] - nav_e) * (1 - prm.noise)
            gap_e = np.where(active, gap_e * (1 - prm.push), gap_e)
            etf_mid_e = nav_e + gap_e
            proceeds = np.zeros(n_s)
            u = n_units
            # average fill across levels for u units: approximate by walking unit by unit
            for k in range(1, int(u.max()) + 1):
                m = u >= k
                etf_px = np.where(side < 0,
                                  walk_cost(etf_mid_e, np.broadcast_to(prof[(e, reg)][:, 0], (n_s, 5)), depth(e, False), np.full(n_s, k), 1, -1),
                                  walk_cost(etf_mid_e, np.broadcast_to(prof[(e, reg)][:, 0], (n_s, 5)), depth(e, True), np.full(n_s, k), 1, 1))
                bkt = np.zeros(n_s)
                for s, q in b.items():
                    off = np.broadcast_to(prof[(s, reg)][:, 0], (n_s, 5))
                    px_s = arr[s][:, je]
                    buy = walk_cost(px_s, off, depth(s, True), np.full(n_s, k), q, 1)
                    sell = walk_cost(px_s, off, depth(s, False), np.full(n_s, k), q, -1)
                    bkt = bkt + q * np.where(side < 0, buy, sell)
                unit = np.where(side < 0, etf_px - bkt, bkt - etf_px)
                proceeds += np.where(m, np.nan_to_num(unit, nan=0.0), 0.0)
            pnl[e] += np.where(go, proceeds, 0.0)
            arbs[e] += go
            units_tot[e] += np.where(go, u, 0)
            # consume depth
            for s, q in [(e, 1)] + list(b.items()):
                for k_side, take_ask in (("a", True), ("b", False)):
                    if s == e:
                        mask = go & ((side > 0) == take_ask)
                        sh = u
                    else:
                        mask = go & ((side < 0) == take_ask)
                        sh = u * q
                    d = avail[(s, k_side)]
                    left = np.where(mask, sh, 0).astype(float)
                    for lv in range(d.shape[1]):
                        t_ = np.minimum(d[:, lv], left)
                        d[:, lv] -= t_
                        left -= t_
    return pnl, arbs, units_tot


def summary(pnl, arbs, units, label):
    tot = sum(pnl.values())
    return {"scenario": label, "mean": tot.mean(), "median": np.median(tot), "p5": np.percentile(tot, 5),
            "min": tot.min(), **{f"{e} $": pnl[e].mean() for e in ETFS}, **{f"{e} arbs": arbs[e].mean() for e in ETFS},
            **{f"{e} units": units[e].mean() for e in ETFS}}


def main():
    prof = book_profile()
    _, tr = load(lambda s: s > 200)
    _, ho = load(lambda s: s <= 200)
    base = ArbParams()
    # Others will run this arbitrage too, so the margin is chosen for the average over the competitive scenarios
    # (not the friendliest one), from a fixed grid that is not widened.
    tune_sc = {k: v for k, v in SCENARIOS.items() if k in ("base: rivals 4 s slower, 30%", "rivals 2 s slower, 50%",
                                                          "rivals as fast, 50%")}
    print("== choosing min_edge on training seasons, averaged over:", ", ".join(tune_sc))
    grid = []
    for me in (0.0005, 0.001, 0.002, 0.004):
        means = []
        for sc in tune_sc.values():
            pnl, arbs, units = simulate(tr, replace(base, min_edge=me, **sc), prof)
            means.append(sum(pnl.values()).mean())
        grid.append((float(np.mean(means)), me))
        print(f"  min_edge {me:<7} " + "  ".join(f"{m:6.0f}" for m in means) + f"   average {np.mean(means):6.0f}", flush=True)
    best = replace(base, min_edge=max(grid)[1])
    print("chosen min_edge", best.min_edge)
    print("\n== budget sensitivity (training)")
    for apm in (25, 45, 70):
        pnl, arbs, _ = simulate(tr, replace(best, actions_per_min=apm), prof)
        print(f"  {apm} actions/min: mean {sum(pnl.values()).mean():7.0f}")

    print("\n== HOLDOUT seasons 1-200")
    rows = []
    for name, sc in SCENARIOS.items():
        rows.append(summary(*simulate(ho, replace(best, **sc), prof), name))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(0).to_string(index=False))
    OUT.mkdir(exist_ok=True)
    (OUT / "etf_arb_params.json").write_text(json.dumps(asdict(best), indent=1))
    print("wrote", OUT / "etf_arb_params.json")


if __name__ == "__main__":
    main()
