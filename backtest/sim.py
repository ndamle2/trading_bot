"""Competitor-aware backtest at 2-second resolution, vectorised across seasons.

    ../.venv/bin/python sim.py                 # tune on 201-600 under the base competition scenario, report holdout

Market model (per 2 s step, per symbol)
  house     px from market.csv (consensus) with a book shaped like testnet books (bot/logs/books.jsonl);
            liquidity refills `refill` of full depth per step; storm-time spreads from storm formation on
  rivals    other bots as smart as us but `lag` seconds behind: they act on our value from `lag` s ago. Where their
            edge beats their margin they take `share` of the liquidity on that side first and move the price we can
            trade at by `push` of their gap
  us        every `every` seconds, value V = px + beta * (fair - px); one limit order per symbol that takes every level
            priced through V -/+ need, need = (margin + size_pen*|pos|/limit + risk*sd/px*|pos|/limit) * px;
            fills at the house price `latency` s later (plus the rivals' push); at most `max_orders` orders per step
            across symbols, biggest dollar edge first (team limit 180 actions/min ~ 3 orders / 2 s)
Settlement: positions marked at settlement values at the end of round 3.
"""

from __future__ import annotations

import itertools
import json
import pickle
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SYMBOLS, case, prices, seasons  # noqa: E402
from backtest import book_profile, phase  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
LIM = np.array([case()["limits"][s] for s in SYMBOLS], float)
STEP = 2
TS = np.arange(0, 780, STEP)                     # 390 steps per round
NSTEP = 3 * len(TS)
PHASE = np.array([phase(r, t) for r in (1, 2, 3) for t in TS])
TREAL = np.tile(TS, 3)
ROUND = np.repeat([1, 2, 3], len(TS))


@dataclass(frozen=True)
class Params:
    margin: float = 0.0025
    size_pen: float = 0.01
    risk: float = 0.0
    every: int = 2          # seconds between our decisions
    latency: int = 2        # seconds from decision to fill
    max_orders: int = 3     # orders per 2 s step across symbols
    cost: float = 1.0
    refill: float = 0.066   # of full depth per 2 s (= 1/3 per 10 s)
    lag: int = 4            # rivals' delay behind us (s); 0 = as fast as us
    share: float = 0.3      # liquidity rivals take first
    push: float = 0.3       # fraction of their gap rivals move the price
    rival_margin: float = 0.0025


SCENARIOS = {
    "no rivals": dict(share=0.0, push=0.0),
    "base: rivals 4 s slower, 30%": dict(lag=4, share=0.3, push=0.3),
    "rivals 2 s slower, 50%": dict(lag=2, share=0.5, push=0.5),
    "rivals as fast, 50%": dict(lag=0, share=0.5, push=0.5),
    "rivals faster, 70%": dict(lag=0, share=0.7, push=0.7, latency=4),
}


# ------------------------------------------------------------------ data
def arrays(fv_file: str):
    fv = pickle.loads((OUT / fv_file).read_bytes())
    fv = fv[fv.t < 780]
    p = prices()
    ss = np.array(sorted(fv.season.unique()))
    p = p[p.season.isin(ss)]
    key = ["season", "round", "t"]
    fv = fv.sort_values(key)
    p = p.set_index(key).loc[pd.MultiIndex.from_frame(fv[key])].reset_index()
    n = len(ss)
    F = fv[[f"fv_{s}" for s in SYMBOLS]].to_numpy().reshape(n, NSTEP, 12)
    SD = fv[[f"sd_{s}" for s in SYMBOLS]].to_numpy().reshape(n, NSTEP, 12)
    P = p[SYMBOLS].to_numpy().reshape(n, NSTEP, 12)
    se = seasons().set_index("season")
    SET = np.stack([se.loc[ss, f"set_{s}"].to_numpy() for s in SYMBOLS], 1)
    return {"season": ss, "fv": F, "sd": SD, "px": P, "settle": SET}


def fit_beta(A):
    """beta[phase, sym] by regression through the origin of (settle - px) on (fv - px), clipped to [0, 1.5]."""
    x = A["fv"] - A["px"]
    y = A["settle"][:, None, :] - A["px"]
    phases = np.unique(PHASE)
    B = {}
    for ph in phases:
        m = PHASE == ph
        xx, yy = x[:, m].reshape(-1, 12), y[:, m].reshape(-1, 12)
        B[int(ph)] = np.clip((xx * yy).sum(0) / np.maximum((xx * xx).sum(0), 1e-9), 0, 1.5)
    return B


def levels_arrays():
    prof = book_profile()
    out = {}
    for reg in ("pre", "storm"):
        L = np.stack([prof[(s, reg)] for s in SYMBOLS])  # [12, 5, 2]
        out[reg] = (L[..., 0], L[..., 1])
    return out


# ------------------------------------------------------------------ simulation
def simulate(A, beta, prm: Params, book=None):
    book = book or levels_arrays()
    n = len(A["season"])
    px_all, fv_all, sd_all = A["px"], A["fv"], A["sd"]
    Bm = np.stack([beta[int(ph)] for ph in PHASE])             # [NSTEP, 12]
    V_all = px_all + Bm[None] * (fv_all - px_all)              # our value at each step
    pos = np.zeros((n, 12)); cash = np.zeros((n, 12)); trades = np.zeros((n, 12))
    avail = None
    lag_s, lat_s, ev_s = prm.lag // STEP, prm.latency // STEP, max(prm.every // STEP, 1)
    for j in range(NSTEP):
        reg = "pre" if TREAL[j] < 60 else "storm"
        off, full = book[reg]                                   # [12, 5]
        full = np.broadcast_to(full, (n, 2, 12, 5))
        if avail is None or TREAL[j] == 0:
            avail = full.copy()
        else:
            avail = np.minimum(full, avail + prm.refill * full)
        px, V = px_all[:, j], V_all[:, j]
        # rivals act on information `lag` seconds old (nothing from before this round's open)
        jr = j - lag_s
        Vr = V_all[:, jr] if (jr >= 0 and ROUND[jr] == ROUND[j]) else px
        gr = Vr - px
        act = np.abs(gr) > prm.rival_margin * px
        side_r = np.where(gr > 0, 0, 1)                         # 0 = they buy (take asks), 1 = they sell
        for sd_ in (0, 1):
            m = act & (side_r == sd_)
            avail[:, sd_][m] *= (1 - prm.share)
        push = prm.push * gr * act
        if j % ev_s:
            continue
        je = min(j + lat_s, NSTEP - 1) if ROUND[min(j + lat_s, NSTEP - 1)] == ROUND[j] else j
        pxe = px_all[:, je] + push                              # where we can actually trade
        side = np.where(V > pxe, 1.0, -1.0)
        sidx = np.where(side > 0, 0, 1)
        lp = pxe[..., None] * (1 + side[..., None] * off[None] * prm.cost)          # [n,12,5] level prices
        edge = side[..., None] * (V[..., None] - lp)                                  # per-share edge at each level
        # largest position (in our direction) still worth holding at each level price
        unit = prm.size_pen + prm.risk * sd_all[:, j][..., None] / px[..., None]
        cap = np.where(unit > 0, (edge / px[..., None] - prm.margin) / np.maximum(unit, 1e-12), np.inf) * LIM[None, :, None]
        cap = np.where(edge > prm.margin * px[..., None], np.minimum(cap, LIM[None, :, None]), -np.inf)
        av = np.take_along_axis(avail, sidx[:, None, :, None].repeat(5, 3), 1)[:, 0]  # [n,12,5]
        cur = side * pos
        q = np.zeros_like(av)
        run = cur.copy()
        for l in range(5):
            take = np.floor(np.clip(np.minimum(av[..., l], cap[..., l] - run), 0, None))
            q[..., l] = take
            run = run + take
        qty = q.sum(-1)
        value = (q * edge).sum(-1)
        if prm.max_orders < 12:  # rate limit: keep the best orders by dollar edge
            rank = np.argsort(-value, axis=1)
            keep = np.zeros_like(value, bool)
            np.put_along_axis(keep, rank[:, : prm.max_orders], True, axis=1)
            q = q * keep[..., None]
            qty = q.sum(-1)
        cash -= side * (q * lp).sum(-1)
        pos += side * qty
        trades += qty > 0
        # consume liquidity
        for sd_ in (0, 1):
            m = (sidx == sd_)[..., None]
            avail[:, sd_] = avail[:, sd_] - np.where(m, q, 0)
    pnl = cash + pos * A["settle"]
    return pnl, trades, pos


def report(pnl, label):
    per = pnl.sum(1)
    return {"scenario": label, "mean": per.mean(), "median": np.median(per), "sd": per.std(),
            "p5": np.percentile(per, 5), "win": (per > 0).mean()}


def main():
    print("loading 2 s fair-value series ...", flush=True)
    tr, ho = arrays("fv2_201-600.pkl"), arrays("fv2_1-200.pkl")
    beta = fit_beta(tr)
    book = levels_arrays()
    base = Params(**SCENARIOS["base: rivals 4 s slower, 30%"])

    print("== tuning on training seasons 201-600 under the base scenario")
    grid = []
    for m, sp, rk in itertools.product((0.0025, 0.005, 0.01), (0.0, 0.01, 0.02, 0.04), (0.0, 0.05, 0.2)):
        prm = replace(base, margin=m, size_pen=sp, risk=rk)
        pnl, _, _ = simulate(tr, beta, prm, book)
        per = pnl.sum(1)
        grid.append((per.mean(), per.std(), m, sp, rk))
    grid.sort(reverse=True)
    for g in grid[:8]:
        print(f"  margin {g[2]:<6} size_pen {g[3]:<5} risk {g[4]:<5} mean {g[0]:8.0f} sd {g[1]:7.0f}")
    _, _, m, sp, rk = grid[0]
    best = replace(base, margin=m, size_pen=sp, risk=rk)
    # the most profitable setting whose sd is at least 25% lower (a lower-variance alternative)
    top_sd = grid[0][1]
    safe = next((g for g in grid if g[1] <= 0.75 * top_sd), grid[0])
    safe_p = replace(base, margin=safe[2], size_pen=safe[3], risk=safe[4])
    print(f"chosen: margin {m} size_pen {sp} risk {rk};  lower-variance alternative: "
          f"margin {safe[2]} size_pen {safe[3]} risk {safe[4]}")

    print("\n== HOLDOUT seasons 1-200")
    rows = []
    for name, sc in SCENARIOS.items():
        for lab, p in (("tuned", best), ("lower-var", safe_p)):
            prm = replace(p, **sc)
            pnl, tr_, _ = simulate(ho, beta, prm, book)
            rows.append({**report(pnl, name), "params": lab, "orders/season": tr_.sum(1).mean()})
    old = replace(best, **SCENARIOS["base: rivals 4 s slower, 30%"], every=10, latency=4)
    pnl, tr_, _ = simulate(ho, beta, old, book)
    rows.append({**report(pnl, "base, but deciding every 10 s (old bot)"), "params": "tuned", "orders/season": tr_.sum(1).mean()})
    for every in (4, 6):
        pnl, tr_, _ = simulate(ho, beta, replace(best, every=every), book)
        rows.append({**report(pnl, f"base, deciding every {every} s"), "params": "tuned", "orders/season": tr_.sum(1).mean()})
    for mo in (1, 2, 6):
        pnl, tr_, _ = simulate(ho, beta, replace(best, max_orders=mo), book)
        rows.append({**report(pnl, f"base, {mo} orders per 2 s"), "params": "tuned", "orders/season": tr_.sum(1).mean()})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round({"mean": 0, "median": 0, "sd": 0, "p5": 0, "win": 2, "orders/season": 0}).to_string(index=False))

    pnl, _, _ = simulate(ho, beta, best, book)
    print("\nholdout pnl by symbol, base scenario:",
          dict(zip(SYMBOLS, np.round(pnl.mean(0)).astype(int).tolist())))
    (OUT / "live_params.json").write_text(json.dumps({
        "tuned": asdict(best), "lower_variance": asdict(safe_p),
        "beta": {str(k): v.tolist() for k, v in beta.items()}, "symbols": SYMBOLS,
    }, indent=1))
    print("wrote", OUT / "live_params.json")


if __name__ == "__main__":
    main()
