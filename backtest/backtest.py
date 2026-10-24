"""Backtest the fair-value strategy against the house price series, with a book-based cost model.

    ../.venv/bin/python backtest.py                    # tune on 201-600, report once on the holdout 1-200
    ../.venv/bin/python backtest.py --lag 10 --cost 2  # stress: slower fills, double costs

Signal   edge = beta[sym, round, phase] * (fair value - house price). beta and the residual sd are fitted on the
         training seasons by regressing (settlement - house) on (fair value - house), so the edge is shrunk
         toward the house where the house is better informed.
Orders   limit order at V -/+ (margin + size_pen * |position| / limit) * price, where V = px + edge: takes every
         book level priced through it, rest cancelled; positions are otherwise held to settlement
Fills    decide at t with information up to t, execute at the house price at t + lag, walking a book whose
         shape comes from testnet snapshots (bot/logs/books.jsonl). Storm-time spreads apply from storm
         formation on. The book refills `refill` of its depth per 10 s step.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from common import SYMBOLS, case, prices, seasons  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
LIMITS = case()["limits"]
pd.set_option("display.width", 220)


# ------------------------------------------------------------------ cost model
def book_profile(path=ROOT / "bot" / "logs" / "books.jsonl", n_levels=5):
    """Median ask-side levels (offset from mid as a fraction, quantity) per symbol, for 'pre' and 'storm'."""
    rows = defaultdict(list)
    for line in open(path):
        b = json.loads(line)
        if not (b["bids"] and b["asks"]):
            continue
        t = datetime.fromisoformat(b["clock"])
        rt = ((t.hour - 9) * 3600 + (t.minute - 30) * 60 + t.second) / 30
        mid = (b["bids"][0]["price"] + b["asks"][0]["price"]) / 2
        lv = [(l["price"] / mid - 1, l["quantity"]) for l in b["asks"]][:n_levels]
        if len(lv) == n_levels:
            rows[(b["symbol"], "pre" if rt < 60 else "storm")].append(lv)
    prof = {}
    for s in SYMBOLS:
        for ph in ("pre", "storm"):
            L = rows.get((s, ph)) or rows[(s, "storm")]
            prof[(s, ph)] = np.median(np.array(L), axis=0)  # [levels, (offset, qty)]
    return prof


def fill_cost(levels, qty, avail, cost_mult):
    """Average offset (fraction of price) to trade qty shares through levels with available quantities."""
    left, cost, done = qty, 0.0, 0
    for (off, _), a in zip(levels, avail):
        take = min(left, a)
        cost += take * off * cost_mult
        done += take
        left -= take
        if left <= 0:
            break
    return done, cost


# ------------------------------------------------------------------ data
def load(fv_file):
    fv = pickle.loads((OUT / fv_file).read_bytes())
    p = prices()
    p = p[p.season.isin(fv.season.unique())][["season", "round", "t"] + SYMBOLS]
    se = seasons().set_index("season")
    return fv, p.set_index(["season", "round", "t"]).sort_index(), se


def phase(rnd, t):
    if t < 60:
        b = 0
    elif t < 540:
        b = 1 + int((t - 60) // 60)
    elif t < 570:
        b = 9
    elif t < 740:
        b = 10
    else:
        b = 11
    return rnd * 100 + b


def frame(fv, p, se, lag):
    """One row per (season, round, t, symbol): fv, sd, house price at t, house at t+lag, settlement."""
    out = []
    pt = p.reset_index()
    for s in SYMBOLS:
        a = fv[["season", "round", "t", f"fv_{s}", f"sd_{s}"]].rename(columns={f"fv_{s}": "fv", f"sd_{s}": "sd"})
        a = a.merge(pt[["season", "round", "t", s]].rename(columns={s: "px"}), on=["season", "round", "t"])
        lagged = pt[["season", "round", "t", s]].rename(columns={s: "px_exec"})
        lagged["t"] = lagged["t"] - lag
        a = a.merge(lagged, on=["season", "round", "t"], how="left")
        a["px_exec"] = a["px_exec"].fillna(a["px"])
        a["settle"] = a.season.map(se[f"set_{s}"])
        a["sym"] = s
        out.append(a)
    df = pd.concat(out, ignore_index=True)
    df["phase"] = [phase(r, t) for r, t in zip(df["round"], df["t"])]
    return df


def fit_signal(df):
    """beta and residual sd per (sym, phase): settle - px = beta * (fv - px) + e."""
    par = {}
    for (s, ph), g in df.groupby(["sym", "phase"]):
        x, y = (g.fv - g.px).to_numpy(), (g.settle - g.px).to_numpy()
        beta = float(np.clip(np.dot(x, y) / max(np.dot(x, x), 1e-9), 0, 1.5))
        par[(s, ph)] = (beta, float(np.std(y - beta * x)))
    return par


# ------------------------------------------------------------------ simulation
def simulate(df, par, prof, margin=0.002, cost_mult=1.0, refill=0.33, naive=False, phases=None, skew=0.0,
             size_pen=0.0):
    """PnL per season and symbol for the limit-order rule.

    Each step the strategy computes the shrunk value V = px + beta * (fv - px) and sends a limit buy at
    V - margin*px (or a limit sell at V + margin*px). The order takes every book level priced through
    the limit, up to the position limit, and any remainder is cancelled. Positions are held otherwise.
    size_pen: each share needs margin + size_pen * |position after the share| / limit of edge (fraction of price).
    skew: the house quotes move against our inventory by skew * (position / limit) * price (permanent impact).
    """
    res = []
    df = df.sort_values(["season", "sym", "round", "t"])
    for (season, s), g in df.groupby(["season", "sym"], sort=False):
        lim = LIMITS[s]
        pos, cash, trades, traded = 0, 0.0, 0, 0
        avail = {1: None, -1: None}
        cols = [g[c].to_numpy() for c in ("round", "t", "fv", "px", "px_exec", "phase", "sd")]
        for rnd, t, fv, px, pxe, ph, sd in zip(*cols):
            levels = prof[(s, "pre" if t < 60 else "storm")]
            full = levels[:, 1]
            for side in avail:
                avail[side] = full.copy() if avail[side] is None else np.minimum(full, avail[side] + refill * full)
            if phases is not None and ph % 100 not in phases:
                continue
            beta = 1.0 if naive else par[(s, ph)][0]
            V = px + beta * (fv - px)
            side = 1 if V > px else -1
            room = lim - side * pos
            if room <= 0:
                continue
            chunk = max(1, lim // 20)
            done, notional = 0, 0.0
            for i, (off, _) in enumerate(levels):
                stop = False
                while avail[side][i] >= 1 and done < room:
                    q = int(min(chunk, room - done, avail[side][i]))
                    new_pos = pos + side * (done + q)
                    level_px = pxe * (1 + side * off * cost_mult + skew * new_pos / lim)
                    need = (margin + size_pen * max(side * new_pos, 0) / lim) * px
                    if side * (V - level_px) <= need:
                        stop = True
                        break
                    avail[side][i] -= q
                    done += q
                    notional += q * level_px
                if stop or done >= room:
                    break
            if done:
                cash -= side * notional
                pos += side * done
                trades += 1
                traded += done
        res.append({"season": season, "sym": s, "pnl": cash + pos * g.settle.iloc[0], "trades": trades,
                    "shares": traded, "final_pos": pos})
    return pd.DataFrame(res)


def summary(res, label):
    per = res.groupby("season").pnl.sum()
    print(f"{label:34s} mean/season {per.mean():9.0f}  sd {per.std():8.0f}  "
          f"5%/50%/95% {per.quantile(.05):8.0f} {per.median():8.0f} {per.quantile(.95):8.0f}  "
          f"win {np.mean(per > 0):.0%}  sharpe/season {per.mean() / per.std():.2f}")
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=4, help="real seconds from decision to fill")
    ap.add_argument("--cost", type=float, default=1.0, help="multiplier on book offsets")
    ap.add_argument("--refill", type=float, default=0.33)
    a = ap.parse_args()

    prof = book_profile()
    fv_tr, p_tr, se = load("fv_train_201-600.pkl")
    fv_ho, p_ho, _ = load("fv_train_1-200.pkl")
    tr, ho = frame(fv_tr, p_tr, se, a.lag), frame(fv_ho, p_ho, se, a.lag)
    par = fit_signal(tr)

    print("== tuning margin on training seasons 201-600")
    best = None
    for margin in (0.0025, 0.005, 0.01):
        for size_pen in (0.0, 0.01, 0.02, 0.04, 0.08):
            res = simulate(tr, par, prof, margin, a.cost, a.refill, size_pen=size_pen)
            per = res.groupby("season").pnl.sum()
            print(f"  margin={margin:<6} size_pen={size_pen:<5} mean pnl/season {per.mean():8.0f}  sd {per.std():7.0f}"
                  f"  trades/season {res.groupby('season').trades.sum().mean():5.0f}", flush=True)
            if best is None or per.mean() > best[0]:
                best = (per.mean(), margin, size_pen)
    _, margin, size_pen = best
    print(f"chosen margin={margin} size_pen={size_pen}")

    print(f"\n== HOLDOUT seasons 1-200 (lag {a.lag}s, cost x{a.cost}, refill {a.refill}, margin {margin})")
    res = simulate(ho, par, prof, margin, a.cost, a.refill, size_pen=size_pen)
    summary(res, "strategy")
    summary(simulate(ho, par, prof, margin, a.cost, a.refill, naive=True, size_pen=size_pen), "naive (beta=1)")
    summary(simulate(ho, par, prof, margin, 2 * a.cost, a.refill, size_pen=size_pen), "stress: cost x2")
    summary(simulate(ho, par, prof, margin, a.cost, 0.1, size_pen=size_pen), "stress: refill 0.1")
    for sk in (0.01, 0.03):
        summary(simulate(ho, par, prof, margin, a.cost, a.refill, skew=sk, size_pen=size_pen), f"stress: inventory skew {sk:.0%} at limit")
    print("\nholdout pnl by symbol (mean per season):")
    print(res.groupby("sym")[["pnl", "trades", "shares"]].mean().round(0).sort_values("pnl", ascending=False).to_string())
    print("\nholdout pnl by phase (each phase traded alone):")
    names = {0: "pre-storm", 1: "storm 0-60s", 2: "storm 60-120", 3: "storm 120-180", 4: "storm 180-240",
             5: "storm 240-300", 6: "storm 300-360", 7: "storm 360-420", 8: "storm 420-480",
             9: "dissipated", 10: "after prelim", 11: "after final"}
    for b, n in names.items():
        r = simulate(ho, par, prof, margin, a.cost, a.refill, phases={b}, size_pen=size_pen)
        print(f"  {n:14s} {r.groupby('season').pnl.sum().mean():8.0f}")

    # where the money comes from: pnl attributed by round of entry is not tracked; show by phase instead
    OUT.mkdir(exist_ok=True)
    (OUT / "strategy_params.json").write_text(json.dumps({
        "margin": margin, "size_pen": size_pen, "lag": a.lag,
        "signal": {f"{s}|{ph}": v for (s, ph), v in par.items()},
    }, indent=1))
    res.to_pickle(OUT / "holdout_results.pkl")
    print(f"\nwrote {OUT / 'strategy_params.json'}")


if __name__ == "__main__":
    main()
