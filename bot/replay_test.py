"""Offline end-to-end test: run the real LiveBot against a simulated exchange that replays research seasons.

    ../.venv/bin/python replay_test.py --seasons 1-10            # holdout seasons
    ../.venv/bin/python replay_test.py --seasons 3 --verbose

The fake exchange serves the season's feed, the house price (market.csv) as the mid of a book shaped like the
testnet books, fills limit orders against that book (depth refills like in backtest/sim.py), keeps positions and
cash, and settles at the season's settlement values. Time is virtual: the bot's cycle runs on every storm message
and at least every 2 s, as live. No network is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time as _time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT / "analysis"))
import live_bot  # noqa: E402
from backtest import book_profile  # noqa: E402
from common import SYMBOLS, case  # noqa: E402

HIST = ROOT / "hurricane-season-research" / "history"
ROUND_LEN, SPAN = 780, 840


class VirtualTime:
    """Stands in for the `time` module inside live_bot (monotonic/time follow the virtual clock)."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def strftime(self, fmt):
        return f"v{self.now:7.1f}"

    def sleep(self, s):
        self.now += s


class FakeExchange:
    def __init__(self, season: int, vt: VirtualTime, refill=0.066):
        self.vt = vt
        self.case_data = json.loads((ROOT / "hurricane-season-research" / "case.json").read_text())
        man = json.loads((HIST / "manifest.json").read_text())
        self.meta = next(s for s in man["seasons"] if s["season"] == season)
        d = HIST / self.meta["dir"]
        self.feed = [json.loads(l) for l in d.open()] if False else [json.loads(l) for l in (d / "feed.jsonl").open()]
        import csv
        self.px = {}
        for row in csv.DictReader((d / "market.csv").open()):
            t = live_bot.real_t(row["time"])
            self.px[(int(row["round"]), int(t))] = {s: float(row[s]) for s in SYMBOLS}
        self.settle = json.loads((d / "settlement.json").read_text())
        self.prof = book_profile()
        self.refill = refill
        self.pos = {s: 0 for s in SYMBOLS}
        self.cash = {s: 0.0 for s in SYMBOLS}
        self.resting = []
        self.actions = []
        self.round, self.t = 1, 0.0
        self.avail = {}
        self.orders = 0

    # -- clock
    def set_time(self, rnd, t):
        new_round = rnd != self.round
        self.round, self.t = rnd, t
        if new_round:
            self.avail = {}

    def date(self):
        return self.meta["storms"][self.round - 1]["open"][:10]

    def fictional(self):
        secs = int(self.t * 30)
        h, m, s = 9 + (30 * 60 + secs) // 3600, ((30 * 60 + secs) % 3600) // 60, secs % 60
        return f"{self.date()}T{h:02d}:{m:02d}:{s:02d}"

    def mid(self, sym):
        k = int(self.t // 2 * 2)
        while (self.round, k) not in self.px and k > 0:
            k -= 2
        return self.px[(self.round, k)][sym]

    def levels(self, sym):
        reg = "pre" if self.t < 60 else "storm"
        L = self.prof[(sym, reg)]
        key = (sym, reg)
        full = L[:, 1]
        if key not in self.avail:
            self.avail[key] = {"t": self.t, 1: full.copy(), -1: full.copy()}
        a = self.avail[key]
        steps = (self.t - a["t"]) / 2
        for side in (1, -1):
            a[side] = np.minimum(full, a[side] + self.refill * steps * full)
        a["t"] = self.t
        return L[:, 0], a

    # -- API used by LiveBot
    def case(self):
        return self.case_data

    def time(self):
        return {"competition_id": f"replay-{self.meta['season']}", "state": "running" if self.t < ROUND_LEN else "ready",
                "fictional_time": self.fictional(), "elapsed_active_seconds": (self.round - 1) * SPAN + self.t}

    def positions(self):
        out = []
        for s, q in self.pos.items():
            if q or self.cash[s]:
                avg = -self.cash[s] / q if q else None   # cost basis folded into the average price
                out.append({"symbol": s, "quantity": q, "average_price": avg, "current_price": self.mid(s),
                            "realized_pnl": 0.0 if q else self.cash[s],
                            "unrealized_pnl": (self.mid(s) - avg) * q if q else 0.0})
        return {"positions": out}

    def orderbook(self, sym):
        off, a = self.levels(sym)
        p = self.mid(sym)
        return {"symbol": sym,
                "bids": [{"price": round(p * (1 - o), 2), "quantity": int(q)} for o, q in zip(off, a[-1])],
                "asks": [{"price": round(p * (1 + o), 2), "quantity": int(q)} for o, q in zip(off, a[1])]}

    def limit(self, sym, side, qty, price):
        self.actions.append(self.vt.now)
        self.orders += 1
        sgn = 1 if side == "buy" else -1
        off, a = self.levels(sym)
        p = self.mid(sym)
        left = qty
        for i, o in enumerate(off):
            lp = round(p * (1 + sgn * o), 2)
            if sgn * (price - lp) < -1e-9 or left <= 0:
                break
            take = int(min(left, a[sgn][i]))
            a[sgn][i] -= take
            left -= take
            self.pos[sym] += sgn * take
            self.cash[sym] -= sgn * take * lp
        if left > 0:
            self.resting.append((sym, side, left, price))
        return {"order_id": f"o{self.orders}", "status": "pending"}

    def _convert(self, sym, q):
        self.actions.append(self.vt.now)
        basket = self.case_data["etfs"][sym]["basket"]
        nav = sum(n * self.mid(c) for c, n in basket.items())
        self.pos[sym] += q
        self.cash[sym] -= q * nav                 # ETF booked at the basket value
        for c, n in basket.items():
            self.pos[c] -= q * n
            self.cash[c] += q * n * self.mid(c)  # stocks booked at their marks
        return {"status": "ok"}

    def create_etf(self, sym, quantity=1):
        return self._convert(sym, quantity)

    def redeem_etf(self, sym, quantity=1):
        return self._convert(sym, -quantity)

    def open_orders(self):
        return {"orders": [{"symbol": s, "side": d, "quantity": q, "price": p} for s, d, q, p in self.resting]}

    def cancel_all(self, symbol=None):
        self.actions.append(self.vt.now)
        self.resting = []
        return {"cancelled_count": 0}

    def pnl(self):
        return {s: self.cash[s] + self.pos[s] * self.settle[s] for s in SYMBOLS}


def run_season(season, profile="tuned", verbose=False):
    vt = VirtualTime()
    fx = FakeExchange(season, vt)
    live_bot.time = vt                                   # virtual clock inside the bot
    live_bot.STATE = Path(tempfile.mkdtemp())            # fresh state directory
    live_bot.Exchange = lambda *a, **k: fx
    if not verbose:
        live_bot.log = lambda *a: None
    args = SimpleNamespace(url="", tag="train", profile=profile, scale=1.0, step=2.0, min_interval=0.5,
                           actions_per_min=170, max_orders=0, skip_symbols="", samples=1500, dry_run=False, verbose=False, testnet_history=False)
    bot = live_bot.LiveBot(args)
    bot.pool = SimpleNamespace(submit=lambda f, *a: SimpleNamespace(result=lambda: f(*a)),
                               map=lambda f, xs: [f(x) for x in xs])   # sequential: deterministic
    bot.set_competition(fx.time()["competition_id"])
    feed = fx.feed
    k = 0
    for rnd in (1, 2, 3):
        msgs = [m for m in feed if m["round"] == rnd]
        k = 0
        last_cycle = -99.0
        for t2 in range(0, ROUND_LEN, 1):            # 1 s virtual resolution
            t = float(t2)
            fx.set_time(rnd, t)
            vt.now = (rnd - 1) * SPAN + t
            news = False
            while k < len(msgs) and live_bot.real_t(msgs[k]["time"]) <= t:
                bot.on_news(msgs[k])
                news = True
                k += 1
            for s in SYMBOLS:
                p = fx.mid(s)
                bot.quotes[s] = {"bid": p * 0.999, "ask": p * 1.001}
            if (news and t - last_cycle >= 0.5) or t - last_cycle >= 2.0:
                bot.step()
                last_cycle = t
    pnl = fx.pnl()
    acts = np.array(fx.actions)
    peak = max((np.sum((acts > a - 60) & (acts <= a)) for a in acts), default=0)
    return pnl, fx.orders, peak


def run_combo(season, mode, main_budget=170, arb_budget=50):
    """mode: 'main' (alone, trades ETFs), 'main-noetf', 'main+arb' (main skips ETFs + arb bot),
    'mainetf+arb' (main keeps trading ETFs + arb bot)."""
    import etf_arb_bot
    vt = VirtualTime()
    fx = FakeExchange(season, vt)
    live_bot.time = vt
    live_bot.STATE = Path(tempfile.mkdtemp())
    live_bot.Exchange = lambda *a, **k: fx
    live_bot.log = lambda *a: None
    etf_arb_bot.time = vt
    etf_arb_bot.Exchange = lambda *a, **k: fx
    etf_arb_bot.STATE = live_bot.STATE
    etf_arb_bot.log = lambda *a: None
    seq = SimpleNamespace(submit=lambda f, *a: SimpleNamespace(result=lambda: f(*a)), map=lambda f, xs: [f(x) for x in xs])
    args = SimpleNamespace(url="", tag="train", profile="tuned", scale=1.0, step=2.0, min_interval=0.5,
                           actions_per_min=main_budget, max_orders=0,
                           skip_symbols="ARCH,REBLD" if mode in ("main-noetf", "main+arb") else "", samples=1500,
                           dry_run=False, verbose=False, testnet_history=False)
    bot = live_bot.LiveBot(args)
    bot.pool = seq
    bot.set_competition(fx.time()["competition_id"])
    arb = None
    if mode.endswith("+arb"):
        arb = etf_arb_bot.ArbBot(SimpleNamespace(url="", min_edge=None, actions_per_min=arb_budget, min_interval=0.3,
                                                 max_units=0, dry_run=False))
        arb.pool = seq
        arb.state = "running"
    for rnd in (1, 2, 3):
        msgs = [m for m in fx.feed if m["round"] == rnd]
        k, last_cycle = 0, -99.0
        for t2 in range(0, ROUND_LEN):
            t = float(t2)
            fx.set_time(rnd, t)
            vt.now = (rnd - 1) * SPAN + t
            news = False
            while k < len(msgs) and live_bot.real_t(msgs[k]["time"]) <= t:
                bot.on_news(msgs[k]); news = True; k += 1
            for s in SYMBOLS:
                b = fx.orderbook(s)
                q = {"bid": b["bids"][0]["price"], "ask": b["asks"][0]["price"]}
                bot.quotes[s] = q
                if arb:
                    arb.quotes[s] = q
            if (news and t - last_cycle >= 0.5) or t - last_cycle >= 2.0:
                bot.step()
                last_cycle = t
            if arb:
                arb.last_try = {e: -1e9 for e in arb.etfs}
                arb.tick()
    pnl = fx.pnl()
    out = {"total": sum(pnl.values()), "etf": pnl["ARCH"] + pnl["REBLD"]}
    if arb:
        out.update(arb_stats=dict(arb.stats), pending=dict(arb.pending))
    return out


def _combo(job):
    return run_combo(*job)


def _one(job):
    season, profile = job
    pnl, n, peak = run_season(season, profile)
    return sum(pnl.values()), n, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="1-10")
    ap.add_argument("--profile", default="tuned")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--combo", action="store_true", help="compare main / main-noetf / main+arb")
    a = ap.parse_args()
    lo, hi = (map(int, a.seasons.split("-")) if "-" in a.seasons else (int(a.seasons), int(a.seasons)))
    tot = []
    if a.combo:
        from concurrent.futures import ProcessPoolExecutor
        for mode, mb, ab in (("main", 170, 0), ("main-noetf", 170, 0), ("main+arb", 125, 50), ("mainetf+arb", 125, 50)):
            jobs = [(s, mode, mb, ab) for s in range(lo, hi + 1)]
            with ProcessPoolExecutor(max(a.workers, 1)) as ex:
                res = list(ex.map(_combo, jobs))
            tot = np.array([r["total"] for r in res])
            line = f"{mode:11s} {len(tot)} seasons: mean {tot.mean():8.0f}  median {np.median(tot):8.0f}  5% {np.percentile(tot, 5):8.0f}"
            line = f"{mode:11s} budget {mb}+{ab}: " + line[12:]
            if mode.endswith("+arb"):
                st = [r["arb_stats"] for r in res]
                line += (f"  | arb: {np.mean([x['arbs'] for x in st]):.0f} arbs/season, est ${np.mean([x['est_profit'] for x in st]):,.0f}, "
                         f"unwind orders {np.mean([x['unwinds'] for x in st]):.1f}, pending at end {sum(abs(v) for r in res for v in r['pending'].values())}")
            print(line, flush=True)
        return
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_one, [(s, a.profile) for s in range(lo, hi + 1)]))
        tot = np.array([r[0] for r in res])
        peaks = [r[2] for r in res]
        print(f"{len(tot)} seasons: mean {tot.mean():.0f}  median {np.median(tot):.0f}  sd {tot.std():.0f}  "
              f"5% {np.percentile(tot, 5):.0f}  win {np.mean(tot > 0):.0%}  max actions/min {max(peaks)}  "
              f"orders/season {np.mean([r[1] for r in res]):.0f}")
        return
    for s in range(lo, hi + 1):
        t0 = _time.time()
        pnl, n, peak = run_season(s, a.profile, a.verbose)
        tot.append(sum(pnl.values()))
        print(f"season {s:3d}: pnl {tot[-1]:9.0f}  orders {n:4d}  peak actions/min {peak:3d}  "
              f"({_time.time() - t0:.0f}s)  top: " +
              ", ".join(f"{k} {v:+.0f}" for k, v in sorted(pnl.items(), key=lambda kv: -abs(kv[1]))[:3]), flush=True)
    print(f"mean {np.mean(tot):.0f}  median {np.median(tot):.0f}  over {len(tot)} seasons")


if __name__ == "__main__":
    main()
