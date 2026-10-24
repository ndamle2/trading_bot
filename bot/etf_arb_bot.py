"""ETF arbitrage bot: trades ONLY the ETF / basket mispricing and always ends flat.

    ../.venv/bin/python etf_arb_bot.py --dry-run          # log opportunities, send nothing
    ../.venv/bin/python etf_arb_bot.py                    # live (normally started by ./run_bot.sh --with-arb)

When an ETF's bid beats the cost of buying its basket (or the basket's bids beat the ETF's ask) by more than
min_edge of the basket value per unit (backtest/out/etf_arb_params.json):
  1. take the shared trade lock (bot/coord.py) so the main bot is not sending orders at the same moment
  2. read the books and positions, size n units by walking every book (marginal unit must clear min_edge) and the
     position-limit room left by the other bot
  3. send the ETF leg and all basket legs in parallel as marketable limit orders; cancel any remainder
  4. measure fills from the change in positions; CREATE (rich ETF) or REDEEM (cheap ETF) the matched units so every
     position returns to where it was; unwind any unmatched shares at once
If a create/redeem is refused, the hedged units are kept (an ETF settles at its basket value, so the hedge is
locked to settlement) and retried later. Overrides: {"arb": {"halt": true}} in bot/overrides.json stops it.
Writes bot/state/arb_status.json for the dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from client import ApiError, Exchange  # noqa: E402
from coord import TradeLock  # noqa: E402
from live_bot import ActionBudget  # noqa: E402

STATE = HERE / "state"
OVERRIDES = HERE / "overrides.json"
PARAMS = ROOT / "backtest" / "out" / "etf_arb_params.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[arb]", *a, flush=True)


class ArbBot:
    def __init__(self, args):
        self.args = args
        self.ex = Exchange(args.url, actions_per_min=290)
        self.case = self.ex.case()
        self.limits = self.case["limits"]
        self.etfs = {e: spec["basket"] for e, spec in self.case["etfs"].items()}
        p = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
        self.min_edge = args.min_edge if args.min_edge is not None else p.get("min_edge", 0.001)
        self.budget = ActionBudget(args.actions_per_min)
        self.pool = ThreadPoolExecutor(12)
        self.lock = TradeLock(timeout=0.0)
        self.quotes: dict[str, dict] = {}
        self.last_try = {e: 0.0 for e in self.etfs}
        self.pending: dict[str, int] = {}        # hedged units waiting for a create (+) / redeem (-)
        self.stats = {"arbs": 0, "units": 0, "est_profit": 0.0, "unwinds": 0, "refused": 0, "skipped_lock": 0}
        self.recent = deque(maxlen=40)
        self.gaps = {}
        self.state = None
        self.halted = False
        self._last_status = 0.0
        STATE.mkdir(exist_ok=True)

    # ------------------------------------------------------------ helpers
    def positions(self):
        resp = self.ex.positions()
        rows = resp.get("positions", resp) if isinstance(resp, dict) else resp
        out = {s: 0 for s in self.limits}
        for r in rows or []:
            if r.get("symbol") in out:
                out[r["symbol"]] = int(next((r[k] for k in ("quantity", "position", "qty") if k in r), 0))
        return out

    @staticmethod
    def walk(levels, shares):
        """(worst price, total cost) to take `shares` through levels; None if the book is too thin."""
        left, cost, worst = shares, 0.0, None
        for lv in levels:
            take = min(left, int(lv["quantity"]))
            if take <= 0:
                continue
            cost += take * float(lv["price"])
            worst = float(lv["price"])
            left -= take
            if left <= 0:
                return worst, cost
        return None

    def top_edge(self, etf):
        """Per-unit edge at the top of book from the quote stream: (+) ETF rich, (-) ETF cheap, and basket mid."""
        b = self.etfs[etf]
        q = {s: self.quotes.get(s) for s in [etf, *b]}
        if any(not x or x.get("bid") is None or x.get("ask") is None for x in q.values()):
            return None
        nav = sum(n * (q[s]["bid"] + q[s]["ask"]) / 2 for s, n in b.items())
        rich = q[etf]["bid"] - sum(n * q[s]["ask"] for s, n in b.items())
        cheap = sum(n * q[s]["bid"] for s, n in b.items()) - q[etf]["ask"]
        return rich, cheap, nav

    def size(self, etf, side, books, pos, nav):
        """Largest n whose marginal unit clears min_edge, within position-limit room. side +1 = ETF rich."""
        b = self.etfs[etf]
        # ETF rich: sell ETF (bids), buy basket (asks); cheap: buy ETF (asks), sell basket (bids)
        etf_lv = books[etf]["bids" if side > 0 else "asks"]
        room = self.limits[etf] + (pos[etf] if side > 0 else -pos[etf])
        for s, n in b.items():
            r = self.limits[s] - (pos[s] if side > 0 else -pos[s])
            room = min(room, r // n)
        room = min(room, self.args.max_units or room)
        best, prev_e, prev_b = 0, 0.0, 0.0
        for u in range(1, max(room, 0) + 1):
            w = self.walk(etf_lv, u)
            if w is None:
                break
            legs = [self.walk(books[s]["asks" if side > 0 else "bids"], u * n) for s, n in b.items()]
            if any(x is None for x in legs):
                break
            e_cost, b_cost = w[1], sum(x[1] for x in legs)
            marg = ((e_cost - prev_e) - (b_cost - prev_b)) if side > 0 else ((b_cost - prev_b) - (e_cost - prev_e))
            if marg <= self.min_edge * nav:
                break
            best, prev_e, prev_b = u, e_cost, b_cost
        if not best:
            return None
        plan = {etf: (("sell" if side > 0 else "buy"), best, self.walk(etf_lv, best)[0])}
        for s, n in b.items():
            plan[s] = (("buy" if side > 0 else "sell"), best * n, self.walk(books[s]["asks" if side > 0 else "bids"], best * n)[0])
        est = (prev_e - prev_b) if side > 0 else (prev_b - prev_e)
        return best, plan, est

    # ------------------------------------------------------------ one arbitrage
    def try_arb(self, etf):
        te = self.top_edge(etf)
        if te is None:
            return
        rich, cheap, nav = te
        self.gaps[etf] = {"rich": rich, "cheap": cheap, "nav": nav, "ts": time.time()}
        side = 1 if rich > self.min_edge * nav else -1 if cheap > self.min_edge * nav else 0
        if not side or self.halted:
            return
        need = 2 + len(self.etfs[etf]) + 1          # legs + create/redeem + one cancel-all
        if self.budget.available() < need:
            return
        if not self.lock.acquire():
            self.stats["skipped_lock"] += 1
            return
        try:
            self._arb(etf, side, nav)
        finally:
            self.lock.release()

    def _arb(self, etf, side, nav):
        syms = [etf, *self.etfs[etf]]
        f_pos = self.pool.submit(self.positions)
        books = dict(zip(syms, self.pool.map(self.ex.orderbook, syms)))
        pos0 = f_pos.result()
        sized = self.size(etf, side, books, pos0, nav)
        if not sized:
            return
        n, plan, est = sized
        log(f"{etf} {'rich' if side > 0 else 'cheap'}: {n} units, est ${est:,.2f} "
            + " ".join(f"{s}:{d[0][0]}{d[1]}@{d[2]:.2f}" for s, d in plan.items()))
        if self.args.dry_run:
            self.recent.appendleft({"ts": time.time(), "etf": etf, "dir": "rich" if side > 0 else "cheap",
                                    "units": n, "est": est, "matched": 0, "dry": True})
            return
        if not self.budget.take(len(plan)):
            return

        def send(item):
            s, (d, q, px) = item
            try:
                return self.ex.limit(s, d, q, px)
            except ApiError as e:
                log(f"  {s} rejected: {e}")
        list(self.pool.map(send, plan.items()))
        resting = (self.ex.open_orders() or {}).get("orders", [])
        if resting and self.budget.take(1):
            self.ex.cancel_all()
        pos1 = self.positions()
        fill = {s: pos1[s] - pos0[s] for s in syms}
        etf_units = -fill[etf] if side > 0 else fill[etf]
        leg_units = min(((fill[s] if side > 0 else -fill[s]) // q) for s, q in self.etfs[etf].items())
        m = max(0, min(etf_units, leg_units))
        pend_before = self.pending.get(etf, 0)
        if m:
            self.convert(etf, m if side > 0 else -m)
        pend_after = self.pending.get(etf, 0)
        # target: where we started, plus any hedge we now deliberately keep because a create/redeem was refused
        keep = self.hedge_effect(etf, pend_after - pend_before)
        pos2 = self.positions() if m else pos1
        resid = {s: pos2[s] - pos0[s] - keep.get(s, 0) for s in syms}
        self.unwind(resid, books)
        prof = est * m / n
        self.stats["arbs"] += 1 if m else 0
        self.stats["units"] += m
        self.stats["est_profit"] += prof
        self.recent.appendleft({"ts": time.time(), "etf": etf, "dir": "rich" if side > 0 else "cheap", "units": n,
                                "matched": m, "est": prof, "fills": fill})
        log(f"  matched {m}/{n} units, est profit ${prof:,.2f}; residual {dict((k, v) for k, v in resid.items() if v)}")

    def hedge_effect(self, etf, k):
        """Position change of k hedged units not converted: k > 0 = short k ETF + long k baskets (sold it rich)."""
        if not k:
            return {}
        eff = {etf: -k}
        for sym, q in self.etfs[etf].items():
            eff[sym] = k * q
        return eff

    def convert(self, etf, units):
        """units > 0: create (give basket, get ETF); units < 0: redeem. On refusal keep it pending."""
        total = units + self.pending.get(etf, 0)
        if not total or not self.budget.take(1):
            self.pending[etf] = total
            return
        try:
            if total > 0:
                self.ex.create_etf(etf, total)
            else:
                self.ex.redeem_etf(etf, -total)
            self.pending[etf] = 0
        except ApiError as e:
            self.pending[etf] = total
            self.stats["refused"] += 1
            log(f"  {'create' if total > 0 else 'redeem'} {abs(total)} {etf} refused ({e}); hedge kept, will retry")

    def unwind(self, resid, books):
        orders = []
        for s, r in resid.items():
            if not r:
                continue
            levels = books[s]["bids" if r > 0 else "asks"]
            w = self.walk(levels, abs(r))
            px = w[0] if w else float(levels[-1]["price"]) if levels else None
            if px is None:
                continue
            px = px * (0.99 if r > 0 else 1.01)                    # marketable: make sure it goes
            orders.append((s, "sell" if r > 0 else "buy", abs(r), round(px, 2)))
        if not orders or not self.budget.take(len(orders)):
            if orders:
                log(f"  WARNING: no budget to unwind {orders}")
            return
        self.stats["unwinds"] += len(orders)
        for s, d, q, px in orders:
            try:
                self.ex.limit(s, d, q, px)
            except ApiError as e:
                log(f"  unwind {s} rejected: {e}")
        if (self.ex.open_orders() or {}).get("orders") and self.budget.take(1):
            self.ex.cancel_all()

    # ------------------------------------------------------------ status / loops
    def read_overrides(self):
        try:
            o = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
        except json.JSONDecodeError:
            o = {}
        self.halted = bool(o.get("halt")) or bool((o.get("arb") or {}).get("halt"))

    def write_status(self, force=False):
        now = time.time()
        if not force and now - self._last_status < 1.0:
            return
        self._last_status = now
        st = {"ts": now, "pid": os.getpid(), "dry_run": self.args.dry_run, "halted": self.halted,
              "state": self.state, "min_edge": self.min_edge,
              "budget": {"limit": self.args.actions_per_min, "left": self.budget.available()},
              "stats": self.stats, "pending": self.pending,
              "gaps": {e: {**g, "rich_pct": 100 * g["rich"] / g["nav"], "cheap_pct": 100 * g["cheap"] / g["nav"]}
                       for e, g in self.gaps.items()},
              "recent": list(self.recent)[:25]}
        tmp = STATE / "arb_status.json.tmp"
        tmp.write_text(json.dumps(st, default=float))
        os.replace(tmp, STATE / "arb_status.json")

    def tick(self, symbol=None):
        for etf, b in self.etfs.items():
            if symbol is not None and symbol != etf and symbol not in b:
                continue
            if time.time() - self.last_try[etf] < self.args.min_interval:
                continue
            self.last_try[etf] = time.time()
            if self.pending.get(etf):
                self.convert(etf, 0)               # retry a refused create/redeem
            self.try_arb(etf)

    async def run(self):
        loop = asyncio.get_running_loop()
        busy = asyncio.Lock()

        async def clock():
            while True:
                try:
                    t = await asyncio.to_thread(self.ex.time)
                    self.state = t.get("state")
                    self.read_overrides()
                except OSError as e:
                    log("network error:", e)
                    self.state = "network down"
                self.write_status(force=True)
                await asyncio.sleep(2)

        async def feed():
            async for fr in self.ex.stream(["quotes"], list(self.limits)):
                if fr.get("type") != "quote":
                    continue
                self.quotes[fr["symbol"]] = fr
                if self.state != "running" or busy.locked():
                    continue
                async with busy:
                    try:
                        await asyncio.to_thread(self.tick, fr["symbol"])
                    except OSError as e:
                        log("network error:", e)
                    except Exception:
                        log("arb failed:\n" + traceback.format_exc())
                self.write_status()

        await asyncio.gather(clock(), feed())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="https://cornelltradingcompetition.org")
    ap.add_argument("--min-edge", type=float, default=None, help="per-unit edge as a fraction of the basket value")
    ap.add_argument("--actions-per-min", type=int, default=50, help="this bot's share of the team's 180/min")
    ap.add_argument("--min-interval", type=float, default=0.3, help="seconds between attempts per ETF")
    ap.add_argument("--max-units", type=int, default=0, help="cap units per arbitrage (0 = limit)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        asyncio.run(ArbBot(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
