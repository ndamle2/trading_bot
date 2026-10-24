"""Live trading bot: the backtested fair-value strategy (backtest/sim.py) on the exchange.

    ../.venv/bin/python live_bot.py --dry-run          # compute and print orders, send nothing
    ../.venv/bin/python live_bot.py --scale 0.2        # trade at 20% of the position limits
    ../.venv/bin/python live_bot.py                    # full size, profile "tuned"
    ../.venv/bin/python live_bot.py --profile lower_variance

A trading cycle runs on every storm message (every 2 s), at once on reports / advisories, and every --step s
otherwise. Each cycle, for each symbol:
  px   = mid of the house quote,  V = px + beta[phase][sym] * (engine fair value - px)
  walk the order book and take every level priced through V -/+ need, where
  need = (margin + (size_pen + risk * sd / px) * |position| / limit) * px    (params: backtest/out/live_params.json)
Orders are ranked by dollar edge and sent in parallel within a rolling action budget (team limit 180/min); each
order is one action and, if anything rests afterwards, ONE cancel-all clears it.

State: every news message is appended to bot/state/<competition_id>.jsonl and replayed on restart. The round comes
from the exchange clock (elapsed_active_seconds), so the bot trades round 1 before the storm forms and refuses to
trade when it joined a season after round 1 without saved history.

Overrides (bot/overrides.json, re-read every cycle; edit it while the bot runs to react to announcements):
  {"halt": false,                                   # true: cancel all orders, trade nothing
   "scale": 1.0,                                    # multiplies --scale
   "symbols": {"HAVEN": {"mult": 1.0, "add": 0.0,   # fair value -> value * mult + add (ETFs follow)
                         "params": {"attach": 500},  # replace valuation-formula parameters (see engine/model.py)
                         "disable": false},          # stop trading one symbol
               "_loss_coef": {"params": {"coef": [9, 6, 2.5, 7, 1.2]}}}}   # insured $M per damage point
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from client import ApiError, Exchange, order_id  # noqa: E402
from coord import TradeLock  # noqa: E402
from engine.model import SYMBOLS, FairValue  # noqa: E402

STATE = HERE / "state"
STATUS_EVERY = 1.0  # seconds between dashboard snapshots
OVERRIDES = HERE / "overrides.json"
PARAMS = ROOT / "backtest" / "out" / "live_params.json"
ROUND_SPAN = 840  # active seconds per round on the exchange clock (780 trading + 60 before the open)
KNOWN = {"storm", "track", "advisory", "impact", "damage_report", "storm_formed", "storm_dissipated", "snapshot"}


def real_t(ts: str) -> float:
    t = datetime.fromisoformat(ts)
    return ((t.hour - 9) * 3600 + (t.minute - 30) * 60 + t.second) / 30


def phase(rnd: int, t: float) -> int:
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


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class LiveBot:
    def __init__(self, args):
        self.args = args
        self.ex = Exchange(args.url, actions_per_min=290)  # client limiter is only a safety net; ActionBudget rules
        self.case = self.ex.case()
        self.limits = self.case["limits"]
        lp = json.loads(PARAMS.read_text())
        prof = lp[args.profile]
        self.margin, self.size_pen, self.risk = prof["margin"], prof["size_pen"], prof["risk"]
        self.beta = {int(k): dict(zip(lp["symbols"], v)) for k, v in lp["beta"].items()}
        self.budget = ActionBudget(args.actions_per_min)
        self.skip = {x.strip().upper() for x in (getattr(args, "skip_symbols", "") or "").split(",") if x.strip()}
        self.trade_lock = TradeLock(timeout=0.3)
        self.pool = ThreadPoolExecutor(16)
        self.clock = None
        self.wake = None  # asyncio.Event, created in run()
        self.loop = None
        self.fv = FairValue.load(args.tag, n_samples=args.samples)
        self.quotes: dict[str, dict] = {}
        self.positions: dict[str, int] = {s: 0 for s in SYMBOLS}
        self.comp_id = None
        self.state_file = None
        self.last_msg_time = None
        self.overrides = {}
        self.seen_rounds: set = set()
        self.seen_headlines: set = set()
        self.pos_rows: dict[str, dict] = {}
        self.recent_orders: deque = deque(maxlen=60)
        self.cycle_ms = None
        self._last_status = 0.0
        self._trading = False
        self._last_reason = None
        self._net_down = False
        STATE.mkdir(exist_ok=True)

    # ------------------------------------------------------------ state
    def set_competition(self, comp_id: str):
        if comp_id == self.comp_id:
            return
        self.comp_id = comp_id
        self.fv.reset_season()
        self.state_file = STATE / f"{comp_id}.jsonl"
        if self.args.testnet_history and not self.state_file.exists():
            self.seed_from_research()
        self.seen_rounds, self.last_msg_time, n = set(), None, 0
        if self.state_file.exists():
            for line in self.state_file.open():
                m = json.loads(line)
                self.fv.on_message(m)
                self.mark(m)
                n += 1
        log(f"competition {comp_id}: replayed {n} saved news messages (round {self.fv.round}, "
            f"rounds seen from formation: {sorted(self.seen_rounds)})")

    def mark(self, m: dict):
        """Remember the rounds whose storm we saw from formation (complete history for that round)."""
        if m.get("type") == "storm_formed" or (m.get("type") == "storm" and m.get("age", 999) <= 4):
            self.seen_rounds.add(m.get("round"))
        if m.get("time"):
            self.last_msg_time = m["time"]

    def seed_from_research(self):
        """TESTNET ONLY: the testnet replays research seasons exactly, so rebuild missed messages from feed.jsonl."""
        clock = self.ex.time()
        date = clock["fictional_time"][:10]
        hist = ROOT / "hurricane-season-research" / "history"
        man = json.loads((hist / "manifest.json").read_text())
        season = next((s for s in man["seasons"] if any(st["open"].startswith(date) for st in s["storms"])), None)
        if season is None:
            log(f"testnet history: no research season on {date}")
            return
        now = clock["fictional_time"]
        n = 0
        with self.state_file.open("w") as f:
            for line in (hist / season["dir"] / "feed.jsonl").open():
                m = json.loads(line)
                if m["time"] <= now and m["type"] in KNOWN - {"snapshot"}:
                    f.write(json.dumps(m) + "\n")
                    n += 1
        log(f"testnet history: seeded {n} messages of research season {season['season']} up to {now}")

    @staticmethod
    def clock_round(clock) -> int | None:
        """Round from the exchange clock, or None if the clock does not fit the 840 s-per-round pattern."""
        el = clock.get("elapsed_active_seconds")
        if el is None:
            return None
        k = (el - real_t(clock["fictional_time"])) / ROUND_SPAN
        return int(round(k)) + 1 if abs(k - round(k)) < 0.1 else None

    def current_round(self, clock) -> int:
        r = self.clock_round(clock)
        if r is not None:
            return r
        new_day = (self.last_msg_time or "")[:10] != clock["fictional_time"][:10]
        return self.fv.round + (1 if new_day and self.fv.round else 0)

    def ready(self, clock) -> str | None:
        """Why we must not trade right now (None = ok)."""
        rnd = self.current_round(clock)
        if self.fv.round == 0 and rnd != 1:
            return "waiting for the first feed message of the season"
        if rnd > 3:
            return "season finished"
        missing = [r for r in range(1, rnd) if r not in self.seen_rounds]
        if missing:
            return f"no history for earlier round(s) {missing}: fair values would ignore their damage"
        if rnd > max(self.fv.round, 1) and self.fv.storm.final is None and self.fv.round:
            return f"final damage report of round {self.fv.round} was not received"
        s = self.fv.storm
        if rnd == self.fv.round and not s.obs and s.final is None and real_t(clock["fictional_time"]) >= 64:
            return "storm is under way but we have no observations of it (waiting for a snapshot)"
        return None

    def on_news(self, m: dict):
        t = m.get("type")
        if t not in KNOWN:
            log(f"*** NEWS / ANNOUNCEMENT ({t}): {m.get('headline', m)}  -> consider editing overrides.json")
        if m.get("headline"):
            self.seen_headlines.add(m["headline"])
        if t == "snapshot":
            self.backfill(m)
            self.recover_from_snapshot(m)
            return
        if t in ("storm", "advisory", "damage_report", "storm_dissipated", "storm_formed", "impact", "track"):
            with self.state_file.open("a") as f:
                f.write(json.dumps(m) + "\n")
            self.fv.on_message(m)
            self.mark(m)
            if t in ("storm", "advisory", "damage_report", "storm_dissipated", "storm_formed", "impact"):
                self.poke()
        if t not in ("storm", "track"):
            log(f"[{m.get('time')}] r{m.get('round')} {t}: {m.get('headline', '')}")

    def backfill(self, snap: dict):
        """If we joined a storm late, fill the missing early observations from the snapshot track (approximate:
        the track has no intensity, so the snapshot's current intensity is used)."""
        rnd, track, obs = snap.get("round"), snap.get("track"), snap.get("obs")
        if not track or not obs or rnd is None or rnd < self.fv.round:
            return
        have = self.fv.storm.obs if rnd == self.fv.round else []
        first_age = have[0][0] if have else float("inf")
        pts = [p for p in track if p[0] < first_age]
        if not pts or (have and first_age <= 4):
            return
        if rnd > self.fv.round:  # snapshot of a round whose messages we have not seen yet
            self.fv.on_message({"type": "storm", "round": rnd, "storm": snap.get("storm"), "age": pts[0][0],
                                "x": pts[0][1], "y": pts[0][2], "vx": 0, "vy": 0, "radius": pts[0][3],
                                "intensity": obs["intensity"]})
            pts = pts[1:]
        I = obs["intensity"]
        self.fv.storm.obs = [(a, x, y, 0.0, 0.0, R, I) for a, x, y, R in pts] + self.fv.storm.obs
        self.fv.storm.obs.sort(key=lambda o: o[0])
        self.fv._cache = None
        log(f"backfilled {len(pts)} track points of round {rnd} from a snapshot (intensity approximated)")

    def recover_from_snapshot(self, snap: dict):
        """Snapshots (on connect and every 30 s) carry the round's damage reports and headlines: replay any damage
        report we missed (e.g. during a network drop) and print headlines we have not seen (announcements)."""
        rnd = snap.get("round")
        for item in snap.get("damage") or []:
            try:
                if not isinstance(item, dict) or "damage_pct" not in item:
                    continue
                msg = {**item, "type": "damage_report", "round": item.get("round", rnd),
                       "storm": item.get("storm", snap.get("storm")), "stage": item.get("stage", "final")}
                if msg["round"] != self.fv.round:
                    continue
                st = self.fv.storm
                missing = (msg["stage"] == "final" and st.final is None and "industry_loss_musd" in msg) or \
                          (msg["stage"] == "preliminary" and st.prelim is None and st.final is None)
                if missing:
                    log(f"recovered a missed {msg['stage']} damage report of round {msg['round']} from a snapshot")
                    self.on_news(msg)
            except Exception:  # never let a malformed snapshot stop the bot
                log("snapshot damage item not understood:", str(item)[:200])
        for h in snap.get("headlines") or []:
            text = h.get("headline") or h.get("text") or json.dumps(h) if isinstance(h, dict) else str(h)
            if text and text not in self.seen_headlines:
                self.seen_headlines.add(text)
                log(f"*** HEADLINE: {text}  -> if it changes a company's economics, adjust it on the dashboard")

    # ------------------------------------------------------------ trading
    def poke(self):
        if self.wake is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(self.wake.set)

    def load_overrides(self):
        try:
            self.overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
        except json.JSONDecodeError as e:
            log("overrides.json is not valid JSON, ignoring:", e)
        self.fv.val.overrides = {k: v for k, v in self.overrides.get("symbols", {}).items()
                                 if "mult" in v or "add" in v or "params" in v}
        self.fv._cache = None

    def parse_positions(self, resp):
        rows = resp.get("positions", resp) if isinstance(resp, dict) else resp
        pos = {s: 0 for s in SYMBOLS}
        for r in rows or []:
            sym = r.get("symbol")
            q = next((r[k] for k in ("quantity", "position", "qty", "net_position") if k in r), 0)
            if sym in pos:
                pos[sym] = int(q)
                self.pos_rows[sym] = r
        present = {r.get("symbol") for r in rows or []}
        for sym in list(self.pos_rows):
            if sym not in present:
                del self.pos_rows[sym]
        return pos

    def plan(self, sym, V, px, sd, book, scale):
        """(side, qty, worst price, dollar edge) walking the book; the same rule as backtest/sim.py."""
        side = 1 if V > px else -1
        L = self.limits[sym]
        room = int(L * scale) - side * self.positions[sym]
        if room <= 0:
            return None
        unit = self.size_pen + self.risk * sd / px
        run, qty, worst, value = side * self.positions[sym], 0, None, 0.0
        for lv in book.get("asks" if side == 1 else "bids") or []:
            lp, avail = float(lv["price"]), int(lv["quantity"])
            edge = side * (V - lp)
            if edge <= self.margin * px:
                break
            cap = L if unit <= 0 else min(L, (edge / px - self.margin) / unit * L)
            take = int(max(0, min(avail, cap - run, room - qty)))
            if take <= 0:
                break
            qty, run, worst, value = qty + take, run + take, lp, value + take * edge
            if qty >= room:
                break
        return (side, qty, worst, value) if qty else None

    def candidates(self, mean, sd, rnd, t):
        """Symbols whose top of book (from the quote stream) is already through V -/+ margin."""
        out = []
        for sym in SYMBOLS:
            if sym in self.skip or self.overrides.get("symbols", {}).get(sym, {}).get("disable"):
                continue
            q = self.quotes.get(sym)
            beta = self.beta.get(phase(rnd, t), {}).get(sym, 0.0)
            if not q or q.get("bid") is None or q.get("ask") is None:
                out.append((sym, None, beta))  # no quote yet: let the book decide
                continue
            px = (q["bid"] + q["ask"]) / 2
            V = px + beta * (mean[sym] - px)
            best = q["ask"] if V > px else q["bid"]
            if (1 if V > px else -1) * (V - best) > self.margin * px:
                out.append((sym, px, beta))
        return out

    def step(self):
        self._trading = False
        try:
            self._step()
        finally:
            self.write_status()

    def _step(self):
        t_start = time.time()
        f_clock = self.pool.submit(self.ex.time)
        f_pos = self.pool.submit(self.ex.positions)
        clock = f_clock.result()
        self.clock = clock
        self.set_competition(clock["competition_id"])
        if clock.get("state") != "running":
            return
        reason = self.ready(clock)
        if reason:
            if reason != self._last_reason:
                log("not trading:", reason)
            self._last_reason = reason
            self.positions = self.parse_positions(f_pos.result())
            return
        if self._last_reason:
            log("trading enabled")
            self._last_reason = None
        self.load_overrides()
        if self.overrides.get("halt"):
            if not self.args.dry_run and self.budget.take(1):
                self.ex.cancel_all()
            log("halted by overrides.json")
            return
        t = real_t(clock["fictional_time"])
        rnd = self.current_round(clock)
        scale = self.args.scale * float(self.overrides.get("scale", 1.0))
        mean, sd = self.fv.value()
        self._trading = True
        cands = self.candidates(mean, sd, rnd, t)
        self.positions = self.parse_positions(f_pos.result())
        if not cands:
            return
        books = dict(zip([c[0] for c in cands], self.pool.map(lambda c: self.ex.orderbook(c[0]), cands)))
        plans = []
        for sym, _, beta in cands:
            b = books[sym]
            if not (b.get("bids") and b.get("asks")):
                continue
            px = (float(b["bids"][0]["price"]) + float(b["asks"][0]["price"])) / 2
            V = px + beta * (mean[sym] - px)
            p = self.plan(sym, V, px, sd[sym], b, scale)
            if p:
                plans.append((p[3], sym, p, px, V, beta))
        plans.sort(reverse=True)
        # pace the budget: at most per_min/30 orders per cycle (one cycle ~ 2 s), keep one action for the cancel
        per_cycle = self.args.max_orders or max(1, self.args.actions_per_min // 30)
        n_ok = min(len(plans), per_cycle, self.budget.available() - 1)
        if n_ok < len(plans):
            log(f"action budget: sending {max(n_ok, 0)} of {len(plans)} orders")
        plans = plans[: max(n_ok, 0)]
        for value, sym, (side, qty, price, _), px, V, beta in plans:
            log(f"{'buy' if side == 1 else 'sell':4s} {qty:4d} {sym:5s} @ {price:.2f}  edge ${value:,.0f}  "
                f"(mid {px:.2f}, fair {mean[sym]:.2f}, V {V:.2f}, beta {beta:.2f}, pos {self.positions[sym]})")
        if self.args.dry_run or not plans:
            return

        def send(item):
            _, sym, (side, qty, price, _), *_ = item
            if not self.budget.take(1):
                return None
            try:
                return self.ex.limit(sym, "buy" if side == 1 else "sell", qty, price)
            except ApiError as e:
                log(f"  {sym} rejected: {e}")
                return None

        if not self.trade_lock.acquire():  # the ETF-arb bot is mid-trade: try again next cycle
            log("trade lock busy (ETF arb in progress); skipping this cycle's orders")
            return
        try:
            sent = [r for r in self.pool.map(send, plans) if r]
            if sent:  # one cancel-all, only if something rests (inside the lock: never cancels the arb's orders)
                resting = (self.ex.open_orders() or {}).get("orders", [])
                if resting and self.budget.take(1):
                    self.ex.cancel_all()
                    log(f"  cancelled {len(resting)} resting remainder(s)")
        finally:
            self.trade_lock.release()
        now = time.time()
        for value, sym, (side, qty, price, _), px, V, beta in plans[: len(sent)]:
            self.recent_orders.appendleft({"ts": now, "clock": clock["fictional_time"], "sym": sym,
                                           "side": "buy" if side == 1 else "sell", "qty": qty, "price": price,
                                           "edge": value, "fair": mean[sym], "mid": px})
        self.cycle_ms = 1000 * (time.time() - t_start)
        if self.args.verbose:
            log(f"cycle r{rnd} t={t:.0f}s: {len(sent)} orders in {1000 * (time.time() - t_start):.0f} ms, "
                f"budget left {self.budget.available()}")

    # ------------------------------------------------------------ dashboard
    def write_status(self, force: bool = False):
        """Snapshot for bot/dashboard.py: bot/state/status.json (atomic) + one line of PnL history."""
        now = time.time()
        if not force and now - self._last_status < STATUS_EVERY:
            return
        self._last_status = now
        try:
            mean, sd = self.fv.value()
            clock = self.clock or {}
            rnd = self.current_round(clock) if clock.get("fictional_time") else self.fv.round
            t = real_t(clock["fictional_time"]) if clock.get("fictional_time") else None
            syms, exp_tot, mtm_tot, gross = {}, 0.0, 0.0, 0.0
            for sym in SYMBOLS:
                q = self.quotes.get(sym) or {}
                bid, ask = q.get("bid"), q.get("ask")
                mid = (bid + ask) / 2 if bid is not None and ask is not None else None
                row = self.pos_rows.get(sym, {})
                pos = self.positions.get(sym, 0)
                avg = row.get("average_price")
                realized = float(row.get("realized_pnl") or 0.0)
                unreal = float(row.get("unrealized_pnl") or 0.0)
                exp = realized + (pos * (mean[sym] - avg) if pos and avg is not None else 0.0)
                beta = self.beta.get(phase(rnd, t), {}).get(sym) if t is not None and rnd else None
                syms[sym] = {"pos": pos, "limit": self.limits[sym], "avg": avg, "bid": bid, "ask": ask, "mid": mid,
                             "fair": mean[sym], "sd": sd[sym], "beta": beta,
                             "value": (mid + beta * (mean[sym] - mid)) if (mid is not None and beta is not None) else None,
                             "exp_pnl": exp, "mtm_pnl": realized + unreal}
                exp_tot += exp
                mtm_tot += realized + unreal
                gross += abs(pos) * (mid if mid is not None else mean[sym])
            st = self.fv.storm
            obs = st.obs[-1] if st.obs else None
            official = [k for src, k in st.adv if src == "official"]
            from engine.model import OUTLOOKS
            post = self.fv.pattern_posterior() if st.obs else None
            status = {
                "ts": now, "pid": os.getpid(), "version": "v2", "profile": self.args.profile, "dry_run": self.args.dry_run,
                "scale": self.args.scale * float(self.overrides.get("scale", 1.0)) if self.overrides else self.args.scale,
                "competition_id": self.comp_id, "state": clock.get("state"), "fictional_time": clock.get("fictional_time"),
                "round": rnd, "t": t, "phase": phase(rnd, t) if (t is not None and rnd) else None,
                "trading": self._trading, "reason": self._last_reason, "net_down": self._net_down,
                "halted": bool(self.overrides.get("halt")),
                "budget": {"limit": self.args.actions_per_min, "left": self.budget.available()},
                "cycle_ms": self.cycle_ms, "seen_rounds": sorted(r for r in self.seen_rounds if r),
                "totals": {"exp_pnl": exp_tot, "mtm_pnl": mtm_tot, "gross": gross},
                "symbols": syms,
                "storm": {"name": st.name, "round": self.fv.round, "dissipated": st.dissipated,
                          "obs": obs and {"age": obs[0], "x": obs[1], "y": obs[2], "radius": obs[5], "intensity": obs[6]},
                          "track": [(o[0], o[1], o[2], o[5]) for o in st.obs[::5]] + ([(obs[0], obs[1], obs[2], obs[5])] if obs else []),
                          "cone": st.cone, "outlook": OUTLOOKS[official[-1]] if official else None,
                          "posterior": dict(zip(OUTLOOKS, map(float, post))) if post is not None else None,
                          "prelim_loss": float(self.fv.val.loss(st.prelim)) if st.prelim is not None else None,
                          "final_loss": st.final_loss},
                "season_loss_done": float(self.fv.val.loss(self.fv.D_done)),
                "orders": list(self.recent_orders)[:40],
            }
            tmp = STATE / "status.json.tmp"
            tmp.write_text(json.dumps(status, default=float))
            os.replace(tmp, STATE / "status.json")
            if self.comp_id and clock.get("state") == "running":
                with (STATE / f"history-{self.comp_id}.jsonl").open("a") as f:
                    f.write(json.dumps({"ts": now, "round": rnd, "t": t, "exp": round(exp_tot, 2),
                                        "mtm": round(mtm_tot, 2), "gross": round(gross, 0)}) + "\n")
        except Exception:  # the dashboard must never break trading
            log("status write failed:\n" + traceback.format_exc())

    # ------------------------------------------------------------ loops
    async def feed(self):
        async for frame in self.ex.stream(["news", "quotes"], SYMBOLS):
            if frame.get("type") == "quote":
                self.quotes[frame["symbol"]] = frame
            elif frame.get("type") == "news" and isinstance(frame.get("message"), dict):
                if self.state_file is not None:
                    self.on_news(frame["message"])

    async def trader(self):
        last = 0.0
        while True:
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=self.args.step)
            except asyncio.TimeoutError:
                pass
            self.wake.clear()
            gap = time.time() - last
            if gap < self.args.min_interval:
                await asyncio.sleep(self.args.min_interval - gap)
            last = time.time()
            try:
                await asyncio.to_thread(self.step)
                if self._net_down:
                    log("network back")
                    self._net_down = False
            except OSError as e:  # network down (URLError is an OSError): log once, keep retrying
                if not self._net_down:
                    log(f"network error, retrying: {e}")
                self._net_down = True
                self.write_status(force=True)
            except Exception:  # keep trading through other transient errors
                log("step failed:\n" + traceback.format_exc())

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.wake = asyncio.Event()
        self.set_competition(self.ex.time()["competition_id"])
        await asyncio.gather(self.feed(), self.trader())


class ActionBudget:
    """Rolling 60 s count of order actions (place, cancel, cancel-all, create, redeem)."""

    def __init__(self, per_min: int):
        self.per_min, self.stamps, self.lock = per_min, deque(), threading.Lock()

    def _trim(self):
        now = time.monotonic()
        while self.stamps and now - self.stamps[0] > 60:
            self.stamps.popleft()

    def available(self) -> int:
        with self.lock:
            self._trim()
            return self.per_min - len(self.stamps)

    def take(self, n: int) -> bool:
        with self.lock:
            self._trim()
            if len(self.stamps) + n > self.per_min:
                return False
            self.stamps.extend([time.monotonic()] * n)
            return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="https://cornelltradingcompetition.org")
    ap.add_argument("--tag", default="train", help="engine parameters: train (as backtested) or all")
    ap.add_argument("--profile", default="tuned", choices=["tuned", "lower_variance"])
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the position limits to use")
    ap.add_argument("--step", type=float, default=2.0, help="max seconds between cycles without news")
    ap.add_argument("--min-interval", type=float, default=0.5, help="min seconds between cycles")
    ap.add_argument("--actions-per-min", type=int, default=170, help="order-action budget (team limit is 180)")
    ap.add_argument("--skip-symbols", default="", help="comma list not to trade (e.g. ARCH,REBLD when the ETF-arb bot runs)")
    ap.add_argument("--max-orders", type=int, default=0, help="orders per cycle (0 = actions-per-min / 30)")
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--testnet-history", action="store_true",
                    help="TESTNET ONLY: rebuild missed messages of this season from the research files")
    args = ap.parse_args()
    if not OVERRIDES.exists():
        OVERRIDES.write_text(json.dumps({"halt": False, "scale": 1.0, "symbols": {}}, indent=2))
    try:
        asyncio.run(LiveBot(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
