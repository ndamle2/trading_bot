"""FALLBACK (v1) live bot, kept as it was before the v2 rewrite. Use it if live_bot.py misbehaves:
    ./run_bot.sh --v1            (from trading_comp/)
It polls every --step s (default 5) instead of trading on each message, and has no action budget or risk term.
Parameters are pinned to bot/v1_strategy_params.json. It shares bot/state/ with v2, so you can switch mid-season.

Live trading bot: the backtested fair-value strategy on the exchange.

    ../.venv/bin/python live_bot.py --dry-run          # compute and print orders, send nothing
    ../.venv/bin/python live_bot.py --scale 0.2        # trade at 20% of the position limits
    ../.venv/bin/python live_bot.py                    # full size

Every --step seconds while the market runs, for each symbol:
  px   = mid of the house quote
  V    = px + beta[sym, round, phase] * (engine fair value - px)        (beta from backtest/out/strategy_params.json)
  walk the order book and take every level priced better than V -/+ (margin + size_pen * |position|/limit) * px,
  as ONE limit order at the worst level taken; cancel whatever did not fill. Positions are held otherwise.

State: every news message is appended to bot/state/<competition_id>.jsonl and replayed on restart, so the bot
can be stopped and started within a season. Starting mid-season without that file loses the damage of
earlier rounds (the bot warns).

Overrides (bot/overrides.json, re-read every step; edit it while the bot runs to react to announcements):
  {"halt": false,                                   # true: cancel all orders, trade nothing
   "scale": 1.0,                                    # multiplies --scale
   "symbols": {"HAVEN": {"mult": 1.0, "add": 0.0,   # fair value -> value * mult + add (ETFs follow)
                         "disable": false}}}         # stop trading one symbol
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from client import ApiError, Exchange, order_id  # noqa: E402
from engine.model import SYMBOLS, FairValue  # noqa: E402

STATE = HERE / "state"
OVERRIDES = HERE / "overrides.json"
PARAMS = HERE / "v1_strategy_params.json"  # pinned copy, not overwritten by backtests
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
        self.ex = Exchange(args.url)
        self.case = self.ex.case()
        self.limits = self.case["limits"]
        sp = json.loads(PARAMS.read_text())
        self.margin, self.size_pen = sp["margin"], sp["size_pen"]
        self.signal = {tuple(k.split("|")): v for k, v in sp["signal"].items()}
        self.fv = FairValue.load(args.tag, n_samples=args.samples)
        self.quotes: dict[str, dict] = {}
        self.positions: dict[str, int] = {s: 0 for s in SYMBOLS}
        self.comp_id = None
        self.state_file = None
        self.last_msg_time = None
        self.overrides = {}
        self.seen_rounds: set = set()
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

    def current_round(self, clock) -> int:
        """The engine learns the round from the first message of the day; before that, it is the next round."""
        new_day = (self.last_msg_time or "")[:10] != clock["fictional_time"][:10]
        return self.fv.round + (1 if new_day and self.fv.round else 0)

    def ready(self, clock) -> str | None:
        """Why we must not trade right now (None = ok)."""
        if self.fv.round == 0:
            return "waiting for the first feed message of the season"
        missing = [r for r in range(1, self.fv.round) if r not in self.seen_rounds]
        if missing:
            return f"no history for earlier round(s) {missing}: fair values would ignore their damage"
        rnd = self.current_round(clock)
        if rnd > 3:
            return "season finished"
        if rnd > self.fv.round and self.fv.storm.final is None:
            return f"final damage report of round {self.fv.round} was not received"
        s = self.fv.storm
        if rnd == self.fv.round and not s.obs and s.final is None and real_t(clock["fictional_time"]) >= 64:
            return "storm is under way but we have no observations of it (waiting for a snapshot)"
        return None

    def on_news(self, m: dict):
        t = m.get("type")
        if t not in KNOWN:
            log(f"*** NEWS / ANNOUNCEMENT ({t}): {m.get('headline', m)}  -> consider editing overrides.json")
        if t == "snapshot":
            self.backfill(m)
            return
        if t in ("storm", "advisory", "damage_report", "storm_dissipated", "storm_formed", "impact", "track"):
            with self.state_file.open("a") as f:
                f.write(json.dumps(m) + "\n")
            self.fv.on_message(m)
            self.mark(m)
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

    # ------------------------------------------------------------ trading
    def load_overrides(self):
        try:
            self.overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
        except json.JSONDecodeError as e:
            log("overrides.json is not valid JSON, ignoring:", e)
        self.fv.val.overrides = {k: v for k, v in self.overrides.get("symbols", {}).items()
                                 if "mult" in v or "add" in v}
        self.fv._cache = None

    def refresh_positions(self):
        resp = self.ex.positions()
        rows = resp.get("positions", resp) if isinstance(resp, dict) else resp
        pos = {s: 0 for s in SYMBOLS}
        for r in rows or []:
            sym = r.get("symbol")
            q = next((r[k] for k in ("quantity", "position", "qty", "net_position") if k in r), 0)
            if sym in pos:
                pos[sym] = int(q)
        self.positions = pos

    def plan(self, sym, V, px, book, scale):
        """(side, qty, limit price) walking the book, mirroring backtest.simulate."""
        side = 1 if V > px else -1
        lim = int(self.limits[sym] * scale)
        pos = self.positions[sym]
        room = lim - side * pos
        if room <= 0:
            return None
        levels = book.get("asks" if side == 1 else "bids") or []
        chunk = max(1, lim // 20)
        done, worst = 0, None
        for lv in levels:
            lp, avail = float(lv["price"]), int(lv["quantity"])
            while avail > 0 and done < room:
                q = min(chunk, room - done, avail)
                new_pos = pos + side * (done + q)
                need = (self.margin + self.size_pen * max(side * new_pos, 0) / self.limits[sym]) * px
                if side * (V - lp) <= need:
                    return (side, done, worst) if done else None
                done += q
                avail -= q
                worst = lp
            if done >= room:
                break
        return (side, done, worst) if done else None

    def trade_symbol(self, sym, mean, rnd, t, scale):
        q = self.quotes.get(sym)
        if not q or q.get("bid") is None or q.get("ask") is None:
            return
        if self.overrides.get("symbols", {}).get(sym, {}).get("disable"):
            return
        px = (q["bid"] + q["ask"]) / 2
        beta = self.signal.get((sym, str(phase(rnd, t))), (0.0, 0.0))[0]
        V = px + beta * (mean[sym] - px)
        best = q["ask"] if V > px else q["bid"]
        if (1 if V > px else -1) * (V - best) <= self.margin * px:  # cheap pre-check on the top of book
            return
        book = self.ex.orderbook(sym)
        p = self.plan(sym, V, px, book, scale)
        if not p:
            return
        side, qty, price = p
        name = "buy" if side == 1 else "sell"
        log(f"{name:4s} {qty:4d} {sym:5s} @ {price:.2f}  (mid {px:.2f}, fair {mean[sym]:.2f}, V {V:.2f}, "
            f"beta {beta:.2f}, pos {self.positions[sym]})")
        if self.args.dry_run:
            return
        try:
            resp = self.ex.limit(sym, name, qty, price)
        except ApiError as e:
            log(f"  rejected: {e}")
            return
        oid = order_id(resp)
        if oid is not None:  # take-only: cancel whatever rests
            try:
                self.ex.cancel(oid)
            except ApiError as e:
                if e.code not in (400, 404, 409):
                    log(f"  cancel failed: {e}")

    def step(self):
        clock = self.ex.time()
        self.set_competition(clock["competition_id"])
        if clock.get("state") != "running":
            return
        reason = self.ready(clock)
        if reason:
            if reason != self._last_reason:
                log("not trading:", reason)
            self._last_reason = reason
            return
        if self._last_reason:
            log("trading enabled")
            self._last_reason = None
        self.load_overrides()
        if self.overrides.get("halt"):
            if not self.args.dry_run:
                self.ex.cancel_all()
            log("halted by overrides.json")
            return
        t = real_t(clock["fictional_time"])
        rnd = self.current_round(clock)
        scale = self.args.scale * float(self.overrides.get("scale", 1.0))
        self.refresh_positions()
        t0 = time.time()
        mean, sd = self.fv.value()
        if self.args.verbose:
            log(f"r{rnd} t={t:.0f}s fair values computed in {1000 * (time.time() - t0):.0f} ms: " +
                " ".join(f"{s}={mean[s]:.2f}" for s in SYMBOLS))
        for sym in SYMBOLS:
            try:
                self.trade_symbol(sym, mean, rnd, t, scale)
            except ApiError as e:
                log(f"{sym}: {e}")

    # ------------------------------------------------------------ loops
    async def feed(self):
        async for frame in self.ex.stream(["news", "quotes"], SYMBOLS):
            if frame.get("type") == "quote":
                self.quotes[frame["symbol"]] = frame
            elif frame.get("type") == "news" and isinstance(frame.get("message"), dict):
                if self.state_file is not None:
                    self.on_news(frame["message"])

    async def trader(self):
        while True:
            try:
                await asyncio.to_thread(self.step)
                if self._net_down:
                    log("network back")
                    self._net_down = False
            except OSError as e:  # network down (URLError is an OSError): log once, keep retrying
                if not self._net_down:
                    log(f"network error, retrying every {self.args.step:.0f}s: {e}")
                self._net_down = True
            except Exception:  # keep trading through other transient errors
                log("step failed:\n" + traceback.format_exc())
            await asyncio.sleep(self.args.step)

    async def run(self):
        self.set_competition(self.ex.time()["competition_id"])
        await asyncio.gather(self.feed(), self.trader())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="https://cornelltradingcompetition.org")
    ap.add_argument("--tag", default="train", help="engine parameters: train (as backtested) or all")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the position limits to use")
    ap.add_argument("--step", type=float, default=5.0, help="seconds between trading steps")
    ap.add_argument("--samples", type=int, default=2000)
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
