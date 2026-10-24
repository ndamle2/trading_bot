"""Connectivity bot: checks every part of the exchange interface, then stays connected and logs the feed.

    ../.venv/bin/python bot.py                 # full check, then follow the feed
    ../.venv/bin/python bot.py --no-orders     # read-only check
    ../.venv/bin/python bot.py --check-only    # exit after the check

Checks
  1. REST: case data, clock/state, positions, open orders, one order book
  2. Orders (only while the market is running): a 1-share limit buy far below the bid, confirm it rests in
     /orders/open, cancel it, confirm it is gone. Uses 2 order actions; it cannot fill.
  3. WebSocket: subscribe to news + quotes for all symbols and count message types for --listen seconds.
Then it keeps the latest quote per symbol and prints every storm-feed headline, reconnecting on drops.
Raw frames are appended to bot/logs/feed-<date>.jsonl so they can be checked with check_recording.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from client import ApiError, Exchange, order_id

LOGS = Path(__file__).resolve().parent / "logs"


def show(label: str, obj) -> None:
    text = json.dumps(obj, default=str)
    print(f"  {label}: {text[:400]}{'...' if len(text) > 400 else ''}")


def rest_checks(ex: Exchange, symbols: list[str]) -> dict:
    print("== REST")
    clock = ex.time()
    show("time", clock)
    show("positions", ex.positions())
    show("open orders", ex.open_orders())
    show(f"orderbook {symbols[0]}", ex.orderbook(symbols[0]))
    return clock


def best_bid(book) -> float | None:
    """Best bid from an order book response; tolerant of a few plausible shapes."""
    if not isinstance(book, dict):
        return None
    bids = book.get("bids") or book.get("bid") or []
    if isinstance(bids, (int, float)):
        return float(bids)
    prices = []
    for lvl in bids:
        if isinstance(lvl, dict):
            prices.append(float(lvl.get("price", 0)))
        elif isinstance(lvl, (list, tuple)) and lvl:
            prices.append(float(lvl[0]))
    return max(prices) if prices else None


def order_check(ex: Exchange, symbol: str) -> bool:
    print(f"== Orders ({symbol})")
    bid = best_bid(ex.orderbook(symbol))
    price = round(max(0.01, (bid or 2.0) * 0.5), 2)
    try:
        resp = ex.limit(symbol, "buy", 1, price)
    except ApiError as e:
        print(f"  place rejected: {e}")
        return False
    show("placed", resp)
    oid = order_id(resp)
    if oid is None:
        print("  could not find the order id in the response; cancelling all orders in the symbol instead")
        show("cancel all", ex.cancel_all(symbol))
        return False
    time.sleep(0.5)
    resting = json.dumps(ex.open_orders(), default=str)
    print(f"  order {oid} resting in /orders/open: {str(oid) in resting}")
    show("cancel", ex.cancel(oid))
    time.sleep(0.5)
    gone = str(oid) not in json.dumps(ex.open_orders(), default=str)
    print(f"  order {oid} gone after cancel: {gone}")
    return gone


async def follow(ex: Exchange, symbols: list[str], listen: float, forever: bool) -> None:
    print(f"== WebSocket (news + quotes, {len(symbols)} symbols)")
    LOGS.mkdir(exist_ok=True)
    counts: Counter = Counter()
    quotes: dict[str, dict] = {}
    start = time.monotonic()
    reported = False
    with (LOGS / f"feed-{time.strftime('%Y%m%d')}.jsonl").open("a") as log:
        async for frame in ex.stream(["news", "quotes"], symbols):
            log.write(json.dumps({**frame, "received_at": time.time()}) + "\n")
            kind = frame.get("type")
            msg = frame.get("message") if isinstance(frame.get("message"), dict) else None
            counts[f"{kind}:{msg['type']}" if msg else kind] += 1
            if kind == "quote":
                quotes[frame.get("symbol")] = frame
            if counts[kind] == 1:
                show(f"first {kind} frame", frame)
            if msg and msg.get("type") not in ("storm", "track"):
                print(f"  [{msg.get('time')}] r{msg.get('round')} {msg.get('type')}: {msg.get('headline', '')}")
            if not reported and time.monotonic() - start > listen:
                reported = True
                print(f"  after {listen:.0f}s: {dict(counts)}")
                print("  quotes seen for:", sorted(quotes))
                if not forever:
                    return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="https://cornelltradingcompetition.org")
    ap.add_argument("--symbol", default="FERRY", help="symbol for the order check")
    ap.add_argument("--listen", type=float, default=20, help="seconds of feed to summarise")
    ap.add_argument("--no-orders", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    ex = Exchange(args.url)
    case = ex.case()
    symbols = list(case["limits"])
    print(f"case: {case['name']}, symbols: {symbols}")
    clock = rest_checks(ex, symbols)
    running = str(clock.get("state", "")).lower() == "running"
    if args.no_orders:
        print("== Orders: skipped (--no-orders)")
    elif not running:
        print(f"== Orders: skipped, market state is {clock.get('state')!r} (re-run while it is running)")
    else:
        order_check(ex, args.symbol)
    try:
        asyncio.run(follow(ex, symbols, args.listen, forever=not args.check_only))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
