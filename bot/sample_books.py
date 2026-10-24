"""Record order books of all symbols every few seconds while the market runs (for the backtest cost model).

    ../.venv/bin/python sample_books.py --minutes 30 --out logs/books.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

from client import ApiError, Exchange


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--every", type=float, default=5)
    ap.add_argument("--out", default="logs/books.jsonl")
    args = ap.parse_args()
    ex = Exchange()
    syms = list(ex.case()["limits"])
    end = time.time() + 60 * args.minutes
    with open(args.out, "a") as f:
        while time.time() < end:
            try:
                clock = ex.time()
                if clock.get("state") == "running":
                    for s in syms:
                        f.write(json.dumps({"t": time.time(), "clock": clock["fictional_time"], **ex.orderbook(s)}) + "\n")
                    f.flush()
            except (ApiError, OSError) as e:
                print("error", e, flush=True)
            time.sleep(args.every)


if __name__ == "__main__":
    main()
