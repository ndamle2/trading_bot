"""Replay research seasons through the fair-value engine and save the fair-value time series.

    ../.venv/bin/python fv_series.py --seasons 1-200 --tag train      # holdout, params fitted on 201-600
    ../.venv/bin/python fv_series.py --seasons 201-600 --tag train    # in-sample (own season left out of library)

The engine sees only feed messages already published at each step (strictly by message time).
Output: backtest/out/fv_<tag>_<first>-<last>.pkl, one row per (season, round, t) with fv_<SYM> and sd_<SYM>.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.model import SYMBOLS, FairValue  # noqa: E402

HISTORY = ROOT / "hurricane-season-research" / "history"
OUT = Path(__file__).resolve().parent / "out"
STEP = 10  # real seconds between fair-value evaluations
ROUND_LEN = 780


def real_t(ts: str) -> float:
    """Real seconds since 09:30 of the message's market day."""
    t = datetime.fromisoformat(ts)
    return ((t.hour - 9) * 3600 + (t.minute - 30) * 60 + t.second) / 30


def run_season(args) -> list[dict]:
    season, tag, n, k, cone_sigma, cone_k, step = args
    fv = FairValue.load(tag, n_samples=n, k_analogs=k, seed=season, exclude_season=season,
                        cone_sigma=cone_sigma, cone_k=cone_k)
    feed = [json.loads(line) for line in (HISTORY / f"season-{season:03d}" / "feed.jsonl").open()]
    rows = []
    for rnd in (1, 2, 3):
        msgs = [m for m in feed if m["round"] == rnd]
        j = 0
        for t in range(0, ROUND_LEN + 1, step):
            while j < len(msgs) and real_t(msgs[j]["time"]) <= t:
                fv.on_message(msgs[j])
                j += 1
            if fv.round != rnd:  # before the first message of the round
                fv.round, fv.storm = rnd, type(fv.storm)()
            mean, sd = fv.value()
            rows.append({"season": season, "round": rnd, "t": t,
                         **{f"fv_{s}": mean[s] for s in SYMBOLS}, **{f"sd_{s}": sd[s] for s in SYMBOLS}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="1-200")
    ap.add_argument("--tag", default="train")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--k", type=int, default=100, help="analog storms")
    ap.add_argument("--out", default="")
    ap.add_argument("--cone", type=float, default=None, help="cone sd as a multiple of the spread (off if unset)")
    ap.add_argument("--cone-k", type=int, default=None)
    ap.add_argument("--step", type=int, default=STEP, help="real seconds between evaluations")
    a = ap.parse_args()
    lo, hi = map(int, a.seasons.split("-"))
    jobs = [(s, a.tag, a.samples, a.k, a.cone, a.cone_k, a.step) for s in range(lo, hi + 1)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(run_season, jobs, chunksize=2)):
            rows += r
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} seasons", flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / (a.out or f"fv_{a.tag}_{lo}-{hi}.pkl")
    path.write_bytes(pickle.dumps(pd.DataFrame(rows)))
    print("wrote", path)


if __name__ == "__main__":
    main()
