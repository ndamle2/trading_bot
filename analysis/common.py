"""Shared loaders for the research data, as pandas tables.

    from common import case, storms, seasons, obs, advisories, prices, reports

Every table is built once and cached in analysis/cache/ (delete the folder to rebuild).

Tables
------
seasons()     one row per season: season loss, cumulative damage per company, settlement per symbol
storms()      one row per storm (season, round): pattern, erratic flip, damage_pct per company, loss
obs()         one row per storm observation (every 2 s): season, round, age, x, y, vx, vy, radius, intensity
advisories()  one row per advisory (official + wire): outlook, hurricane_conditions per island, true pattern
reports()     one row per damage report: stage, damage_pct per company, insured loss
impacts()     one row per impact message: island, age
prices()      the house consensus price every 2 s, with season and a round-relative real-time column
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "hurricane-season-research"
HISTORY = DATA / "history"
CACHE = Path(__file__).resolve().parent / "cache"

PHYSICAL = ["PORT", "ISLE", "AQUA", "GRID", "FERRY"]
REBUILD = ["RECON", "MAT", "GEN"]
INSURERS = ["HAVEN", "REIN"]
STOCKS = PHYSICAL + REBUILD + INSURERS
ETFS = ["ARCH", "REBLD"]
SYMBOLS = STOCKS + ETFS
ISLANDS = ["cascadia_prime", "coral_key", "azure_isle", "palm_atoll"]
OUTLOOKS = ["direct", "chain", "recurve", "stall", "eyewall", "erratic"]


@lru_cache
def case() -> dict:
    return json.loads((DATA / "case.json").read_text())


@lru_cache
def manifest() -> dict:
    return json.loads((HISTORY / "manifest.json").read_text())


def sites() -> pd.DataFrame:
    """One row per company site: symbol, name, island, x, y, weight."""
    return pd.DataFrame([{"symbol": sym, **s} for sym, c in case()["companies"].items() for s in c["sites"]])


def _cached(name: str):
    """Cache the DataFrame that the decorated builder returns in analysis/cache/<name>.pkl."""
    def wrap(build):
        @lru_cache
        def get() -> pd.DataFrame:
            path = CACHE / f"{name}.pkl"
            if path.exists():
                return pickle.loads(path.read_bytes())
            df = build()
            CACHE.mkdir(exist_ok=True)
            path.write_bytes(pickle.dumps(df))
            return df
        get.__doc__ = build.__doc__
        return get
    return wrap


def _feed(season_dir: str):
    with (HISTORY / season_dir / "feed.jsonl").open() as f:
        for line in f:
            yield json.loads(line)


@_cached("storms")
def storms() -> pd.DataFrame:
    """One row per storm, with damage_pct as dmg_<SYM> and the running season loss."""
    rows = []
    for s in manifest()["seasons"]:
        running = 0.0
        for st in s["storms"]:
            running += st["industry_loss_musd"]
            rows.append({
                "season": s["season"], "round": st["round"], "storm": st["storm"], "pattern": st["pattern"],
                "erratic_at": st["turned_erratic_at"], "open": st["open"], "loss": st["industry_loss_musd"],
                "cum_loss": running, **{f"dmg_{k}": v for k, v in st["damage_pct"].items()},
            })
    return pd.DataFrame(rows)


@_cached("seasons")
def seasons() -> pd.DataFrame:
    """One row per season: loss, cumulative damage (sum and compounded), settlement of each symbol."""
    st = storms()
    rows = []
    for s in manifest()["seasons"]:
        g = st[st.season == s["season"]]
        r = {"season": s["season"], "loss": s["season_loss_musd"]}
        for c in PHYSICAL:
            r[f"dmg_{c}"] = g[f"dmg_{c}"].sum()
            # damage_pct may be of the remaining value; keep a compounded version to compare
            r[f"cdmg_{c}"] = 100 * (1 - np.prod(1 - g[f"dmg_{c}"] / 100))
        settle = json.loads((HISTORY / s["dir"] / "settlement.json").read_text())
        r.update({f"set_{k}": v for k, v in settle.items()})
        rows.append(r)
    return pd.DataFrame(rows)


@_cached("obs")
def obs() -> pd.DataFrame:
    """Every storm observation (2 s cadence)."""
    rows = []
    for s in manifest()["seasons"]:
        for m in _feed(s["dir"]):
            if m["type"] == "storm":
                rows.append((s["season"], m["round"], m["age"], m["x"], m["y"], m["vx"], m["vy"],
                             m["radius"], m["intensity"]))
    return pd.DataFrame(rows, columns=["season", "round", "age", "x", "y", "vx", "vy", "radius", "intensity"])


@_cached("advisories")
def advisories() -> pd.DataFrame:
    """Official advisories and wire rumors, with the true pattern at that age (erratic after a flip)."""
    st = storms().set_index(["season", "round"])
    rows = []
    for s in manifest()["seasons"]:
        for m in _feed(s["dir"]):
            if m["type"] != "advisory":
                continue
            truth = st.loc[(s["season"], m["round"])]
            flip = truth["erratic_at"]
            now = "erratic" if pd.notna(flip) and m["age"] >= flip else truth["pattern"]
            r = {"season": s["season"], "round": m["round"], "source": m["source"], "age": m["age"],
                 "number": m.get("number"), "outlook": m["outlook"], "pattern": truth["pattern"],
                 "true_now": now, "erratic_at": flip}
            for isl, p in (m.get("hurricane_conditions") or {}).items():
                r[f"p_{isl}"] = p
            r["cone"] = m.get("cone")
            rows.append(r)
    return pd.DataFrame(rows)


@_cached("reports")
def reports() -> pd.DataFrame:
    """Preliminary and final damage reports."""
    rows = []
    for s in manifest()["seasons"]:
        for m in _feed(s["dir"]):
            if m["type"] == "damage_report":
                rows.append({"season": s["season"], "round": m["round"], "stage": m["stage"], "time": m["time"],
                             "loss": m["industry_loss_musd"], **{f"dmg_{k}": v for k, v in m["damage_pct"].items()}})
    return pd.DataFrame(rows)


@_cached("impacts")
def impacts() -> pd.DataFrame:
    rows = []
    for s in manifest()["seasons"]:
        for m in _feed(s["dir"]):
            if m["type"] == "impact":
                rows.append({"season": s["season"], "round": m["round"], "island": m["island"], "age": m["age"]})
    return pd.DataFrame(rows)


@_cached("prices")
def prices() -> pd.DataFrame:
    """House consensus every 2 real s. `t` is real seconds since the round's 09:30 open."""
    frames = []
    for s in manifest()["seasons"]:
        df = pd.read_csv(HISTORY / s["dir"] / "market.csv")
        ts = pd.to_datetime(df["time"])
        df["t"] = (ts - ts.dt.normalize() - pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 30
        df.insert(0, "season", s["season"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def storm_t0(season: int, round: int) -> float:
    """Real seconds from the 09:30 open until the storm forms (pre_storm in case.json)."""
    return case()["timing"]["pre_storm"]


if __name__ == "__main__":
    for name, fn in [("storms", storms), ("seasons", seasons), ("advisories", advisories),
                     ("reports", reports), ("impacts", impacts), ("obs", obs), ("prices", prices)]:
        df = fn()
        print(f"{name:11s} {df.shape}")
