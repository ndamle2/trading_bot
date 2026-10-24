"""Per-site storm exposure, and how it explains damage and outage costs.

    ../.venv/bin/python exposure.py

Wind model (calibrated from impact messages, see README): wind = intensity * (radius / d) ** K,
hurricane conditions (site closed) when wind > THRESH. For every storm and site this computes
  peak    the highest wind at the site
  closed  real seconds with wind above THRESH (outage time)
  excess  integral of max(wind - THRESH, 0) dt
Then it reports
  1. how well per-site features predict each company's damage_pct for a storm
  2. whether closure time explains what damage does not, in the settlement of GRID, FERRY, PORT, GEN
Writes analysis/out/exposure.pkl (one row per storm x site).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common import PHYSICAL, obs, seasons, sites, storms

K, THRESH = 1.3, 1.03
DT = 2.0
OUT = Path(__file__).resolve().parent / "out"


def wind(intensity, radius, d):
    return intensity * (radius / np.maximum(d, 1.0)) ** K


def site_exposure() -> pd.DataFrame:
    o, sd = obs(), sites()
    rows = []
    for (season, rnd), g in o.groupby(["season", "round"], sort=False):
        x, y, R, I = (g[c].to_numpy() for c in ("x", "y", "radius", "intensity"))
        for s in sd.itertuples():
            d = np.hypot(x - s.x, y - s.y)
            w = wind(I, R, d)
            rows.append((season, rnd, s.symbol, s.name, s.island, s.weight, d.min() / np.median(R),
                         w.max(), DT * (w > THRESH).sum(), DT * np.clip(w - THRESH, 0, None).sum()))
    return pd.DataFrame(rows, columns=["season", "round", "symbol", "site", "island", "weight",
                                       "min_dr", "peak", "closed", "excess"])


def damage_fit(ex: pd.DataFrame) -> None:
    """Company damage of a storm vs the weighted sum of a site damage curve f(peak)."""
    st = storms().set_index(["season", "round"])
    print("== Damage of a storm vs site exposure (weighted over sites)")
    for c in PHYSICAL:
        e = ex[ex.symbol == c]
        y = st[f"dmg_{c}"]
        feats = {}
        for name in ("peak", "excess"):
            feats[name] = (e.assign(v=e[name] * e.weight).groupby(["season", "round"]).v.sum())
        X = pd.DataFrame(feats).reindex(y.index)
        X["peak_over"] = (e.assign(v=np.clip(e.peak - THRESH, 0, None) * e.weight)
                          .groupby(["season", "round"]).v.sum()).reindex(y.index)
        for f in X:
            r = np.corrcoef(X[f], y)[0, 1]
            print(f"  {c:5s} corr(damage, weighted {f:9s}) = {r:.3f}")
    print("  Next step: fit a per-site damage curve damage = g(peak, closed) with weights from case.json.")


def outage_fit(ex: pd.DataFrame) -> None:
    """Settlement residual after damage vs total weighted closure time over the season."""
    df = seasons().set_index("season")
    closed = ex.assign(v=ex.closed * ex.weight).groupby(["season", "symbol"]).v.sum().unstack()
    print("\n== Settlement residual (after a linear damage fit) vs weighted closure seconds over the season")
    for c in ("PORT", "GRID", "FERRY"):
        X = np.column_stack([np.ones(len(df)), df[f"dmg_{c}"] / 100])
        y = df[f"set_{c}"].to_numpy()
        base = np.linalg.lstsq(X, y, rcond=None)[0]
        X2 = np.column_stack([X, closed.loc[df.index, c] / 100])
        full = np.linalg.lstsq(X2, y, rcond=None)[0]
        print(f"  {c:5s} resid sd: damage only {np.std(y - X @ base):.3f} -> with closure {np.std(y - X2 @ full):.3f}"
              f"   coef: V0={full[0]:.2f} per-100%-dmg={full[1]:.2f} per-100s-closed={full[2]:.3f}")
    grid_closed = closed.loc[df.index, "GRID"] / 100
    X = np.column_stack([np.ones(len(df)), grid_closed, np.sqrt(grid_closed)])
    y = df["set_GEN"].to_numpy()
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"  GEN   vs grid closure (lin + sqrt): resid sd {np.std(y - X @ c):.3f} (raw sd {y.std():.3f}) coef={np.round(c, 3)}")


def main():
    ex = site_exposure()
    OUT.mkdir(exist_ok=True)
    (OUT / "exposure.pkl").write_bytes(pickle.dumps(ex))
    print(ex.groupby("site")[["peak", "closed"]].mean().round(2).to_string(), "\n")
    damage_fit(ex)
    outage_fit(ex)


if __name__ == "__main__":
    main()
