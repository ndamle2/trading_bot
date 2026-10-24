"""Where is the house consensus wrong? Compare house prices with settlement and with report values.

    ../.venv/bin/python house_errors.py

Reports
  1. house price minus settlement, by symbol, at key moments of each round (mean = bias, sd = error)
  2. preliminary vs final damage report error
  3. ETF price vs basket value (house consensus), to see if the ETFs drift
Real-time offsets in a round (case.json timing): storm forms at 60 s, dissipates at 540 s,
preliminary report at 570 s, final report at 740 s, close at 780 s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import PHYSICAL, SYMBOLS, case, prices, reports, seasons

pd.set_option("display.width", 220)
MOMENTS = {"open": 0, "storm+60": 120, "storm+240": 300, "dissipated": 540, "prelim+10": 580,
           "final+10": 750, "close": 778}


def at(p: pd.DataFrame, t: float) -> pd.DataFrame:
    """The house price of each season and round at real second t (nearest sample at or after)."""
    return p[p.t >= t].groupby(["season", "round"]).first()


def main():
    p = prices()
    settle = seasons().set_index("season")[[f"set_{s}" for s in SYMBOLS]]
    settle.columns = SYMBOLS

    print("== house price - settlement value (mean / sd), in dollars")
    for rnd in (1, 2, 3):
        print(f"\n-- round {rnd}")
        rows = {}
        for name, t in MOMENTS.items():
            snap = at(p, t).xs(rnd, level="round")[SYMBOLS]
            err = snap - settle.loc[snap.index]
            rows[name] = err.mean().round(2).astype(str) + " / " + err.std().round(2).astype(str)
        print(pd.DataFrame(rows).T.to_string())

    print("\n== preliminary - final damage report")
    r = reports()
    cols = ["loss"] + [f"dmg_{c}" for c in PHYSICAL]
    pre = r[r.stage == "preliminary"].set_index(["season", "round"])[cols]
    fin = r[r.stage == "final"].set_index(["season", "round"])[cols]
    d = pre - fin
    print(pd.DataFrame({"mean": d.mean(), "sd": d.std(), "corr": [pre[c].corr(fin[c]) for c in cols]}).round(3))

    print("\n== ETF house price minus house basket value")
    for etf, spec in case()["etfs"].items():
        nav = sum(q * p[s] for s, q in spec["basket"].items())
        gap = p[etf] - nav
        print(f"  {etf:5s} mean {gap.mean():+.3f}  sd {gap.std():.3f}  |gap|>1: {np.mean(np.abs(gap) > 1):.1%}")


if __name__ == "__main__":
    main()
