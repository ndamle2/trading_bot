"""How accurate are official advisories and wire chatter, and how early do they get it right?

    ../.venv/bin/python source_accuracy.py

Reports
  1. hit rate of each source by storm age (vs the true pattern at that age, erratic after a flip)
  2. confusion matrix: what the source says vs the truth
  3. P(true pattern | outlook said) — what to believe when you see a message
  4. calibration of the official hurricane_conditions probabilities against impact messages
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ISLANDS, OUTLOOKS, advisories, impacts, storms

pd.set_option("display.width", 200)


def main():
    a = advisories()
    a["hit"] = a.outlook == a.true_now
    a["age_bin"] = pd.cut(a.age, [0, 60, 120, 180, 240, 300, 360, 420, 480], include_lowest=True)

    print("== Base rates of true patterns (at formation)")
    print(storms().pattern.value_counts(normalize=True).round(3).to_string())
    print("  storms that turn erratic:", f"{storms().erratic_at.notna().mean():.1%}")

    print("\n== Hit rate by source and storm age")
    print(a.pivot_table(index="age_bin", columns="source", values="hit", aggfunc="mean", observed=True).round(3))
    print(a.groupby("source").hit.agg(["mean", "count"]).round(3))

    for src in ("official", "wire"):
        s = a[a.source == src]
        print(f"\n== {src}: P(truth | said)   rows = what the source said")
        print(pd.crosstab(s.outlook, s.true_now, normalize="index").reindex(columns=OUTLOOKS).round(2))

    # official advisories by advisory number: does the first one already know?
    o = a[a.source == "official"]
    print("\n== official hit rate by advisory number")
    print(o.groupby("number").hit.agg(["mean", "count"]).round(3).T)

    # does the wire agreeing with the official help?
    print("\n== consecutive wire messages: P(hit) when the wire agrees with the latest official advisory")
    rows = []
    for (season, rnd), g in a.sort_values("age").groupby(["season", "round"]):
        last = None
        for r in g.itertuples():
            if r.source == "official":
                last = r.outlook
            elif last is not None:
                rows.append((r.outlook == last, r.hit))
    w = pd.DataFrame(rows, columns=["agrees", "hit"])
    print(w.groupby("agrees").hit.agg(["mean", "count"]).round(3))

    # calibration of hurricane_conditions vs whether an impact on that island occurred
    imp = impacts()
    hit_isl = set(zip(imp.season, imp["round"], imp.island))
    rows = []
    for r in o.itertuples():
        for isl in ISLANDS:
            p = getattr(r, f"p_{isl}")
            if pd.notna(p):
                rows.append((r.number, p, (r.season, r.round, isl) in hit_isl))
    c = pd.DataFrame(rows, columns=["number", "p", "impact"])
    c["p_bin"] = pd.cut(c.p, np.linspace(0, 1, 11), include_lowest=True)
    print("\n== calibration of official hurricane_conditions (any impact on the island during the storm)")
    print(c.groupby("p_bin", observed=True).impact.agg(["mean", "count"]).round(3).T)
    print("  (note: later advisories are conditioned on impacts that already happened)")


if __name__ == "__main__":
    main()
