"""Fair-value accuracy: RMSE of the engine (and of the house) against settlement, by storm phase.

    ../.venv/bin/python eval_fv.py out/fv_train_1-200.pkl [out/other.pkl ...]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from common import prices, seasons  # noqa: E402

SYMS = ["PORT", "ISLE", "AQUA", "GRID", "FERRY", "RECON", "MAT", "GEN", "HAVEN", "REIN", "ARCH", "REBLD"]
WINDOWS = {"pre": (0, 59), "storm 0-2m": (60, 179), "storm 2-4m": (180, 299), "storm 4-8m": (300, 539),
           "aftermath": (540, 739)}


def table(fv: pd.DataFrame, house: bool = False) -> pd.DataFrame:
    se = seasons().set_index("season")
    if house:
        p = prices()
        fv = fv[["season", "round", "t"]].merge(p, on=["season", "round", "t"])
        col = lambda s: fv[s]  # noqa: E731
    else:
        col = lambda s: fv[f"fv_{s}"]  # noqa: E731
    out = {}
    for name, (a, b) in WINDOWS.items():
        m = fv.t.between(a, b)
        out[name] = {s: float(np.sqrt(((col(s)[m] - fv.season[m].map(se[f"set_{s}"])) ** 2).mean())) for s in SYMS}
    return pd.DataFrame(out).T


def main():
    pd.set_option("display.width", 220)
    files = sys.argv[1:]
    for i, f in enumerate(files):
        fv = pickle.loads(Path(f).read_bytes())
        if i == 0:
            print("house RMSE vs settlement\n", table(fv, house=True).round(2), "\n")
        print(f"{f} RMSE vs settlement\n", table(fv).round(2), "\n")


if __name__ == "__main__":
    main()
