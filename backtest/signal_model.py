"""Signal models: predict (settlement - house price) from the engine and the house, per symbol and phase.

    ../.venv/bin/python signal_model.py        # fit on training seasons, compare on the holdout, save the best

Candidates (all fitted on seasons 201-600 only; ridge strength chosen by season-grouped CV on the same seasons):
  beta       y_s = b * (fv_s - px_s)                                   (the original signal)
  beta+c     y_s = b * (fv_s - px_s) + c                               (adds the house's average bias in that phase)
  cross      y_s = sum_k b_k * (fv_k - px_k) / px_k * px_s + c        (every symbol's gap, as a fraction of price)
  cross+mom  cross + own house momentum (px_s now - px_s 20 s ago)
Writes backtest/out/signal_model.json with the chosen model's coefficients.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from common import SYMBOLS, prices, seasons  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def phase(rnd, t):
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


def wide(fv_file):
    """One row per (season, round, t): fv_*, px_*, px20_* (house 20 s earlier), set_*."""
    fv = pickle.loads((OUT / fv_file).read_bytes())
    p = prices()
    p = p[p.season.isin(fv.season.unique())][["season", "round", "t"] + SYMBOLS]
    lag = p.copy()
    lag["t"] = lag["t"] + 20
    df = fv.merge(p.rename(columns={s: f"px_{s}" for s in SYMBOLS}), on=["season", "round", "t"])
    df = df.merge(lag.rename(columns={s: f"px20_{s}" for s in SYMBOLS}), on=["season", "round", "t"], how="left")
    se = seasons().set_index("season")
    for s in SYMBOLS:
        df[f"set_{s}"] = df.season.map(se[f"set_{s}"])
        df[f"px20_{s}"] = df[f"px20_{s}"].fillna(df[f"px_{s}"])
    df["phase"] = [phase(r, t) for r, t in zip(df["round"], df["t"])]
    return df


def features(g, s, kind):
    gap = lambda k: (g[f"fv_{k}"] - g[f"px_{k}"]).to_numpy()  # noqa: E731
    own = gap(s)
    if kind == "beta":
        return own[:, None], False
    if kind == "beta+c":
        return own[:, None], True
    px = g[f"px_{s}"].to_numpy()
    X = [own] + [gap(k) / g[f"px_{k}"].to_numpy() * px for k in SYMBOLS if k != s]
    if kind == "cross+mom":
        X.append((g[f"px_{s}"] - g[f"px20_{s}"]).to_numpy())
    return np.column_stack(X), True


def ridge(X, y, alpha, intercept, own_free=True):
    """Ridge with an unpenalised own-gap coefficient (column 0) and intercept."""
    n, k = X.shape
    A = np.column_stack([X, np.ones(n)]) if intercept else X
    pen = np.full(A.shape[1], alpha * n)
    if own_free:
        pen[0] = 0.0
    if intercept:
        pen[-1] = 0.0
    coef = np.linalg.solve(A.T @ A + np.diag(pen), A.T @ y)
    return coef


def predict(X, coef, intercept):
    A = np.column_stack([X, np.ones(len(X))]) if intercept else X
    return A @ coef


def fit_all(tr, kind, alpha):
    models = {}
    for (ph), g in tr.groupby("phase"):
        for s in SYMBOLS:
            X, ic = features(g, s, kind)
            y = (g[f"set_{s}"] - g[f"px_{s}"]).to_numpy()
            a = 0.0 if kind.startswith("beta") else alpha
            models[(s, ph)] = (ridge(X, y, a, ic, own_free=True), ic)
    return models


def rmse(df, models, kind):
    out = {}
    for s in SYMBOLS:
        err = []
        for ph, g in df.groupby("phase"):
            X, ic = features(g, s, kind)
            coef, _ = models[(s, ph)]
            err.append((g[f"set_{s}"] - g[f"px_{s}"]).to_numpy() - predict(X, coef, ic))
        out[s] = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
    return out


def main():
    tr, ho = wide("fv_train_201-600.pkl"), wide("fv_train_1-200.pkl")
    pd.set_option("display.width", 220)
    # choose ridge alpha by 4-fold season-grouped CV on the training seasons
    folds = np.array_split(np.array(sorted(tr.season.unique())), 4)
    best_alpha = {}
    for kind in ("cross", "cross+mom"):
        scores = {}
        for alpha in (0.01, 0.1, 1.0, 10.0):
            err = []
            for f in folds:
                m = fit_all(tr[~tr.season.isin(f)], kind, alpha)
                err.append(np.mean(list(rmse(tr[tr.season.isin(f)], m, kind).values())))
            scores[alpha] = float(np.mean(err))
        best_alpha[kind] = min(scores, key=scores.get)
        print(f"{kind}: CV mean RMSE by alpha {dict((k, round(v, 4)) for k, v in scores.items())} -> {best_alpha[kind]}")

    rows, models = {}, {}
    house = {s: float(np.sqrt(((ho[f"set_{s}"] - ho[f"px_{s}"]) ** 2).mean())) for s in SYMBOLS}
    rows["house only"] = house
    for kind in ("beta", "beta+c", "cross", "cross+mom"):
        m = fit_all(tr, kind, best_alpha.get(kind, 0.0))
        models[kind] = m
        rows[kind] = rmse(ho, m, kind)
    t = pd.DataFrame(rows).T
    t["mean"] = t.mean(1)
    print("\nHOLDOUT RMSE of predicted settlement (all phases)\n", t.round(3))
    # per-round view for the chosen candidate
    choice = min(("beta", "beta+c", "cross", "cross+mom"), key=lambda k: t.loc[k, "mean"])
    print("\nchosen:", choice)
    for r in (1, 2, 3):
        sub = ho[ho["round"] == r]
        a, b = rmse(sub, models["beta"], "beta"), rmse(sub, models[choice], choice)
        print(f"  round {r}: beta mean {np.mean(list(a.values())):.3f} -> {choice} {np.mean(list(b.values())):.3f}")
    (OUT / "signal_model.json").write_text(json.dumps({
        "kind": choice, "alpha": best_alpha.get(choice, 0.0), "symbols": SYMBOLS,
        "coef": {f"{s}|{ph}": [list(map(float, c)), ic] for (s, ph), (c, ic) in models[choice].items()},
    }))
    print("wrote", OUT / "signal_model.json")


if __name__ == "__main__":
    main()
