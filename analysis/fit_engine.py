"""Fit every parameter the fair-value engine needs and write engine/params.json + engine/library.npz.

    ../.venv/bin/python fit_engine.py            # fit on training seasons only (201-600), for the backtest
    ../.venv/bin/python fit_engine.py --all      # fit on all 600 seasons, for live trading

Writes into engine/<tag>/ (tag = "train" or "all"):
  params.json   damage kernel + site vulnerabilities, closure rule, valuation of each symbol, insured-loss
                coefficients, preliminary-report blend, residual pools, advisory likelihoods
  library.npz   the analog storm library: every fitted storm's track, pattern, exposure and closure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from common import ISLANDS, OUTLOOKS, PHYSICAL, advisories, obs, reports, seasons, sites, storms
from damage_model import BINS, DamageModel, fit as fit_damage, season_predictions, storm_bins

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.model import CLOSE_A, CLOSE_B, closure_seconds  # noqa: E402

N_STEPS = 241  # observations per storm (age 0..480 every 2 s)


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def track_arrays(keys):
    o = obs()
    groups = {k: g for k, g in o.groupby(["season", "round"], sort=False)}
    T = np.zeros((len(keys), N_STEPS, 6))
    for i, k in enumerate(keys):
        T[i] = groups[k][["x", "y", "vx", "vy", "radius", "intensity"]].to_numpy()
    return T


def season_closure(closure, keys, sd):
    """Weighted closure seconds / 100 summed over each season, per company (PORT, GRID, FERRY, ISLE, AQUA)."""
    out = {}
    w, sym = sd.weight.to_numpy(), sd.symbol.to_numpy()
    for i, (s, _) in enumerate(keys):
        out.setdefault(s, np.zeros(len(sym)))
        out[s] = out[s] + closure[i]
    return {s: np.array([(c[sym == comp] * w[sym == comp]).sum() / 100 for comp in PHYSICAL]) for s, c in out.items()}


def fit_valuation(se, C, seasons_used):
    """Settlement as a function of season damage D (fraction), closure Cw and insured loss L."""
    se = se.set_index("season").loc[seasons_used]
    D = np.column_stack([se[f"dmg_{c}"] / 100 for c in PHYSICAL])
    Cw = np.stack([C[s] for s in seasons_used])
    L = se["loss"].to_numpy()
    val, report = {}, {}
    for j, c in enumerate(PHYSICAL):
        y = se[f"set_{c}"].to_numpy()
        X = np.column_stack([np.ones(len(y)), D[:, j], D[:, j] ** 2, Cw[:, j], Cw[:, j] ** 2])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        val[c] = {"form": "quad_dc", "coef": coef.tolist()}
        report[c] = np.std(y - X @ coef)
    # insurers: layers
    for c, p0 in (("HAVEN", [75, 400, 1300, 21]), ("REIN", [90, 1300, 2500, 50])):
        y = se[f"set_{c}"].to_numpy()

        def layer(p, L=L):
            top, a, e, b = p
            return top - (top - b) * np.clip((L - a) / max(e - a, 1e-6), 0, 1)

        p = least_squares(lambda p: layer(p) - y, p0).x
        val[c] = {"form": "layer", "top": p[0], "attach": p[1], "exhaust": p[2], "bottom": p[3]}
        if c == "REIN":  # exhaust never observed: keep the slope, push exhaust far out
            slope = (p[0] - p[3]) / (p[2] - p[1])
            val[c].update(exhaust=p[1] + p[0] / slope, bottom=0.0)
        report[c] = np.std(layer(p) - y)
    # rebuilders: base + a * (w . x) ** p, x = damages and closures
    X = np.column_stack([D, Cw[:, [0, 3, 4]]])  # PORT, GRID, FERRY closures
    for c in ("RECON", "MAT", "GEN"):
        y = se[f"set_{c}"].to_numpy()

        def f(p, X=X):
            base, a, pw, *w = p
            return base + a * np.power(X @ np.abs(w) + 1e-9, pw)

        p0 = [y.min() - 5, 10, 0.5] + [1.0] * X.shape[1]
        p = least_squares(lambda p: f(p) - y, p0, x_scale="jac", max_nfev=5000).x
        val[c] = {"form": "power", "base": p[0], "a": p[1], "power": p[2], "w": np.abs(p[3:]).tolist()}
        report[c] = np.std(f(p) - y)
    print("valuation residual sd (train):", {k: round(v, 3) for k, v in report.items()})
    return val


def fit_advisory_likelihood(adv, seasons_used):
    """P(outlook said | pattern at formation), per source, with add-one smoothing."""
    a = adv[adv.season.isin(seasons_used)]
    out = {}
    for src in ("official", "wire"):
        s = a[a.source == src]
        M = np.ones((len(OUTLOOKS), len(OUTLOOKS)))  # [true, said]
        for t, said in zip(s.pattern, s.outlook):
            M[OUTLOOKS.index(t), OUTLOOKS.index(said)] += 1
        out[src] = (M / M.sum(1, keepdims=True)).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    tag = "all" if args.all else "train"
    out_dir = ROOT / "engine" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    st, se, sd = storms(), seasons(), sites()
    use = st.season.to_numpy() > (0 if args.all else 200)
    seasons_used = sorted(st.season[use].unique())
    B, keys = storm_bins()
    Y = st[[f"dmg_{c}" for c in PHYSICAL]].to_numpy()

    print("fitting damage model ...")
    dm = fit_damage(B, keys, Y, use)
    P = season_predictions(dm, B, keys)
    resid = (Y - P)[use]
    print("damage residual sd:", dict(zip(PHYSICAL, resid.std(0).round(2))))

    T = track_arrays(keys)
    closure = np.stack([closure_seconds(T[i, :, 0], T[i, :, 1], T[i, :, 4], T[i, :, 5],
                                        sd.x.to_numpy(), sd.y.to_numpy()) for i in range(len(keys))])
    C = season_closure(closure, keys, sd)

    loss_X = np.column_stack([Y[use]])
    loss_coef = np.linalg.lstsq(loss_X, st.loss.to_numpy()[use], rcond=None)[0]
    print("loss coefficients ($M per damage point):", dict(zip(PHYSICAL, loss_coef.round(3))))

    val = fit_valuation(se, C, seasons_used)

    # preliminary report blend: final ~ a * model + b * prelim  (per company), on fitted storms
    r = reports()
    pre = r[r.stage == "preliminary"].set_index(["season", "round"])
    blend = {}
    idx = [i for i, k in enumerate(keys) if use[i]]
    for j, c in enumerate(PHYSICAL):
        pr = pre.loc[[keys[i] for i in idx], f"dmg_{c}"].to_numpy()
        X = np.column_stack([P[idx, j], pr])
        coef = np.linalg.lstsq(X, Y[idx, j], rcond=None)[0]
        blend[c] = coef.tolist()
        print(f"  blend {c}: final = {coef[0]:.3f} model + {coef[1]:.3f} prelim, resid sd "
              f"{np.std(Y[idx, j] - X @ coef):.2f} (model alone {np.std(Y[idx, j] - P[idx, j]):.2f}, "
              f"prelim alone {np.std(Y[idx, j] - pr):.2f})")

    pattern_idx = np.array([OUTLOOKS.index(p) for p in st.pattern])
    prior = np.bincount(pattern_idx[use], minlength=len(OUTLOOKS)) + 1.0
    params = {
        "tag": tag, "seasons": [int(seasons_used[0]), int(seasons_used[-1])],
        "damage": dm.to_json(), "closure": {"a": CLOSE_A, "b": CLOSE_B},
        "loss_coef": loss_coef.tolist(), "valuation": val, "prelim_blend": blend,
        "advisory": {"prior": (prior / prior.sum()).tolist(), "likelihood": fit_advisory_likelihood(advisories(), seasons_used)},
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=1, default=float))
    np.savez_compressed(out_dir / "library.npz", tracks=T[use], pattern=pattern_idx[use],
                        season=st.season.to_numpy()[use], round=st["round"].to_numpy()[use],
                        bins=B[use], closure=closure[use], resid=resid, pred=P[use])
    print(f"wrote {out_dir}/params.json and library.npz ({use.sum()} storms)")


if __name__ == "__main__":
    main()
