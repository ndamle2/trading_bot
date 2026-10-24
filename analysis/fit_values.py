"""Fit settlement value as a function of season damage and insured losses.

    ../.venv/bin/python fit_values.py

Island businesses: value = V0 - k * damage (tries summed and compounded damage, linear and quadratic).
Insurers:          value = top - slope * clip(loss - attach, 0, exhaust - attach)  (a layer).
Rebuilders:        value = base + a * (weighted damage) ** p  (diminishing returns), weights fitted.
Writes the fitted parameters to analysis/out/values.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from common import PHYSICAL, seasons

OUT = Path(__file__).resolve().parent / "out"


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def fit_physical(df):
    res = {}
    print("== Island businesses: settlement vs season damage")
    for c in PHYSICAL:
        y = df[f"set_{c}"].to_numpy()
        best = None
        for kind in ("dmg", "cdmg"):
            x = df[f"{kind}_{c}"].to_numpy() / 100
            for deg in (1, 2):
                coef = np.polyfit(x, y, deg)
                yhat = np.polyval(coef, x)
                sd = np.std(y - yhat)
                print(f"  {c:5s} {kind:4s} deg{deg}  coef={np.round(coef, 3)}  R2={r2(y, yhat):.4f}  resid_sd={sd:.3f}")
                if best is None or sd < best["resid_sd"] - 1e-3:
                    best = {"x": kind, "coef": coef.tolist(), "resid_sd": sd}
        res[c] = best
        print(f"  -> {c}: max value {y.max():.2f}, best = {best['x']} deg{len(best['coef']) - 1}")
    return res


def layer(p, L):
    top, attach, exhaust, bottom = p
    frac = np.clip((L - attach) / max(exhaust - attach, 1e-6), 0, 1)
    return top - (top - bottom) * frac


def fit_insurers(df):
    print("\n== Insurers: settlement vs season insured loss ($M)")
    L = df["loss"].to_numpy()
    res = {}
    for c, p0 in (("HAVEN", [75, 400, 1300, 21]), ("REIN", [90, 1300, 2500, 50])):
        y = df[f"set_{c}"].to_numpy()
        fit = least_squares(lambda p: layer(p, L) - y, p0)
        sd = np.std(layer(fit.x, L) - y)
        top, attach, exhaust, bottom = fit.x
        slope = (top - bottom) / (exhaust - attach)
        print(f"  {c:5s} top={top:.2f} attach=${attach:.0f}M exhaust=${exhaust:.0f}M bottom={bottom:.2f} "
              f"slope=${slope * 100:.2f}/share per $100M  resid_sd={sd:.3f}")
        res[c] = dict(zip(["top", "attach", "exhaust", "bottom"], fit.x.tolist()), resid_sd=sd)
    print("  (REIN 'exhaust'/'bottom' are extrapolated if losses never reach them: max loss in data "
          f"${L.max():.0f}M)")
    return res


def fit_rebuilders(df):
    print("\n== Rebuilders: settlement = base + a * (sum_i w_i * damage_i) ** p   (w_PORT fixed at 1)")
    D = np.column_stack([df[f"dmg_{c}"].to_numpy() / 100 for c in PHYSICAL])
    res = {}
    for c in ("RECON", "MAT", "GEN"):
        y = df[f"set_{c}"].to_numpy()

        def f(p):
            base, a, pw, *w = p
            x = D @ np.r_[1.0, np.abs(w)]
            return base + a * np.power(x + 1e-9, pw)

        fit = least_squares(lambda p: f(p) - y, [y.min(), 10, 0.7, 1, 1, 1, 1],
                            bounds=([0, -100, 0.05, 0, 0, 0, 0], [200, 200, 3, 20, 20, 20, 20]))
        yhat = f(fit.x)
        base, a, pw, *w = fit.x
        lin = np.linalg.lstsq(np.column_stack([np.ones(len(y)), D]), y, rcond=None)[0]
        print(f"  {c:5s} base={base:.2f} a={a:.2f} power={pw:.3f} weights(PORT..FERRY)="
              f"{np.round(np.r_[1, w], 2)}  R2={r2(y, yhat):.4f} resid_sd={np.std(y - yhat):.3f}")
        print(f"        linear check: intercept={lin[0]:.2f} per-100%-damage={dict(zip(PHYSICAL, np.round(lin[1:], 2)))}")
        res[c] = {"base": base, "a": a, "power": pw, "weights": dict(zip(PHYSICAL, np.r_[1, w].tolist())),
                  "resid_sd": float(np.std(y - yhat))}
    print("  Large residuals for GEN / GRID / FERRY are expected: outage duration is not in damage_pct.")
    return res


def main():
    df = seasons()
    out = {"physical": fit_physical(df), "insurers": fit_insurers(df), "rebuilders": fit_rebuilders(df)}
    OUT.mkdir(exist_ok=True)
    (OUT / "values.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT / 'values.json'}")


if __name__ == "__main__":
    main()
