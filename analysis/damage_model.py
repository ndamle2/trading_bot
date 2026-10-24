"""Structural damage model: storm track -> damage of each site -> damage_pct of each company.

    ../.venv/bin/python damage_model.py          # fit on the training seasons, report on the holdout

Model (fitted, not given):
  exposure  z_site = sum_t  I_t * g(d_t / R_t) * dt          g = piecewise-constant kernel on d/R bins
  site      cumulative site damage s = 1 - exp(-v_site * Z_site), Z = exposure summed over the season so far
  company   damage_pct of a storm = 100 * sum_site weight * (s_after - s_before)
The kernel g is shared by all sites; v_site is each site's vulnerability.
Writes analysis/out/damage_model.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from common import PHYSICAL, obs, sites, storms

OUT = Path(__file__).resolve().parent / "out"
BINS = np.r_[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0]
TRAIN = lambda season: season > 200  # noqa: E731  seasons 1-200 are the holdout
DT = 2.0


def bin_exposure(x, y, R, I, sx, sy) -> np.ndarray:
    """[n_sites, n_bins]: intensity-weighted seconds the site spent in each d/R bin."""
    dr = np.hypot(x[None, :] - sx[:, None], y[None, :] - sy[:, None]) / R[None, :]
    idx = np.digitize(dr, BINS) - 1
    out = np.zeros((len(sx), len(BINS) - 1))
    for k in range(len(BINS) - 1):
        out[:, k] = DT * np.sum(I[None, :] * (idx == k), axis=1)
    return out


def storm_bins() -> tuple[np.ndarray, list]:
    """[n_storms, n_sites, n_bins] in the row order of storms()."""
    sd = sites()
    sx, sy = sd.x.to_numpy(), sd.y.to_numpy()
    o = obs()
    groups = {k: g for k, g in o.groupby(["season", "round"], sort=False)}
    st = storms()
    B = np.stack([bin_exposure(*(groups[(s, r)][c].to_numpy() for c in ("x", "y", "radius", "intensity")), sx, sy)
                  for s, r in zip(st.season, st["round"])])
    return B, list(zip(st.season, st["round"]))


class DamageModel:
    def __init__(self, g, v):
        self.g, self.v = np.asarray(g), np.asarray(v)
        sd = sites()
        self.sym = sd.symbol.to_numpy()
        self.w = sd.weight.to_numpy()

    def z(self, B):
        return B @ self.g  # [..., sites]

    def site_damage(self, Zcum):
        return 1 - np.exp(-self.v * Zcum)

    def company(self, s_before, s_after) -> np.ndarray:
        """damage_pct per company for a storm that moves site damage from s_before to s_after."""
        inc = 100 * self.w * (s_after - s_before)
        return np.stack([inc[..., self.sym == c].sum(-1) for c in PHYSICAL], -1)

    def to_json(self):
        return {"bins": BINS.tolist(), "g": self.g.tolist(), "v": self.v.tolist()}

    @classmethod
    def load(cls, path=OUT / "damage_model.json"):
        d = json.loads(Path(path).read_text())
        return cls(d["g"], d["v"])


def season_predictions(model: DamageModel, B, keys):
    """Predicted damage_pct per storm, carrying site damage through each season."""
    pred = np.zeros((len(keys), len(PHYSICAL)))
    Z = None
    for i, (s, r) in enumerate(keys):
        if r == 1:
            Z = np.zeros(B.shape[1])
        z = model.z(B[i])
        pred[i] = model.company(model.site_damage(Z), model.site_damage(Z + z))
        Z = Z + z
    return pred


def fit(B, keys, Y, mask):
    n_bins, n_sites = B.shape[2], B.shape[1]

    def unpack(p):
        return np.exp(p[:n_bins]), np.exp(np.r_[0.0, p[n_bins:]])  # v of site 0 fixed (scale)

    def resid(p):
        g, v = unpack(p)
        P = season_predictions(DamageModel(g, v), B, keys)
        return (P - Y)[mask].ravel()

    p0 = np.r_[np.log(np.full(n_bins, 3e-3)), np.zeros(n_sites - 1)]
    sol = least_squares(resid, p0, method="trf", x_scale="jac", max_nfev=300)
    return DamageModel(*unpack(sol.x))


def main():
    st = storms()
    B, keys = storm_bins()
    Y = st[[f"dmg_{c}" for c in PHYSICAL]].to_numpy()
    train = st.season.map(TRAIN).to_numpy()
    model = fit(B, keys, Y, train)
    OUT.mkdir(exist_ok=True)
    (OUT / "damage_model.json").write_text(json.dumps(model.to_json(), indent=2))
    np.save(OUT / "storm_bins.npy", B)

    P = season_predictions(model, B, keys)
    print("kernel g by d/R bin:", dict(zip(np.round(BINS[:-1], 2), np.round(model.g * 1e3, 3))))
    sd = sites()
    print("site vulnerability v:", dict(zip(sd.name, np.round(model.v, 3))))
    for name, m in (("train", train), ("holdout", ~train)):
        err = P[m] - Y[m]
        print(f"{name:8s} per-storm damage error sd:", dict(zip(PHYSICAL, np.round(err.std(0), 2))),
              " bias:", dict(zip(PHYSICAL, np.round(err.mean(0), 2))))
    print(f"wrote {OUT / 'damage_model.json'}")


if __name__ == "__main__":
    main()
