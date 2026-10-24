"""Fair-value engine shared by the backtest and the live bot.

    from engine.model import FairValue
    fv = FairValue.load("train")            # or "all" for live trading
    fv.on_message(msg)                      # every storm-feed message, in order
    mean, sd = fv.value()                   # {symbol: expected settlement}, {symbol: sd}

How a value is formed
  past storms     final report damage (exact) and closure computed from the observed track
  current storm   damage so far from the observed track (structural damage model), remainder from an
                  analog ensemble: library storms in a similar state at the same age, weighted by the
                  pattern posterior from advisories, with their future displacement applied to our storm
  future storms   whole storms sampled from the library, applied on top of the current site damage
  settlement      fitted valuation of damage, closure and insured loss; ETFs = basket
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = json.loads((HERE.parent / "hurricane-season-research" / "case.json").read_text())
PHYSICAL = ["PORT", "ISLE", "AQUA", "GRID", "FERRY"]
STOCKS = PHYSICAL + ["RECON", "MAT", "GEN", "HAVEN", "REIN"]
SYMBOLS = STOCKS + ["ARCH", "REBLD"]
OUTLOOKS = list(CASE["outlooks"])
SITES = [(sym, s) for sym, c in CASE["companies"].items() for s in c["sites"]]
SX = np.array([s["x"] for _, s in SITES], float)
SY = np.array([s["y"] for _, s in SITES], float)
SW = np.array([s["weight"] for _, s in SITES], float)
SSYM = np.array([sym for sym, _ in SITES])
COMP = np.stack([(SSYM == c) * SW for c in PHYSICAL])  # [5, sites] weight matrix
CLOSE_A, CLOSE_B = 1.02, 0.45  # site closed while d/R < A * I**B (fitted, see README)
DT = 2.0
N_STEPS = 241


def closure_seconds(x, y, R, I, sx=SX, sy=SY):
    """Seconds each site is closed along a track (arrays over time)."""
    dr = np.hypot(np.asarray(x)[None] - sx[:, None], np.asarray(y)[None] - sy[:, None]) / np.asarray(R)[None]
    return DT * np.sum(dr < CLOSE_A * np.power(np.maximum(I, 0), CLOSE_B)[None], axis=1)


class Physics:
    def __init__(self, p):
        self.bins = np.asarray(p["damage"]["bins"])
        self.g = np.asarray(p["damage"]["g"])
        self.v = np.asarray(p["damage"]["v"])

    def exposure(self, x, y, R, I):
        """(z, closed seconds) per site for track arrays [..., time]; leading dims broadcast."""
        x, y, R, I = (np.asarray(a, float) for a in (x, y, R, I))
        dr = np.hypot(x[..., None, :] - SX[:, None], y[..., None, :] - SY[:, None]) / R[..., None, :]
        k = np.clip(np.digitize(dr, self.bins) - 1, 0, len(self.g))
        g = np.r_[self.g, 0.0][k]  # beyond the last bin: no damage
        Ib = I[..., None, :]
        z = DT * np.sum(Ib * g, axis=-1)
        closed = DT * np.sum(dr < CLOSE_A * np.power(np.maximum(Ib, 0), CLOSE_B), axis=-1)
        return z, closed

    def site_damage(self, Z):
        return 1 - np.exp(-self.v * Z)

    def company(self, Z0, Z1):
        """damage_pct per company when cumulative site exposure moves Z0 -> Z1. [..., 5]"""
        return 100 * (self.site_damage(Z1) - self.site_damage(Z0)) @ COMP.T


class Valuation:
    def __init__(self, p, overrides: dict | None = None):
        self.v = p["valuation"]
        self.loss_coef = np.asarray(p["loss_coef"])
        self.overrides = overrides or {}

    def loss(self, Dpct):
        return Dpct @ self.loss_coef

    def __call__(self, Dpct, Cw) -> dict[str, np.ndarray]:
        """Settlement per symbol. Dpct: season damage pct [..., 5]; Cw: weighted closure/100 [..., 5]."""
        D = Dpct / 100
        out = {}
        for j, c in enumerate(PHYSICAL):
            c0, c1, c2, c3, c4 = self.v[c]["coef"]
            out[c] = c0 + c1 * D[..., j] + c2 * D[..., j] ** 2 + c3 * Cw[..., j] + c4 * Cw[..., j] ** 2
        L = self.loss(Dpct)
        for c in ("HAVEN", "REIN"):
            q = self.v[c]
            frac = np.clip((L - q["attach"]) / (q["exhaust"] - q["attach"]), 0, 1)
            out[c] = q["top"] - (q["top"] - q["bottom"]) * frac
        X = np.concatenate([D, Cw[..., [0, 3, 4]]], axis=-1)
        for c in ("RECON", "MAT", "GEN"):
            q = self.v[c]
            out[c] = q["base"] + q["a"] * np.power(np.maximum(X @ np.asarray(q["w"]), 0) + 1e-9, q["power"])
        for c, o in self.overrides.items():  # manual adjustments for announcements: value*mult + add
            if c in out:
                out[c] = out[c] * o.get("mult", 1.0) + o.get("add", 0.0)
        for etf, spec in CASE["etfs"].items():
            out[etf] = sum(q * out[s] for s, q in spec["basket"].items())
        return out


@dataclass
class Storm:
    """What we know about the storm of the current round."""
    name: str = ""
    obs: list = field(default_factory=list)      # (age, x, y, vx, vy, R, I)
    adv: list = field(default_factory=list)      # (source, outlook index)
    cone: list = field(default_factory=list)     # latest official cone: (age, x, y, spread)
    prelim: np.ndarray | None = None
    final: np.ndarray | None = None
    final_loss: float | None = None
    dissipated: bool = False


class FairValue:
    def __init__(self, params: dict, lib: dict, n_samples: int = 2000, k_analogs: int = 50, seed: int = 0,
                 exclude_season: int | None = None, overrides: dict | None = None, cone_sigma: float | None = None,
                 cone_k: int | None = None):
        self.p = params
        self.phys = Physics(params)
        self.val = Valuation(params, overrides)
        keep = np.ones(len(lib["season"]), bool) if exclude_season is None else lib["season"] != exclude_season
        self.T = lib["tracks"][keep]
        self.pat = lib["pattern"][keep]
        self.lib_z = lib["bins"][keep] @ self.phys.g          # [N, sites] whole-storm exposure
        self.lib_c = lib["closure"][keep]                     # [N, sites] whole-storm closure seconds
        self._noise_model(lib["resid"][keep], lib["pred"][keep])
        adv = params["advisory"]
        self.prior = np.asarray(adv["prior"])
        self.lik = {k: np.asarray(v) for k, v in adv["likelihood"].items()}
        self.blend = params["prelim_blend"]
        self.n, self.k = n_samples, k_analogs
        # analog futures are weighted by how well they pass through the official forecast cone: each cone point is a
        # 2-D gaussian with sd = cone_sigma * spread (None disables). cone_k analogs are drawn before reweighting.
        self.cone_sigma, self.cone_k = cone_sigma, cone_k or k_analogs
        self.rng = np.random.default_rng(seed)
        self.reset_season()

    @classmethod
    def load(cls, tag: str = "train", **kw) -> "FairValue":
        d = HERE / tag
        lib = dict(np.load(d / "library.npz"))
        return cls(json.loads((d / "params.json").read_text()), lib, **kw)

    # ---------------------------------------------------------------- noise
    def _noise_model(self, resid, pred):
        """Residual sd per company grows with predicted damage: sd = a + b * pred. Pool of standardized vectors."""
        self.nsd = np.zeros((5, 2))
        for j in range(5):
            X = np.column_stack([np.ones(len(pred)), pred[:, j]])
            self.nsd[j] = np.linalg.lstsq(X, np.abs(resid[:, j]) * np.sqrt(np.pi / 2), rcond=None)[0]
        self.nsd[:, 0] = np.maximum(self.nsd[:, 0], 0.05)
        self.npool = resid / self._sd(pred)

    def _sd(self, pred):
        return np.maximum(self.nsd[:, 0] + self.nsd[:, 1] * np.maximum(pred, 0), 0.05)

    def _noisy(self, pred, scale=1.0):
        e = self.npool[self.rng.integers(len(self.npool), size=pred.shape[0])]
        return np.maximum(pred + scale * e * self._sd(pred), 0.0)

    # ---------------------------------------------------------------- season state
    def reset_season(self):
        self.round = 0
        self.D_done = np.zeros(5)        # final-report damage pct of completed storms
        self.Z_done = np.zeros(len(SX))  # modelled cumulative site exposure of completed storms
        self.C_done = np.zeros(5)        # weighted closure / 100 of completed storms
        self.storm = Storm()
        self._cache = None

    def _close_round(self):
        """Fold the current storm into the season totals (called when its final report arrives)."""
        s = self.storm
        z, c = self._accrued()
        self.D_done = self.D_done + s.final
        self.Z_done = self.Z_done + z
        self.C_done = self.C_done + (COMP @ c) / 100

    def on_message(self, m: dict) -> None:
        t = m.get("type")
        rnd = m.get("round")
        if rnd is not None and rnd != self.round:
            if rnd < self.round:  # a new season started
                self.reset_season()
            self.round, self.storm = rnd, Storm(name=m.get("storm", ""))
        s = self.storm
        if t == "storm":
            s.obs.append((m["age"], m["x"], m["y"], m["vx"], m["vy"], m["radius"], m["intensity"]))
        elif t == "advisory" and m.get("outlook") in OUTLOOKS:
            s.adv.append((m["source"], OUTLOOKS.index(m["outlook"])))
            if m.get("source") == "official" and m.get("cone"):
                s.cone = [(c["age"], c["x"], c["y"], c["spread"]) for c in m["cone"]]
        elif t == "storm_dissipated":
            s.dissipated = True
        elif t == "damage_report":
            d = np.array([m["damage_pct"][c] for c in PHYSICAL], float)
            if m["stage"] == "preliminary":
                s.prelim = d
            elif s.final is None:
                s.final, s.final_loss = d, m["industry_loss_musd"]
                self._close_round()
        else:
            return
        self._cache = None

    # ---------------------------------------------------------------- current storm
    def _accrued(self):
        if not self.storm.obs:
            return np.zeros(len(SX)), np.zeros(len(SX))
        o = np.asarray(self.storm.obs)
        return self.phys.exposure(o[:, 1], o[:, 2], o[:, 5], o[:, 6])

    def pattern_posterior(self):
        logp = np.log(self.prior)
        for src, k in self.storm.adv:
            logp = logp + (1.0 if src == "official" else 0.8) * np.log(self.lik[src][:, k])
        p = np.exp(logp - logp.max())
        return p / p.sum()

    def _analogs(self):
        """Remaining (z, closure) per site for k analog futures, with weights."""
        o = np.asarray(self.storm.obs)
        i = min(int(o[-1, 0] // 2), N_STEPS - 1)
        if i >= N_STEPS - 1:
            return np.zeros((1, len(SX))), np.zeros((1, len(SX))), np.ones(1)
        j0 = max(i - 10, 0)
        span = max(o[-1, 0] - o[max(len(o) - 11, 0), 0], 1)
        live_v = (o[-1, 1:3] - o[max(len(o) - 11, 0), 1:3]) / span if len(o) > 1 else o[-1, 3:5]
        T = self.T
        lib_v = (T[:, i, :2] - T[:, j0, :2]) / max((i - j0) * DT, 1) if i > 0 else T[:, i, 2:4]
        d2 = (((T[:, i, 0] - o[-1, 1]) / 40) ** 2 + ((T[:, i, 1] - o[-1, 2]) / 40) ** 2
              + (((lib_v - live_v) / 0.4) ** 2).sum(1)
              + ((T[:, i, 4] - o[-1, 5]) / 15) ** 2 + ((T[:, i, 5] - o[-1, 6]) / 0.15) ** 2)
        # nearest states first, then reweight by the advisory likelihood ratio (the library already holds each
        # pattern at its prior rate, so weighting by the posterior itself would count the prior twice)
        cone = [c for c in self.storm.cone if c[0] > o[-1, 0] and c[0] <= 480] if self.cone_sigma else []
        k = self.cone_k if cone else self.k
        top = np.argpartition(d2, k)[:k]
        lr = self.pattern_posterior() / self.prior
        logw = -0.5 * d2[top] + np.log(lr[self.pat[top]] + 1e-12)
        fut = T[top, i + 1:]
        x = o[-1, 1] + fut[..., 0] - T[top, i, 0][:, None]
        y = o[-1, 2] + fut[..., 1] - T[top, i, 1][:, None]
        for age, cx, cy, spread in cone:
            j = int(age // 2) - (i + 1)
            if 0 <= j < x.shape[1]:
                logw = logw - 0.5 * ((x[:, j] - cx) ** 2 + (y[:, j] - cy) ** 2) / (self.cone_sigma * spread) ** 2
        w = np.exp(logw - logw.max())
        R = o[-1, 5] * fut[..., 4] / T[top, i, 4][:, None]
        I = o[-1, 6] * fut[..., 5] / np.maximum(T[top, i, 5], 1e-3)[:, None]
        z, c = self.phys.exposure(x, y, R, I)
        return z, c, w / w.sum()

    def _current_samples(self):
        """Samples of (damage pct [n,5], closure weighted [n,5], Z after storm [n,sites]) for this round's storm."""
        s, n = self.storm, self.n
        z_acc, c_acc = self._accrued()
        if s.final is not None:  # already folded into the *_done totals
            return np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((n, len(SX)))
        if not s.obs:  # storm not formed yet: an unknown storm from the library
            return self._future_storm(np.broadcast_to(self.Z_done, (n, len(SX))))
        if s.dissipated or s.obs[-1][0] >= 480:
            dz, dc, w = np.zeros((1, len(SX))), np.zeros((1, len(SX))), np.ones(1)
        else:
            dz, dc, w = self._analogs()
        m = self.rng.choice(len(w), size=n, p=w)
        Z1 = self.Z_done + z_acc + dz[m]
        pred = self.phys.company(self.Z_done, Z1)
        if s.prelim is not None:
            b = np.array([self.blend[c] for c in PHYSICAL])
            D = pred * b[:, 0] + s.prelim * b[:, 1]
            D = self._noisy(D, scale=0.6)
        else:
            D = self._noisy(pred)
        C = (c_acc + dc[m]) @ COMP.T / 100
        return D, C, Z1

    def _future_storm(self, Z0):
        idx = self.rng.integers(len(self.lib_z), size=Z0.shape[0])
        Z1 = Z0 + self.lib_z[idx]
        D = self._noisy(self.phys.company(Z0, Z1))
        return D, self.lib_c[idx] @ COMP.T / 100, Z1

    # ---------------------------------------------------------------- value
    def samples(self):
        D, C, Z = self._current_samples()
        if self.storm.final is not None:
            Z = np.broadcast_to(self.Z_done, (self.n, len(SX)))
        D = D + self.D_done
        C = C + self.C_done
        for _ in range(max(self.round, 1) + 1, 4):  # storms of later rounds
            d, c, Z = self._future_storm(Z)
            D, C = D + d, C + c
        return self.val(D, C)

    def value(self):
        if self._cache is None:
            v = self.samples()
            self._cache = ({k: float(np.mean(a)) for k, a in v.items()}, {k: float(np.std(a)) for k, a in v.items()})
        return self._cache
