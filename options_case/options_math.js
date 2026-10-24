// Black-Scholes (r = 0) and helpers for the options case. Vol is a decimal (0.35 = 35%), T in years.
(function (root) {
  function erf(x) {
    // Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)
    const s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const N = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
  const phi = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

  // returns {price, delta, gamma, vega (per 1 vol point = 0.01)}
  function bs(type, S, K, vol, T) {
    const call = type === "C";
    if (!(S > 0) || !(K > 0)) return { price: NaN, delta: NaN, gamma: NaN, vega: NaN };
    if (!(vol > 0) || !(T > 0)) {
      const intr = call ? Math.max(S - K, 0) : Math.max(K - S, 0);
      const d = call ? (S > K ? 1 : 0) : (S < K ? -1 : 0);
      return { price: intr, delta: d, gamma: 0, vega: 0 };
    }
    const sq = vol * Math.sqrt(T);
    const d1 = (Math.log(S / K) + 0.5 * vol * vol * T) / sq;
    const d2 = d1 - sq;
    const price = call ? S * N(d1) - K * N(d2) : K * N(-d2) - S * N(-d1);
    return {
      price,
      delta: call ? N(d1) : N(d1) - 1,
      gamma: phi(d1) / (S * sq),
      vega: S * phi(d1) * Math.sqrt(T) / 100,
    };
  }

  // implied vol by bisection; NaN if the price is outside no-arbitrage bounds
  function impliedVol(type, S, K, price, T) {
    const lo0 = bs(type, S, K, 1e-6, T).price, hi0 = bs(type, S, K, 5, T).price;
    if (!(price >= lo0 - 1e-9) || !(price <= hi0)) return NaN;
    let lo = 1e-6, hi = 5;
    for (let i = 0; i < 100; i++) {
      const mid = 0.5 * (lo + hi);
      if (bs(type, S, K, mid, T).price < price) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // Round 3: candidate meanings of "a quantity linked to IV" -> annualised vol (decimal)
  // q is as published; S and T for dollar-based quantities. Percent-style inputs (e.g. 35) are handled by `pct`.
  const CONVERTERS = {
    "IV itself":                       (q) => q,
    "variance (σ²)":                   (q) => Math.sqrt(q),
    "daily vol (σ/√252)":              (q) => q * Math.sqrt(252),
    "daily variance (σ²/252)":         (q) => Math.sqrt(q * 252),
    "expected move to expiry, $ (1sd)": (q, S, T) => q / (S * Math.sqrt(T)),
    "expected move to expiry, % (1sd)": (q, S, T) => q / Math.sqrt(T),
    "daily expected move, $ (1sd)":    (q, S) => q / S * Math.sqrt(252),
    "ATM straddle price, $":           (q, S, T) => q / (Math.sqrt(2 / Math.PI) * S * Math.sqrt(T)),
    "variance to expiry (σ²T)":        (q, S, T) => Math.sqrt(q / T),
  };

  // Analyst scoring. history: [{est: [a1, a2, a3], truth}] oldest first (vol in any consistent unit).
  // Recent events count more (decay per event). Returns per-analyst rmse, bias (est - truth) and blend weights.
  function analystStats(history, decay = 0.6, floor = 1.0) {
    const n = 3;
    const rows = history.filter((r) => isFinite(r.truth) && r.est.some(isFinite));
    const sw = Array(n).fill(0), se = Array(n).fill(0), sb = Array(n).fill(0), cnt = Array(n).fill(0);
    rows.forEach((r, j) => {
      const w = Math.pow(decay, rows.length - 1 - j);
      r.est.forEach((e, i) => {
        if (!isFinite(e)) return;
        sw[i] += w; se[i] += w * (e - r.truth) ** 2; sb[i] += w * (e - r.truth); cnt[i] += 1;
      });
    });
    const rmse = sw.map((w, i) => (w > 0 ? Math.sqrt(se[i] / w) : NaN));
    const bias = sw.map((w, i) => (w > 0 ? sb[i] / w : 0));
    // no history -> equal weights; else inverse mean-squared error with a floor so one lucky call can't dominate
    const weight = rmse.map((e) => (isFinite(e) ? 1 / (e * e + floor * floor) : NaN));
    const known = weight.filter(isFinite);
    const fill = known.length ? Math.min(...known) : 1;
    const w = weight.map((x) => (isFinite(x) ? x : fill));
    const tot = w.reduce((a, b) => a + b, 0);
    return { rmse, bias, weights: w.map((x) => x / tot), count: cnt, events: rows.length };
  }

  // weighted blend of current estimates; optional bias correction. Returns {blend, mean, spread}
  function blendEstimates(est, stats, correctBias = false) {
    let num = 0, den = 0;
    const vals = [];
    est.forEach((e, i) => {
      if (!isFinite(e)) return;
      const v = correctBias ? e - stats.bias[i] : e;
      num += stats.weights[i] * v; den += stats.weights[i]; vals.push(v);
    });
    if (!vals.length) return { blend: NaN, mean: NaN, spread: NaN };
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const spread = Math.max(...vals) - Math.min(...vals);
    return { blend: num / den, mean, spread };
  }

  // Round 3: which converter (and unit scaling) maps the published quantity to the true IV?
  // rows: [{q: [q1, q2, q3], S, truth (decimal vol)}]. Returns candidates sorted by mean abs error (vol points).
  function fitConverters(rows, T) {
    const out = [];
    for (const [name, f] of Object.entries(CONVERTERS)) {
      for (const scale of [1, 0.01]) {
        let err = 0, n = 0;
        for (const r of rows) {
          if (!isFinite(r.truth)) continue;
          for (const q of r.q) {
            if (!isFinite(q)) continue;
            const v = f(q * scale, r.S, T);
            if (!isFinite(v)) { err = Infinity; continue; }
            err += Math.abs(v - r.truth) * 100; n += 1;
          }
        }
        if (n) out.push({ name, scale, mae: err / n, n });
      }
    }
    return out.sort((a, b) => a.mae - b.mae);
  }

  const api = { erf, N, phi, bs, impliedVol, CONVERTERS, analystStats, blendEstimates, fitConverters };
  if (typeof module !== "undefined") module.exports = api; else root.OM = api;
})(this);
