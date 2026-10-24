# Hurricane Season — trading comp workspace

**Running it in the competition: see [RUNBOOK.md](RUNBOOK.md).**

Setup: `.venv/` (Python 3.12) has `websockets numpy pandas scipy scikit-learn matplotlib certifi`.
Run everything with `.venv/bin/python`.

## analysis/ (run from `analysis/`)

| Script | What it does |
|---|---|
| `common.py` | Loads the research data into pandas tables (storms, seasons, obs, advisories, reports, impacts, prices). Cached in `analysis/cache/`; delete that folder to rebuild. |
| `fit_values.py` | Settlement value vs season damage / insured loss for every company. Writes `out/values.json`. |
| `exposure.py` | Wind model per site (peak wind, closure seconds); damage vs exposure; outage cost. Writes `out/exposure.pkl`. |
| `source_accuracy.py` | Accuracy of official advisories and wire rumors by age, confusion matrices, probability calibration. |
| `house_errors.py` | House consensus vs settlement at key moments, prelim vs final reports, ETF vs basket gap. |

### First findings (600 seasons)
- **ISLE = 60 − 38·D**, **AQUA = 35 − 20·D** (D = summed season damage_pct / 100). Exact.
- **HAVEN** = 75 → 21 linearly as season loss goes $400M → $1300M ($6.00/share per $100M).
- **REIN** = 90 until $1300M, then −$3.50/share per $100M (no exhaust point seen up to $1766M).
- **PORT, GRID, FERRY** = V0 − a·D − b·(closure time). Damage alone leaves sd 0.8–1.7; adding closure seconds
  from the wind model brings it to ~0.2. V0 ≈ 100 / 50 / 25.
- **RECON, MAT**: concave in damage (power ≈ 0.4–0.5), R² 0.999. RECON ≈ PORT + GRID damage; MAT all of them.
  **GEN** is driven by grid closure time.
- Wind model: wind ≈ intensity · (radius/d)^1.3, hurricane conditions above ≈1.03 (fitted from impact
  messages; rough, worth refining).
- Official advisories ~70% right at every age (look like independent draws: combine them). Wire ~40%,
  but 88% when it agrees with the latest official.
- ARCH trades ~$2 sd away from its basket (free create/redeem → arb).

| `damage_model.py` | Structural damage model: track -> site exposure (kernel on d/R) -> saturating site damage -> company damage. |
| `fit_engine.py` | Fits everything the engine needs into `engine/train/` (seasons 201-600) or `engine/all/` (`--all`). |

## Strategy (engine/, backtest/)

**Holdout rule:** seasons 1-200 (closest in time to the live season) are never used for fitting or tuning.

`engine/model.py` — `FairValue` follows the feed message by message and returns the expected settlement and sd:
- **Physics (fitted):** site damage = 1 − exp(−v_site · Σ I·g(d/R)·dt), cumulative over the season (saturation);
  site closed while d/R < 1.02·I^0.45; insured loss = 9·PORT + 6·ISLE + 2.5·AQUA + 7·GRID + 1.2·FERRY (exact).
- **Valuation:** settlement from season damage + closure + loss (residual sd 0.003–0.18 $).
- **Current storm:** damage so far from observed track; remainder from 100 analog library storms in a similar
  state at the same age, weighted by the advisory likelihood ratio, their future displacement applied to our storm.
  After the preliminary report: blend 0.75·model + 0.25·prelim. After the final report: exact.
- **Future storms:** whole storms sampled from the library on top of the current site damage; Monte Carlo 2000.

`backtest/fv_series.py` replays seasons through the engine (only already-published messages) every 10 s.
`backtest/backtest.py` trades against the house price series:
- edge = β[sym, round, phase]·(fair − house), β fitted on training (shrinks toward the house where it knows more)
- one limit order per step at V ∓ (margin + size_pen·|pos|/limit)·price, walking a book shaped like testnet books
  (storm-time spreads), filled at the house price 4 s later, book refills 1/3 per 10 s; positions held otherwise.

Holdout (seasons 1-200), per season: **mean +$13.8k, median +$16.1k, 85% of seasons profitable**.
Stress: cost ×2 +$10.8k · refill 0.1 +$11.6k · inventory skew 3% +$10.8k · 20 s fill latency +$8.7k.

## v2: competitor-aware strategy (backtest/sim.py) and faster live bot

`backtest/sim.py` re-runs everything at 2 s resolution with rival bots (as smart as us, `lag` s slower, taking
`share` of the liquidity first and moving the price by `push` of their gap), our fill latency, and the team rate
limit (orders per 2 s, biggest dollar edge first). Tuned on 201-600 under the base scenario; holdout 1-200:

| scenario (holdout, $/season) | tuned | lower_variance |
|---|---|---|
| no rivals | 18.5k | 15.5k |
| **base: rivals 4 s slower, 30%** | **15.0k** (median 16.3k, 90% win) | 11.5k (5th pct −2.1k) |
| rivals 2 s slower, 50% | 12.4k | 9.3k |
| rivals as fast, 50% | 6.5k | 4.7k |
| rivals faster, 70%, our latency 4 s | 4.3k | 3.6k |
| old bot (decides every 10 s), base | 8.2k | |

Tested and rejected (no holdout gain): official cone weighting of analogs; cross-symbol / bias / momentum signal models
(`backtest/signal_model.py`). Kept: per-phase β, plus a risk term (more edge required when the engine is uncertain).

`bot/replay_test.py` runs the real `LiveBot` against a fake exchange replaying research seasons (no network).
Holdout, no rivals: tuned mean 21.4k / median 22.3k / 94% win; lower_variance 18.1k, 5th pct −2.3k;
peak 165 actions/min (budget 170).

## Live bot

```
./run_bot.sh                                   # profile "tuned", full size
./run_bot.sh --profile lower_variance          # ~20% less mean, much smaller bad seasons
./run_bot.sh --dry-run --verbose               # log orders only
./run_bot.sh status | stop
```
- Cycle on every storm message (2 s) and immediately on reports/advisories; requests in parallel; orders ranked by
  dollar edge within a rolling budget (`--actions-per-min 170`, ≤ budget/30 per cycle); one action per order and a
  single cancel-all only if something rests.
- Round comes from the exchange clock (`elapsed_active_seconds`, 840 s per round), so it trades round 1 before the storm
  forms and refuses to trade when it joined after round 1 without saved history (`bot/state/`).
- `bot/overrides.json` (re-read every cycle): `"halt"`, `"scale"`, per symbol `{"mult","add","disable"}` for announcements.
- **ETF arbitrage bot** (`bot/etf_arb_bot.py`, `./run_bot.sh --with-arb`): trades only ETF-vs-basket gaps, then
  creates/redeems so it ends flat; shares a trade lock with the main bot (`bot/coord.py`); budgets 125 + 50 of the
  team's 180 actions/min. Margin 0.4% of basket value, chosen across competitive scenarios (`backtest/etf_arb.py`).
  Holdout arb alone: $2.9k/season with rivals 4 s slower, $0.7k if rivals are as fast, $0.6k if our fills take 4 s.
  Replay of both bots (100 holdout seasons, no rivals): main alone $22.8k; main without ETFs + arb $20.4k;
  **main keeping ETFs + arb $23.1k (default)**; `--arb-owns-etfs` gives the ETFs to the arb bot only.
- **Fallback:** `./run_bot.sh --v1` runs the previous bot (`bot/live_bot_v1.py`, polls every 5 s, no action budget),
  with parameters pinned in `bot/v1_strategy_params.json`. It shares `bot/state/` with v2, so switching mid-season
  keeps the season's history: `./run_bot.sh stop && ./run_bot.sh --v1 --background`.
