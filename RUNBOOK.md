# Competition run-book

Everything runs from the `trading_comp` folder. The commands below are copy-paste ready.

## 1. Before round 1 opens (about 5 minutes ahead)

1. **Use a stable connection.** Wired or strong Wi-Fi, laptop plugged in. A drop of more than a few seconds mid-round
   costs money; a long drop can make the bot miss a damage report (it recovers it from the next snapshot if it can).
2. **Check nothing is running and the account is flat:**
   ```bash
   ./run_bot.sh status
   ```
   It should say `not running`. If a bot from testing is running: `./run_bot.sh stop`.
3. **Check `bot/overrides.json`** is the neutral default (or reset it from the dashboard):
   ```json
   {"halt": false, "scale": 1.0, "symbols": {}}
   ```

## 2. Start

```bash
./run_bot.sh --with-arb --background
```
This starts the main bot (fair-value strategy, 125 actions/min), the ETF-arbitrage bot (50 actions/min), and the
dashboard. **Start it before round 1** — the bot refuses to trade a season it joined after round 1, because it would
not know the earlier damage.

Open the dashboard: **http://127.0.0.1:8765** (or `./run_bot.sh dashboard` to reopen it).

What you should see:
- Top pill **"⏸ Market closed"** until the open, then **"⚠ Not trading — waiting…"** for a moment, then **"✓ Trading"**.
- "updated 1.0 s ago" ticking. If it says **"✕ Bot not updating"**, the bot has stopped — see section 5.
- ETF arbitrage section: **"✓ running"** and Health **"✓ flat"**.

Alternatives:

| Situation | Command |
|---|---|
| Lower-risk version (~20% less on average, much smaller bad seasons) | `./run_bot.sh --with-arb --background --profile lower_variance` |
| Main bot only | `./run_bot.sh --background` |
| Watch without trading (logs the orders it would send) | `./run_bot.sh --with-arb --background --dry-run --arb-dry-run` |
| Half size | add `--scale 0.5` (or set Scale on the dashboard) |

## 3. During the season — what to watch

| Dashboard | Meaning | Action |
|---|---|---|
| ✓ Trading | normal | none |
| ⚠ Not trading — *reason* | the reason says why (waiting for the storm, missing history, …) | usually none; it resumes by itself |
| ! Network down | the bot is retrying | check the connection |
| ✕ Bot not updating | the bot process stopped or hung | section 5 |
| Expected PnL at settlement | the best estimate of the final score | — |
| Exchange marked PnL | marks at the exchange's prices; can swing far from the score mid-season | ignore swings |
| Arb Health "⚠ … pending" | a create/redeem was refused; the hedge is kept (it settles at the basket value) and retried | none unless it persists |

The log panel prints storm events, every order, and **`*** HEADLINE:` / `*** NEWS / ANNOUNCEMENT`** lines for news.

## 4. Announcements (news that changes a company's economics)

The model does not know about announcements; you do. The house takes about 45 s to fully reprice, so act fast:

1. Read the headline. Decide which company it affects and in which direction.
2. On the dashboard, in that symbol's row:
   - **Fair ×** (multiply) / **Fair +** (add) to move the bot's fair value, e.g. a company loses half its value →
     `Fair × 0.5`; a fine of $3/share → `Fair + -3`. ETFs follow their basket automatically.
   - or untick **on** to stop trading that symbol until you understand the news.
3. Press **Save overrides**. The bot applies it on its next cycle (≤ 2 s).
4. **Unsure? Untick the symbol** — not trading it is safer than trading on a wrong value.

Emergency stop for everything: **HALT trading** (click twice). It cancels open orders and stops new ones; positions are
kept. Press **RESUME trading** to continue. `ETF arb bot on` toggles only the arbitrage bot.

## 5. Problems

| Problem | Fix |
|---|---|
| Bot not updating / crashed | `./run_bot.sh status`. The supervisor restarts a crashed bot within 5 s and it replays its saved history. If nothing is running: `./run_bot.sh --with-arb --background` (same season: it keeps its history). |
| The v2 bot misbehaves (wrong orders, errors every cycle) | `./run_bot.sh stop && ./run_bot.sh --v1 --background` — the v1 fallback keeps the season's history. (No arb bot with v1.) |
| Network was down during a damage report | normally recovered from the next snapshot (log: "recovered a missed … damage report"). If the log says "final damage report of round N was not received", the bot stops trading to stay safe — tell whoever is on call; it resumes next season. |
| Rate-limit errors (HTTP 429) in the log | lower the budget: `./run_bot.sh stop`, then `./run_bot.sh --with-arb --background --arb-budget 30` |
| Dashboard not loading | `./run_bot.sh dashboard` |

## 6. Stop

```bash
./run_bot.sh stop          # stops both bots and the dashboard and cancels open orders; positions are kept
```
Positions settle automatically at the end of round 3 — there is nothing to close. Do not stop the bot between rounds;
it waits through the breaks by itself.

## Quick reference

```bash
./run_bot.sh --with-arb --background   # start (before round 1)
./run_bot.sh status                    # health + last log lines
./run_bot.sh dashboard                 # reopen http://127.0.0.1:8765
./run_bot.sh stop                      # stop everything, cancel orders
tail -f bot/logs/live-*.log            # raw main-bot log   (bot/logs/arb-*.log for the arb bot)
```
