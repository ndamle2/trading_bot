#!/usr/bin/env bash
# Run the hurricane live bot: keeps the Mac awake, restarts the bot if it crashes, logs to bot/logs/.
#
#   ./run_bot.sh                 start at full size (foreground; Ctrl-C stops it)
#   ./run_bot.sh --scale 0.25    any live_bot.py options pass through (--dry-run, --scale, --step, --tag ...)
#   ./run_bot.sh --background    start detached, then return (log: bot/logs/live-<time>.log)
#   ./run_bot.sh status          is it running? last log lines
#   ./run_bot.sh stop            stop it (open orders are cancelled first)
#   ./run_bot.sh --v1 ...        run the old fallback bot (bot/live_bot_v1.py) instead of the v2 bot
#   ./run_bot.sh dashboard       (re)start only the dashboard: http://127.0.0.1:8765
#   ./run_bot.sh --with-arb      also run the ETF-arbitrage bot (bot/etf_arb_bot.py). Budgets split the team's
#                                180 actions/min: main 125, arb 50 (--arb-budget N changes the arb share).
#                                The main bot keeps trading ARCH/REBLD on fair value (best in the replay backtest);
#                                add --arb-owns-etfs to give the ETFs to the arb bot only (main gets --skip-symbols).
#                                --arb-dry-run runs the arb bot in dry-run mode.
# The dashboard (bot/dashboard.py) starts automatically with the bot and is stopped by `stop`.
#
# Needs trading_comp/.env with HURRICANE_API_KEY=... and the venv in trading_comp/.venv.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BOT="$ROOT/bot"
PY="$ROOT/.venv/bin/python"
PIDFILE="$BOT/state/run_bot.pid"
DASH_PORT="${DASH_PORT:-8765}"
mkdir -p "$BOT/logs" "$BOT/state"

running_pid() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then cat "$PIDFILE"; return 0; fi
  pgrep -f "live_bot(_v1)?\.py|etf_arb_bot\.py" | head -1 || return 1
}

start_dashboard() {
  pkill -f "dashboard.py --port $DASH_PORT" 2>/dev/null || true
  (cd "$BOT" && nohup "$PY" dashboard.py --port "$DASH_PORT" >>"$BOT/logs/dashboard.log" 2>&1 &)
  sleep 0.5
  echo "dashboard: http://127.0.0.1:$DASH_PORT"
}

cancel_orders() {
  (cd "$BOT" && "$PY" -c "from client import Exchange; print('cancel all:', Exchange().cancel_all())") || true
}

case "${1:-}" in
  dashboard)
    start_dashboard
    command -v open >/dev/null && open "http://127.0.0.1:$DASH_PORT"
    exit 0 ;;
  stop)
    pkill -f "dashboard.py --port $DASH_PORT" 2>/dev/null && echo "dashboard stopped" || true
    if pid=$(running_pid); then
      echo "stopping bot (pid $pid)"
      kill "$pid" 2>/dev/null || true
      [[ -f "$PIDFILE.arb" ]] && kill "$(cat "$PIDFILE.arb")" 2>/dev/null || true
      pkill -f "live_bot(_v1)?\.py|etf_arb_bot\.py" 2>/dev/null || true
      rm -f "$PIDFILE.arb"
      rm -f "$PIDFILE"
      cancel_orders
    else
      echo "bot is not running"
    fi
    exit 0 ;;
  status)
    if pid=$(running_pid); then echo "running (pid $pid)"; else echo "not running"; fi
    pgrep -f "etf_arb_bot\.py" >/dev/null && echo "ETF arb bot: running" || echo "ETF arb bot: not running"
    latest=$(ls -t "$BOT"/logs/live-*.log "$BOT"/logs/live_season.log 2>/dev/null | head -1 || true)
    [[ -n "$latest" ]] && { echo "--- $latest"; tail -n 15 "$latest" | cut -c1-200; }
    exit 0 ;;
esac

# preflight
[[ -x "$PY" ]] || { echo "missing venv: $PY"; exit 1; }
[[ -f "$ROOT/.env" ]] || [[ -n "${HURRICANE_API_KEY:-}" ]] || { echo "missing $ROOT/.env with HURRICANE_API_KEY"; exit 1; }
[[ -f "$ROOT/backtest/out/live_params.json" ]] || { echo "missing backtest/out/live_params.json (run backtest/sim.py)"; exit 1; }
if pid=$(running_pid); then echo "bot already running (pid $pid). Use './run_bot.sh stop' first."; exit 1; fi

BACKGROUND=0
SCRIPT=live_bot.py
WITH_ARB=0
ARB_BUDGET=50
ARB_OWNS_ETFS=0
ARB_ARGS=()
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --background) BACKGROUND=1 ;;
    --v1) SCRIPT=live_bot_v1.py ;;
    --with-arb) WITH_ARB=1 ;;
    --arb-budget) ARB_BUDGET="$2"; shift ;;
    --arb-owns-etfs) ARB_OWNS_ETFS=1 ;;
    --main-trades-etfs) ARB_OWNS_ETFS=0 ;;
    --arb-dry-run) ARB_ARGS+=(--dry-run) ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
[[ $SCRIPT == live_bot_v1.py ]] && echo "using the v1 fallback bot"
if [[ $WITH_ARB -eq 1 ]]; then
  [[ $SCRIPT == live_bot.py ]] || { echo "--with-arb needs the v2 bot (it shares the trade lock)"; exit 1; }
  [[ " ${ARGS[*]:-} " == *" --actions-per-min "* ]] || ARGS+=(--actions-per-min "$((175 - ARB_BUDGET))")
  [[ $ARB_OWNS_ETFS -eq 1 ]] && ARGS+=(--skip-symbols ARCH,REBLD)
  ARB_ARGS+=(--actions-per-min "$ARB_BUDGET")
  echo "with ETF arb bot: main ${ARGS[*]}; arb ${ARB_ARGS[*]}"
fi
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$BOT/logs/live-$STAMP.log"
ARB_LOG="$BOT/logs/arb-$STAMP.log"

(cd "$BOT" && "$PY" -c "from client import Exchange; t=Exchange().time(); print('exchange ok:', t['state'], t['fictional_time'])") \
  || { echo "cannot reach the exchange (check network / key)"; exit 1; }

supervise() {
  # supervise SCRIPT ARG...: restart on crash; stop on Ctrl-C / kill
  local script="$1"; shift
  trap 'echo "$(date +%T) supervisor stopping"; kill "${child:-}" 2>/dev/null; exit 0' INT TERM
  while true; do
    echo "$(date +%T) starting $script $*"
    (cd "$BOT" && exec caffeinate -i "$PY" -u "$script" "$@") &
    child=$!
    wait "$child" && code=0 || code=$?
    echo "$(date +%T) $script exited with code $code; restarting in 5 s (it replays its saved state)"
    sleep 5
  done
}

start_dashboard
if [[ $WITH_ARB -eq 1 ]]; then
  supervise etf_arb_bot.py ${ARB_ARGS[@]+"${ARB_ARGS[@]}"} >>"$ARB_LOG" 2>&1 &
  echo $! >"$PIDFILE.arb"
  disown
  echo "ETF arb bot started (log: $ARB_LOG)"
fi
if [[ $BACKGROUND -eq 1 ]]; then
  supervise "$SCRIPT" ${ARGS[@]+"${ARGS[@]}"} >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  disown
  echo "bot started in the background (pid $(cat "$PIDFILE")), log: $LOG"
  echo "  ./run_bot.sh status    ./run_bot.sh stop    tail -f $LOG"
else
  echo $$ >"$PIDFILE"
  [[ $WITH_ARB -eq 1 ]] && trap 'kill "$(cat "$PIDFILE.arb")" 2>/dev/null; pkill -f etf_arb_bot.py; exit 0' INT TERM
  echo "log: $LOG (Ctrl-C to stop)"
  supervise "$SCRIPT" ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee -a "$LOG"
fi
