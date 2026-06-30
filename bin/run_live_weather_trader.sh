#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ACTUAL_ROOT="$(pwd)"
EXPECTED_ROOT="${DAILYWEATHER_EXPECTED_ROOT:-$ACTUAL_ROOT}"
if [[ "$ACTUAL_ROOT" != "$EXPECTED_ROOT" ]]; then
  echo "Refusing to run live trader from unexpected workspace: $ACTUAL_ROOT" >&2
  echo "Expected: $EXPECTED_ROOT" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-.venv-live/bin/python}"
PYCACHE="${PYTHONPYCACHEPREFIX:-/tmp/codex-pycache}"
LEDGER="${DAILYWEATHER_LIVE_LEDGER:-work/data/weather_live_money_50.sqlite}"
LOG_DIR="${DAILYWEATHER_LIVE_LOG_DIR:-work/logs/live_money}"
PROFILE="${DAILYWEATHER_STRATEGY_PROFILE:-live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70}"
RUN_TESTS="${DAILYWEATHER_RUN_TESTS:-0}"
LIMIT="${DAILYWEATHER_DISCOVERY_LIMIT:-2000}"
DISCOVERY_PAGES="${DAILYWEATHER_DISCOVERY_PAGES:-20}"
MAX_RUNTIME_SECONDS="${DAILYWEATHER_MAX_RUNTIME_SECONDS:-3600}"
PROGRESS_EVERY="${DAILYWEATHER_PROGRESS_EVERY:-25}"

if [[ "$RUN_TESTS" == "1" ]]; then
  env PYTHONPYCACHEPREFIX="$PYCACHE" "$PYTHON_BIN" -m unittest discover -s tests
fi

env PYTHONPYCACHEPREFIX="$PYCACHE" "$PYTHON_BIN" -m weather_strategy.cli paper-run \
  --ledger "$LEDGER" \
  --strategy-profile "$PROFILE" \
  --limit "$LIMIT" \
  --discovery-request-limit 50 \
  --discovery-pages "$DISCOVERY_PAGES" \
  --max-runtime-seconds "$MAX_RUNTIME_SECONDS" \
  --progress-every "$PROGRESS_EVERY" \
  --same-day-entry-start-hour 11 \
  --same-day-entry-cutoff-hour 17 \
  --weights-file work/data/model_weights.json \
  --run-log-dir "$LOG_DIR" \
  --execution-mode live \
  --confirm-live \
  --live-env-file .env.local

env PYTHONPYCACHEPREFIX="$PYCACHE" "$PYTHON_BIN" -m weather_strategy.cli report \
  --ledger "$LEDGER" \
  --bankroll-usd 50

env PYTHONPYCACHEPREFIX="$PYCACHE" "$PYTHON_BIN" -m weather_strategy.cli calibration \
  --ledger "$LEDGER"
