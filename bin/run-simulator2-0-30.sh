#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run-simulator-0-30.sh 2025-09-17               # single-model, default model path
#   run-simulator-0-30.sh 2025-09-17 --model       # single-model, default ./output/xgb_model.json
#   run-simulator-0-30.sh 2025-09-17 --model ./output/custom.json
#
# Notes:
# - This script is for the streaming simulator using a SINGLE model.
# - It limits simulation/placements to the last 30 minutes pre-off.
# - Adjust the Python module below if your entry point differs (ml.sim vs ml.sim2).
#
# Optional: load env (comment out if not needed)
# source "$(dirname "$0")/set-env-vars-prod.sh" 2>/dev/null || true
#
# Optional: change into your project directory
# cd /opt/BetfairBotML 2>/dev/null || true

DATE_ARG="${1:-}"
if [[ -z "$DATE_ARG" ]]; then
  echo "Usage: $0 YYYY-MM-DD [--model [PATH]]"
  exit 1
fi
shift || true

MODEL_PATH="./output/xgb_model.json"
if [[ "${1:-}" == "--model" ]]; then
  shift || true
  if [[ -n "${1:-}" && "${1}" != --* ]]; then
    MODEL_PATH="${1}"
    shift || true
  fi
fi

echo "Running streaming simulator (single-model) for date: ${DATE_ARG}"
echo "Model: ${MODEL_PATH}"

# ---- tweak these if desired ----
STAKE_CAP_MKT=50
STAKE_CAP_DAY=2000
MAX_EXPOSURE_DAY=5000
PREOFF_MINS=30
MIN_EDGE=0.02
KELLY=0.25
COMMISSION=0.05
TOP_N=1
SIDE=auto

STREAM_BUCKET=5
LATENCY_MS=300
COOLDOWN_SECS=60
PLACE_UNTIL=1

CURATED="/mnt/nvme/betfair-curated"
SPORT="horse-racing"
OUT_BETS="./output/bets.csv"
# --------------------------------

BASE_ARGS=(
  --stake-cap-market ${STAKE_CAP_MKT}
  --stake-cap-day ${STAKE_CAP_DAY}
  --max-exposure-day ${MAX_EXPOSURE_DAY}

  --days-before 0
  --curated "${CURATED}"
  --sport "${SPORT}"
  --date "${DATE_ARG}"

  --preoff-mins ${PREOFF_MINS}

  --min-edge ${MIN_EDGE}
  --kelly ${KELLY}
  --commission ${COMMISSION}
  --top-n-per-market ${TOP_N}
  --side ${SIDE}

  --bets-out "${OUT_BETS}"

  --stream-bucket-secs ${STREAM_BUCKET}
  --latency-ms ${LATENCY_MS}
  --cooldown-secs ${COOLDOWN_SECS}
  --place-until-mins ${PLACE_UNTIL}
)

# Choose the Python entry point for the streaming sim.
# If your single-model entry point is ml.sim2, change "ml.sim" to "ml.sim2" below.
PY_ENTRY="ml.sim2"

# Use local venv if present, otherwise fall back to system python
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

exec ${PY} -m ${PY_ENTRY} --model "${MODEL_PATH}" "${BASE_ARGS[@]}"
