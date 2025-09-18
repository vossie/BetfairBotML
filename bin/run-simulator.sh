#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bin/run-simulator.sh 2025-09-15            # dual-model default
#   bin/run-simulator.sh 2025-09-15 --model    # single-model default ./output/xgb_model.json
#   bin/run-simulator.sh 2025-09-15 --model ./output/custom.json

DATE_ARG="${1:-}"
if [[ -z "$DATE_ARG" ]]; then
  echo "Usage: $0 YYYY-MM-DD [--model [PATH]]"
  exit 1
fi
shift || true

MODE="dual"
MODEL_PATH=""

if [[ "${1:-}" == "--model" ]]; then
  MODE="single"
  shift || true
  MODEL_PATH="${1:-}"
  if [[ -z "$MODEL_PATH" || "$MODEL_PATH" == --* ]]; then
    MODEL_PATH="./output/xgb_model.json"
  else
    shift || true
  fi
fi

# Load environment
git pull
source "$(dirname "$0")/set-env-vars-prod.sh"

echo "Endpoint: ${AWS_ENDPOINT_URL:-<unset>}"
echo "Region:   ${AWS_REGION:-<unset>}"
echo "Key set:  $( [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && echo yes || echo no )"

cd /opt/BetfairBotML

BASE_ARGS=(
  --stake-cap-market 50
  --stake-cap-day 2000
  --max-exposure-day 5000
  --days-before 1
  --curated /mnt/nvme/betfair-curated
  --sport horse-racing
  --date "$DATE_ARG" --days 1
  --preoff-mins 180
  --min-edge 0.02
  --kelly 0.25
  --commission 0.05
  --top-n-per-market 1
  --side auto
  --bets-out ./output/bets.csv
)

if [[ "$MODE" == "single" ]]; then
  echo "Running simulator in SINGLE-model mode with: $MODEL_PATH"
  .venv/bin/python -m ml.sim --model "$MODEL_PATH" "${BASE_ARGS[@]}"
else
  echo "Running simulator in DUAL-model mode with: ./output/model_30.json and ./output/model_180.json"
  .venv/bin/python -m ml.sim \
    --model-30 ./output/model_30.json \
    --model-180 ./output/model_180.json \
    --gate-mins 45 \
    "${BASE_ARGS[@]}"
fi
