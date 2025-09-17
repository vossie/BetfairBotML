#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bin/run-simulator-country.sh 2025-09-15
#   bin/run-simulator-country.sh 2025-09-15 --model ./output/xgb_country.json --meta ./output/xgb_country_meta.json
#   bin/run-simulator-country.sh 2025-09-15 --days-before 7 --country GB
#
# Notes:
# - Runs the country-aware simulator (ml.sim_country) to produce overall and per-country PnL.
# - By default it looks for ./output/xgb_country.json and ./output/xgb_country_meta.json.

DATE_ARG="${1:-}"
if [[ -z "$DATE_ARG" ]]; then
  echo "Usage: $0 YYYY-MM-DD [--model PATH] [--meta PATH] [--days-before N] [--country CC]"
  exit 1
fi
shift || true

MODEL_PATH="./output/xgb_model_country.json"
META_PATH="./output/xgb_country_meta.features.txt"
DAYS_BEFORE="7"
COUNTRY_FILTER=""

# Parse simple flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      shift || true
      MODEL_PATH="${1:-$MODEL_PATH}"
      ;;
    --meta)
      shift || true
      META_PATH="${1:-$META_PATH}"
      ;;
    --days-before)
      shift || true
      DAYS_BEFORE="${1:-$DAYS_BEFORE}"
      ;;
    --country)
      shift || true
      COUNTRY_FILTER="${1:-}"
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
  shift || true
done

# Load environment and position in repo (mirrors bin/run-simulator.sh approach)
git pull || true
source "$(dirname "$0")/set-env-vars-prod.sh"

echo "Endpoint: ${AWS_ENDPOINT_URL:-<unset>}"
echo "Region:   ${AWS_REGION:-<unset>}"
echo "Key set:  $( [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && echo yes || echo no )"

cd /opt/BetfairBotML

BASE_ARGS=(
  --model "$MODEL_PATH"
  --meta "$META_PATH"
  --curated /mnt/nvme/betfair-curated
  --sport horse-racing
  --date "$DATE_ARG"
  --days-before "$DAYS_BEFORE"
  --preoff-mins 180
  --min-edge 0.02
  --kelly 0.25
  --commission 0.02
  --top-n-per-market 1
  --side auto
  --bets-out ./output/bets_country.csv
  --pnl-by-country-out ./output/pnl_by_country.csv
)

if [[ -n "$COUNTRY_FILTER" ]]; then
  BASE_ARGS+=( --country-filter "$COUNTRY_FILTER" )
fi

echo "Running country simulator with:"
printf '  %q\n' "${BASE_ARGS[@]}"

.venv/bin/python -m ml.sim_country "${BASE_ARGS[@]}"
