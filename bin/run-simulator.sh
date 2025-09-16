#!/bin/bash
# run.sh
cd /opt/BetfairBotML || exit 1

# Load environment
source "$(dirname "$0")/set-env-vars-prod.sh"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <DATE: YYYY-MM-DD>"
  exit 1
fi

TARGET_DATE="$1"

.venv/bin/python -m ml.sim \
  --model-30 ../output/model_30.json \
  --model-180 ../output/model_180.json \
  --gate-mins 45 \
  --curated /mnt/nvme/betfair-curated \
  --sport horse-racing \
  --date "$TARGET_DATE" --days 10 \
  --preoff-mins 180 \
  --min-edge 0.02 \
  --kelly 0.25 \
  --commission 0.02 \
  --top-n-per-market 1 \
  --side auto \
  --bets-out bets_dual.csv
