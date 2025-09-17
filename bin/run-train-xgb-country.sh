#!/bin/bash
# run.sh okk
cd /opt/BetfairBotML || exit

if [ $# -lt 1 ]; then
  echo "Usage: $0 <DATE: YYYY-MM-DD>"
  exit 1
fi

TARGET_DATE="$1"
BASE_DATE="2025-09-05"

# Calculate number of days between BASE_DATE and TARGET_DATE
DAYS=$(( ( $(date -d "$TARGET_DATE" +%s) - $(date -d "$BASE_DATE" +%s) ) / 86400 + 1 ))

if [ $DAYS -le 0 ]; then
  echo "Error: target date $TARGET_DATE must be on or after $BASE_DATE"
  exit 1
fi

cd /opt/BetfairBotML || exit 1

# Load environment
source "$(dirname "$0")/set-env-vars-prod.sh"

git pull

echo "▶ Training for date=$TARGET_DATE covering $DAYS days from $BASE_DATE"

.venv/bin/python -m ml.train_xgb_country \
  --curated /opt/BetfairBotML/betfair-curated \
  --sport horse-racing \
  --date "$TARGET_DATE" \
  --days "$DAYS" \
  --preoff-mins 30 --batch-markets 100 --downsample-secs 0 --chunk-days 2 \
  --device auto --n-estimators 3000 --learning-rate 0.02 --early-stopping-rounds 100


