#!/usr/bin/env bash
set -euo pipefail

# Usage: run-train-xgb-country.sh 2025-09-15 [extra xgb args...]
if [ $# -lt 1 ]; then
  echo "Usage: $0 <DATE: YYYY-MM-DD> [extra args...]"
  exit 1
fi

TARGET_DATE="$1"
shift || true

# Same base as your other trainer (inclusive range)
BASE_DATE="2025-09-05"
DAYS=$(( ( $(date -d "$TARGET_DATE" +%s) - $(date -d "$BASE_DATE" +%s) ) / 86400 + 1 ))
if [ $DAYS -le 0 ]; then
  echo "Error: target date $TARGET_DATE must be on or after $BASE_DATE"
  exit 1
fi

cd /opt/BetfairBotML || exit 1

# Env (OK if missing)
if [ -f "$(dirname "$0")/set-env-vars-prod.sh" ]; then
  # shellcheck disable=SC1090
  source "$(dirname "$0")/set-env-vars-prod.sh"
fi

git pull

echo "▶ Training for date=$TARGET_DATE covering $DAYS days from $BASE_DATE"
echo "   curated=${CURATED_ROOT:-/mnt/nvme/betfair-curated} sport=${SPORT:-horse-racing}"

# Python from project venv
PY="/opt/BetfairBotML/.venv/bin/python"

exec "$PY" -m ml.train_xgb_country \
  --curated "${CURATED_ROOT:-/mnt/nvme/betfair-curated}" \
  --sport   "${SPORT:-horse-racing}" \
  --date "$TARGET_DATE" \
  --days "$DAYS" \
  --chunk-days "${CHUNK_DAYS:-2}" \
  --preoff-mins "${PREOFF_MINS:-30}" \
  --batch-markets "${BATCH_MARKETS:-100}" \
  --downsample-secs "${DOWNSAMPLE_SECS:-0}" \
  --device "${DEVICE:-auto}" \
  "$@"
