#!/bin/bash
# Usage: ./sync_betfair.sh YYYY-MM-DD

set -euo pipefail

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 YYYY-MM-DD"
  exit 1
fi

DATE="$1"
BUCKET="betfair-curated"
SPORT="horse-racing"
DEST_ROOT="/mnt/nvme"

DATASETS=(market_definitions orderbook_snapshots_5s results)

for p in "${DATASETS[@]}"; do
  SRC="pub/${BUCKET}/${p}/sport=${SPORT}/date=${DATE}/"
  DEST="${DEST_ROOT}/${BUCKET}/${p}/sport=${SPORT}/date=${DATE}/"

  echo "Preparing destination $DEST ..."
  mkdir -p "$DEST"
  chown -R "$USER:$USER" "$DEST"

  echo "Syncing $SRC -> $DEST"
  mc mirror --overwrite --preserve "$SRC" "$DEST"
done

echo "✅ Sync complete for $DATE"