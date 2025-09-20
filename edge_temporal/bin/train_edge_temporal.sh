#!/bin/bash
set -euo pipefail

echo "=== Edge Temporal Training (LOCAL) ==="

CURATED_ROOT="/mnt/nvme/betfair-curated"
SPORT="${SPORT:-horse-racing}"
ASOF_ARG="${1:-$(date -d 'yesterday' +%F)}"
TRAIN_DAYS="${TRAIN_DAYS:-12}"
VALID_DAYS="${VALID_DAYS:-2}"

export TZ="Europe/London"
TODAY="$(date +%F)"
ASOF="$ASOF_ARG"

VAL_START="$(date -d "${ASOF} -$((VALID_DAYS-1)) day" +%F)"
TRAIN_END="$(date -d "${VAL_START} -1 day" +%F)"
TRAIN_START="$(date -d "${TRAIN_END} -$((TRAIN_DAYS-1)) day" +%F)"

echo "Curated root:         $CURATED_ROOT"
echo "Today:                $TODAY"
echo "ASOF (arg to trainer):$ASOF   # validation excludes today"
echo "Validation window:    $VAL_START .. $ASOF"
echo "Training window:      $TRAIN_START .. $TRAIN_END"
echo "Computed TRAIN_DAYS:  $TRAIN_DAYS"
echo "Computed VALID_DAYS:  $VALID_DAYS"

ML_PY="/opt/BetfairBotML/edge_temporal/ml/train_edge_temporal.py"

python3 "$ML_PY" \
  --curated "$CURATED_ROOT" \
  --sport "$SPORT" \
  --asof "$ASOF" \
  --train-days "$TRAIN_DAYS" \
  --valid-days "$VALID_DAYS"
