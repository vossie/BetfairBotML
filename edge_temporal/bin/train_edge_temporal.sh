#!/usr/bin/env bash
set -euo pipefail

# Usage: CURATED_ROOT=/path/to/curated /opt/BetfairBotML/edge_temporal/bin/train_edge_temporal.sh YYYY-MM-DD
ASOF_DATE="${1:?need ASOF date, e.g. 2025-09-21}"
CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"
START_DATE="${START_DATE:-2025-09-05}"

VALID_DAYS="${VALID_DAYS:-7}"            # default rolling 7-day validation
SPORT="${SPORT:-horse-racing}"
DEVICE="${DEVICE:-cuda}"
COMMISSION="${COMMISSION:-0.02}"
BANKROLL_NOM="${BANKROLL_NOM:-5000}"
KELLY_CAP="${KELLY_CAP:-0.05}"
KELLY_FLOOR="${KELLY_FLOOR:-0.002}"

echo "=== Edge Temporal Training (LOCAL) ==="
echo "Curated root:         $CURATED_ROOT"
echo "ASOF (arg to trainer):$ASOF_DATE"
echo "Start date (train):   $START_DATE"

python3 /opt/BetfairBotML/edge_temporal/ml/train_edge_temporal.py \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF_DATE" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --device "$DEVICE" \
  --commission "$COMMISSION" \
  --bankroll-nom "$BANKROLL_NOM" \
  --kelly-cap "$KELLY_CAP" \
  --kelly-floor "$KELLY_FLOOR"
