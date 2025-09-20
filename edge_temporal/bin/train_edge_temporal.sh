#!/usr/bin/env bash
set -euo pipefail

ASOF_DATE="${1:?need asof date (e.g. 2025-09-20)}"
CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

START_DATE="2025-09-05"   # fixed training start

echo "=== Edge Temporal Training (LOCAL) ==="
echo "Curated root:         $CURATED_ROOT"
echo "ASOF (arg to trainer):$ASOF_DATE"
echo "Start date (train):   $START_DATE"

python3 /opt/BetfairBotML/edge_temporal/ml/train_edge_temporal.py \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF_DATE" \
  --start-date "$START_DATE" \
  --valid-days "${VALID_DAYS:-3}" \
  --sport "${SPORT:-horse-racing}" \
  --device "${DEVICE:-cuda}" \
  --commission "${COMMISSION:-0.02}" \
  --bankroll-nom "${BANKROLL_NOM:-1000}" \
  --kelly-cap "${KELLY_CAP:-0.05}" \
  --kelly-floor "${KELLY_FLOOR:-0.0}"
