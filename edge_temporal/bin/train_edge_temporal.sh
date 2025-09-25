#!/usr/bin/env bash
set -euo pipefail

# Usage (example):
#   CURATED_ROOT=/mnt/nvme/betfair-curated \
#   EDGE_THRESH=0.024 PER_MARKET_TOPK=1 LTP_MIN=2.2 LTP_MAX=3.8 PREOFF_MINS=20 \
#   /opt/BetfairBotML/edge_temporal/bin/train_edge_temporal.sh 2025-09-24

ASOF_DATE="${1:?need ASOF date, e.g. 2025-09-24}"
CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

START_DATE="${START_DATE:-2025-09-05}"
VALID_DAYS="${VALID_DAYS:-7}"
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
  --kelly-floor "$KELLY_FLOOR" \
  ${FIT_CALIB:+--fit-calib} \
  ${CALIB_OUT:+--calib-out "$CALIB_OUT"}
