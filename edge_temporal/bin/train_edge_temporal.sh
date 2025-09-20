#!/usr/bin/env bash
set -euo pipefail

ASOF_DATE="$1"

python3 /opt/BetfairBotML/edge_temporal/ml/train_edge_temporal.py \
  --curated "${CURATED_ROOT:-/mnt/nvme/betfair-curated}" \
  --asof "$ASOF_DATE" \
  --train-days "${TRAIN_DAYS:-12}" \
  --valid-days "${VALID_DAYS:-3}" \
  --edge-thresh "${EDGE_THRESH:-0.015}" \
  --pm-cutoff "${PM_CUTOFF:-0.65}" \
  --per-market-topk "${PER_MARKET_TOPK:-1}" \
  --stake "${STAKE:-flat}" \
  --kelly-cap "${KELLY_CAP:-0.05}" \
  --kelly-floor "${KELLY_FLOOR:-0.0}" \
  --bankroll-nom "${BANKROLL_NOM:-1000}" \
  --ltp-min "${LTP_MIN:-1.5}" \
  --ltp-max "${LTP_MAX:-5.0}" \
  --device "cuda" \
  --sport "${SPORT:-horse-racing}" \
  --preoff-mins "${PREOFF_MINS:-30}" \
  --commission "${COMMISSION:-0.02}"
