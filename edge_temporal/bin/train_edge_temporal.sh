#!/usr/bin/env bash
set -euo pipefail

ASOF_DATE="${1:?need asof date (e.g. 2025-09-20)}"
CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"
START_DATE="${START_DATE:-2025-09-05}"

python3 /opt/BetfairBotML/edge_temporal/ml/train_edge_temporal.py \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF_DATE" \
  --start-date "$START_DATE" \
  --valid-days "${VALID_DAYS:-3}" \
  --sport "${SPORT:-horse-racing}" \
  --device "${DEVICE:-cuda}" \
  --commission "${COMMISSION:-0.02}" \
  --bankroll-nom "${BANKROLL_NOM:-5000}" \
  --kelly-cap "${KELLY_CAP:-0.05}" \
  --kelly-floor "${KELLY_FLOOR:-0.002}" \
  --preoff-mins "${PREOFF_MINS:-30}" \
  --pm-cutoff "${PM_CUTOFF:-0.65}" \
  --edge-thresh "${EDGE_THRESH:-0.015}" \
  --per-market-topk "${PER_MARKET_TOPK:-1}" \
  --ltp-min "${LTP_MIN:-1.5}" \
  --ltp-max "${LTP_MAX:-5.0}"
