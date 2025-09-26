#!/usr/bin/env bash
set -euo pipefail

ASOF="${1:?usage: train_price_trend.sh YYYY-MM-DD}"
CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

START_DATE="${START_DATE:-2025-09-05}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"
DEVICE="${DEVICE:-cuda}"

HORIZON_SECS="${HORIZON_SECS:-120}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"

# Simulation/backtest controls (used only by simulate_stream.py)
EDGE_THRESH="${EDGE_THRESH:-0.0}"
STAKE="${STAKE:-kelly}"
KELLY_CAP="${KELLY_CAP:-0.02}"
KELLY_FLOOR="${KELLY_FLOOR:-0.001}"
BANKROLL_NOM="${BANKROLL_NOM:-5000}"

# Control whether to run the post-training simulation (1=yes, 0=no)
RUN_SIM="${RUN_SIM:-1}"

OUTDIR="${OUTDIR:-/opt/BetfairBotML/train_price_trend/output}"
MODEL_PATH="$OUTDIR/models/xgb_trend_reg.json"

echo "=== Price Trend Training ==="
echo "Curated root:    $CURATED_ROOT"
echo "ASOF:            $ASOF"
echo "Start date:      $START_DATE"
echo "Valid days:      $VALID_DAYS"
echo "Horizon (secs):  $HORIZON_SECS"
echo "Pre-off max (m): $PREOFF_MAX"
echo "Stake mode:      $STAKE (cap=$KELLY_CAP floor=$KELLY_FLOOR)"

# ---- TRAIN (no --edge-thresh here) ----
python3 /opt/BetfairBotML/train_price_trend/ml/train_price_trend.py \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --device "$DEVICE" \
  --horizon-secs "$HORIZON_SECS" \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --stake-mode "$STAKE" \
  --kelly-cap "$KELLY_CAP" \
  --kelly-floor "$KELLY_FLOOR" \
  --bankroll-nom "$BANKROLL_NOM" \
  --output-dir "$OUTDIR"

# ---- OPTIONAL: SIMULATE / BACKTEST ----
if [[ "$RUN_SIM" == "1" ]]; then
  echo "=== Streaming Backtest ==="
  echo "EV threshold:    $EDGE_THRESH per £1"
  python3 /opt/BetfairBotML/train_price_trend/ml/simulate_stream.py \
    --curated "$CURATED_ROOT" \
    --asof "$ASOF" \
    --start-date "$START_DATE" \
    --valid-days "$VALID_DAYS" \
    --sport "$SPORT" \
    --horizon-secs "$HORIZON_SECS" \
    --preoff-max "$PREOFF_MAX" \
    --commission "$COMMISSION" \
    --edge-thresh "$EDGE_THRESH" \
    --stake-mode "$STAKE" \
    --kelly-cap "$KELLY_CAP" \
    --kelly-floor "$KELLY_FLOOR" \
    --bankroll-nom "$BANKROLL_NOM" \
    --device "$DEVICE" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTDIR/stream"
fi
