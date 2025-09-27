#!/usr/bin/env bash
set -euo pipefail

ASOF="${1:?usage: sweep_price_trend.sh YYYY-MM-DD}"
CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$BIN_DIR/.." && pwd)"
ML_DIR="$BASE_DIR/ml"
OUTBASE="${OUTDIR:-$BASE_DIR/output/stream}"

# Defaults aligned with your train/sim setup
START_DATE="${START_DATE:-2025-09-05}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"
DEVICE="${DEVICE:-cuda}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"
BANKROLL_NOM="${BANKROLL_NOM:-5000}"
KELLY_CAP="${KELLY_CAP:-0.01}"
KELLY_FLOOR="${KELLY_FLOOR:-0.001}"
EV_MODE="${EV_MODE:-mtm}"

# Grids (comma-separated)
EDGE_THRESH_GRID="${EDGE_THRESH_GRID:-0.0005,0.001,0.002,0.003,0.005}"
STAKE_MODES_GRID="${STAKE_MODES_GRID:-flat,kelly}"
ODDS_BANDS_GRID="${ODDS_BANDS_GRID:-none,2.2:3.6,1.5:5.0}"

TAG="${TAG:-grid1}"
MODEL_PATH="${MODEL_PATH:-$BASE_DIR/output/models/xgb_trend_reg.json}"

echo "=== Trend Sweep ==="
echo "ASOF:            $ASOF"
echo "Start date:      $START_DATE"
echo "Valid days:      $VALID_DAYS"
echo "Pre-off max (m): $PREOFF_MAX"
echo "Device:          $DEVICE"
echo "EV mode:         $EV_MODE"
echo "Grids: EDGE=[$EDGE_THRESH_GRID]  STAKE=[$STAKE_MODES_GRID]  ODDS=[$ODDS_BANDS_GRID]"
echo "Output base:     $OUTBASE"

python3 "$ML_DIR/sim_sweep.py" \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --device "$DEVICE" \
  --ev-mode "$EV_MODE" \
  --model-path "$MODEL_PATH" \
  --base-output-dir "$OUTBASE" \
  --bankroll-nom "$BANKROLL_NOM" \
  --kelly-cap "$KELLY_CAP" \
  --kelly-floor "$KELLY_FLOOR" \
  --edge-thresh "$EDGE_THRESH_GRID" \
  --stake-modes "$STAKE_MODES_GRID" \
  --odds-bands "$ODDS_BANDS_GRID" \
  --tag "$TAG"
