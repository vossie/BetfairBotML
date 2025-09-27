#!/usr/bin/env bash
set -euo pipefail

# ---------- ASOF date (defaults to yesterday) ----------
if [[ $# -ge 1 ]]; then
  ASOF="$1"
else
  # Linux (GNU date)
  if date -d "yesterday" +%Y-%m-%d >/dev/null 2>&1; then
    ASOF="$(date -d "yesterday" +%Y-%m-%d)"
  # macOS / BSD date
  elif date -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    ASOF="$(date -v-1d +%Y-%m-%d)"
  # GNU coreutils 'gdate' (sometimes on mac via brew)
  elif command -v gdate >/dev/null 2>&1; then
    ASOF="$(gdate -d "yesterday" +%Y-%m-%d)"
  else
    echo "ERROR: can't compute 'yesterday' (need GNU date or BSD date). Provide ASOF explicitly." >&2
    exit 1
  fi
fi

CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$BIN_DIR/.." && pwd)"

# ---------- Defaults (override via env) ----------
START_DATE="${START_DATE:-2025-09-05}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"
DEVICE="${DEVICE:-cuda}"

HORIZON_SECS="${HORIZON_SECS:-120}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"

EDGE_THRESH="${EDGE_THRESH:-0.002}"
STAKE="${STAKE:-kelly}"
KELLY_CAP="${KELLY_CAP:-0.02}"
KELLY_FLOOR="${KELLY_FLOOR:-0.001}"
BANKROLL_NOM="${BANKROLL_NOM:-5000}"

EV_MODE="${EV_MODE:-mtm}"

# Optional odds band + liquidity (off by default)
ODDS_MIN="${ODDS_MIN:-}"
ODDS_MAX="${ODDS_MAX:-}"
ENFORCE_LIQUIDITY="${ENFORCE_LIQUIDITY:-0}"     # set to 1 to enable
LIQUIDITY_LEVELS="${LIQUIDITY_LEVELS:-1}"

OUTDIR="${OUTDIR:-$BASE_DIR/output}"

MODEL_PATH="$OUTDIR/models/xgb_trend_reg.json"

echo "=== Price Trend Training ==="
echo "Curated root:    $CURATED_ROOT"
echo "ASOF:            $ASOF"
echo "Start date:      $START_DATE"
echo "Valid days:      $VALID_DAYS"
echo "Horizon (secs):  $HORIZON_SECS"
echo "Pre-off max (m): $PREOFF_MAX"
echo "Stake mode:      $STAKE (cap=$KELLY_CAP floor=$KELLY_FLOOR)"
echo "EV mode:         $EV_MODE"
echo "Odds band:       ${ODDS_MIN:--} .. ${ODDS_MAX:--}"
echo "Liquidity:       enforce=${ENFORCE_LIQUIDITY} levels=${LIQUIDITY_LEVELS}"
echo "Device:          $DEVICE"
echo "Output dir:      $OUTDIR"

# ---------- Train (skip if model exists and FORCE_TRAIN!=1) ----------
if [[ "${FORCE_TRAIN:-0}" == "1" || ! -f "$MODEL_PATH" ]]; then
  python3 "$BASE_DIR/ml/train_price_trend.py" \
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
    --ev-mode "$EV_MODE" \
    --output-dir "$OUTDIR"
else
  echo "Model exists at $MODEL_PATH — skipping training (set FORCE_TRAIN=1 to retrain)."
fi

# ---------- Simulate (stream backtest) ----------
echo "=== Streaming Backtest ==="
echo "EV threshold:    $EDGE_THRESH per £1"

SIM_ARGS=(
  --curated "$CURATED_ROOT"
  --asof "$ASOF"
  --start-date "$START_DATE"
  --valid-days "$VALID_DAYS"
  --sport "$SPORT"
  --preoff-max "$PREOFF_MAX"
  --commission "$COMMISSION"
  --edge-thresh "$EDGE_THRESH"
  --stake-mode "$STAKE"
  --kelly-cap "$KELLY_CAP"
  --kelly-floor "$KELLY_FLOOR"
  --bankroll-nom "$BANKROLL_NOM"
  --ev-mode "$EV_MODE"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTDIR/stream"
  --device "$DEVICE"
  --horizon-secs "$HORIZON_SECS"
)

# Optional odds band
[[ -n "${ODDS_MIN}" ]] && SIM_ARGS+=( --odds-min "$ODDS_MIN" )
[[ -n "${ODDS_MAX}" ]] && SIM_ARGS+=( --odds-max "$ODDS_MAX" )

# Optional liquidity enforcement
if [[ "$ENFORCE_LIQUIDITY" == "1" ]]; then
  SIM_ARGS+=( --enforce-liquidity --liquidity-levels "$LIQUIDITY_LEVELS" )
fi

python3 "$BASE_DIR/ml/simulate_stream.py" "${SIM_ARGS[@]}"
