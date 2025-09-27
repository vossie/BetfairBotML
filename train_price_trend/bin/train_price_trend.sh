#!/usr/bin/env bash
set -euo pipefail

# ---------- ASOF (default: yesterday) ----------
if [[ $# -ge 1 ]]; then
  ASOF="$1"
else
  if date -d "yesterday" +%Y-%m-%d >/dev/null 2>&1; then
    ASOF="$(date -d "yesterday" +%Y-%m-%d)"
  elif date -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    ASOF="$(date -v-1d +%Y-%m-%d)"
  elif command -v gdate >/dev/null 2>&1; then
    ASOF="$(gdate -d "yesterday" +%Y-%m-%d)"
  else
    echo "ERROR: cannot compute yesterday. Pass YYYY-MM-DD." >&2
    exit 1
  fi
fi

CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

# ---------- Paths ----------
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$BIN_DIR/.." && pwd)"

# ---------- Defaults / env knobs ----------
START_DATE="${START_DATE:-2025-09-05}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"

DEVICE="${DEVICE:-cuda}"        # XGBoost training device
SIM_DEVICE="${SIM_DEVICE:-cpu}" # Simulator device (default cpu to avoid device mismatch warnings)

HORIZON_SECS="${HORIZON_SECS:-120}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"

# Simulation/trading knobs (NOT used by the trainer)
EV_MODE="${EV_MODE:-mtm}"
EDGE_THRESH="${EDGE_THRESH:-0.0}"
STAKE="${STAKE:-kelly}"
KELLY_CAP="${KELLY_CAP:-0.02}"
KELLY_FLOOR="${KELLY_FLOOR:-0.001}"
BANKROLL_NOM="${BANKROLL_NOM:-5000}"

# EV rescaling + liquidity strictness (sim only)
EV_SCALE="${EV_SCALE:-1.0}"
REQUIRE_BOOK="${REQUIRE_BOOK:-0}"
MIN_FILL_FRAC="${MIN_FILL_FRAC:-0.0}"

# Odds band (optional)
ODDS_MIN="${ODDS_MIN:-}"
ODDS_MAX="${ODDS_MAX:-}"

# Liquidity enforcement
ENFORCE_LIQUIDITY="${ENFORCE_LIQUIDITY:-0}"
LIQUIDITY_LEVELS="${LIQUIDITY_LEVELS:-1}"

OUTDIR="${OUTDIR:-$BASE_DIR/output}"
mkdir -p "$OUTDIR" "$OUTDIR/stream"

# ---------- Training header ----------
echo "=== Price Trend: TRAIN ==="
echo "Curated root:    $CURATED_ROOT"
echo "ASOF:            $ASOF"
echo "Start date:      $START_DATE"
echo "Valid days:      $VALID_DAYS"
echo "Horizon (secs):  $HORIZON_SECS"
echo "Pre-off max (m): $PREOFF_MAX"
echo "Commission:      $COMMISSION"
echo "XGBoost device:  $DEVICE"
echo "Model dir:       $OUTDIR/models"

# ---------- Train (only training-supported args) ----------
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
  --output-dir "$OUTDIR"

# ---------- Simulation header ----------
echo "=== Price Trend: SIMULATE ==="
echo "EV threshold:    $EDGE_THRESH per £1"
echo "EV mode:         $EV_MODE    (scale=$EV_SCALE, cap not shown)"
if [[ -n "$ODDS_MIN" || -n "$ODDS_MAX" ]]; then
  echo "Odds band:       ${ODDS_MIN:--} .. ${ODDS_MAX:--}"
else
  echo "Odds band:       - .. -"
fi
echo "Stake mode:      $STAKE (cap=$KELLY_CAP floor=$KELLY_FLOOR)  bankroll=$BANKROLL_NOM"
echo "Liquidity:       enforce=$ENFORCE_LIQUIDITY levels=$LIQUIDITY_LEVELS require_book=$REQUIRE_BOOK min_fill_frac=$MIN_FILL_FRAC"
echo "Sim device:      $SIM_DEVICE"
echo "Output dir:      $OUTDIR/stream"

# ---------- Simulate (all trading knobs forwarded here) ----------
SIM_ARGS=(
  --curated "$CURATED_ROOT"
  --asof "$ASOF"
  --start-date "$START_DATE"
  --valid-days "$VALID_DAYS"
  --sport "$SPORT"
  --horizon-secs "$HORIZON_SECS"
  --preoff-max "$PREOFF_MAX"
  --commission "$COMMISSION"

  --edge-thresh "$EDGE_THRESH"
  --stake-mode "$STAKE"
  --kelly-cap "$KELLY_CAP"
  --kelly-floor "$KELLY_FLOOR"
  --bankroll-nom "$BANKROLL_NOM"

  --ev-mode "$EV_MODE"
  --ev-scale "$EV_SCALE"
  --min-fill-frac "$MIN_FILL_FRAC"

  --device "$SIM_DEVICE"
  --output-dir "$OUTDIR/stream"
)

# Optional odds band
if [[ -n "$ODDS_MIN" ]]; then SIM_ARGS+=( --odds-min "$ODDS_MIN" ); fi
if [[ -n "$ODDS_MAX" ]]; then SIM_ARGS+=( --odds-max "$ODDS_MAX" ); fi

# Liquidity flags
if [[ "$ENFORCE_LIQUIDITY" == "1" ]]; then
  SIM_ARGS+=( --enforce-liquidity --liquidity-levels "$LIQUIDITY_LEVELS" )
fi
if [[ "$REQUIRE_BOOK" == "1" ]]; then
  SIM_ARGS+=( --require-book )
fi

python3 "$BASE_DIR/ml/simulate_stream.py" "${SIM_ARGS[@]}"
