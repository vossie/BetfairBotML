#!/usr/bin/env bash
set -euo pipefail

# ------------- helpers -------------
_yesterday() {
  if date -d "yesterday" +%Y-%m-%d >/dev/null 2>&1; then
    date -d "yesterday" +%Y-%m-%d
  elif date -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    date -v-1d +%Y-%m-%d
  elif command -v gdate >/dev/null 2>&1; then
    gdate -d "yesterday" +%Y-%m-%d
  else
    echo ""
  fi
}

_is_date() {
  [[ "${1:-}" =~ ^20[0-9]{2}-[01][0-9]-[0-3][0-9]$ ]]
}

# ------------- locate dirs -------------
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$BIN_DIR/.." && pwd)"

# ------------- parse args -------------
ASOF=""
EXTRA_SIM_ARGS=()

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    # no explicit date; pass-through starts after --
    shift
    EXTRA_SIM_ARGS=("$@")
  elif _is_date "$1"; then
    ASOF="$1"
    shift
    if [[ "${1:-}" == "--" ]]; then
      shift
      EXTRA_SIM_ARGS=("$@")
    fi
  else
    echo "ERROR: first arg must be YYYY-MM-DD or '--'. Got: $1" >&2
    exit 1
  fi
fi

if [[ -z "${ASOF}" ]]; then
  ASOF="$(_yesterday)"
  if [[ -z "$ASOF" ]]; then
    echo "ERROR: cannot compute yesterday; pass ASOF YYYY-MM-DD." >&2
    exit 1
  fi
fi

# ------------- required env -------------
CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

# ------------- knobs (env) -------------
START_DATE="${START_DATE:-2025-09-05}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"

DEVICE="${DEVICE:-cuda}"        # XGB train device
SIM_DEVICE="${SIM_DEVICE:-cpu}" # simulate device (avoid device mismatch warnings)

HORIZON_SECS="${HORIZON_SECS:-120}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"
EV_MODE="${EV_MODE:-mtm}"

EDGE_THRESH="${EDGE_THRESH:-0.0}"
STAKE="${STAKE:-kelly}"
KELLY_CAP="${KELLY_CAP:-0.02}"
KELLY_FLOOR="${KELLY_FLOOR:-0.001}"
BANKROLL_NOM="${BANKROLL_NOM:-5000}"

# EV rescaling + liquidity
EV_SCALE="${EV_SCALE:-1.0}"
REQUIRE_BOOK="${REQUIRE_BOOK:-0}"
MIN_FILL_FRAC="${MIN_FILL_FRAC:-0.0}"

# Odds band
ODDS_MIN="${ODDS_MIN:-}"
ODDS_MAX="${ODDS_MAX:-}"

# Liquidity
ENFORCE_LIQUIDITY="${ENFORCE_LIQUIDITY:-0}"
LIQUIDITY_LEVELS="${LIQUIDITY_LEVELS:-1}"

# Portfolio sizing
PER_MARKET_TOPK="${PER_MARKET_TOPK:-1}"
PER_MARKET_BUDGET="${PER_MARKET_BUDGET:-10}"
BASKET_SIZING="${BASKET_SIZING:-equal_ev}"
EXIT_ON_MOVE_TICKS="${EXIT_ON_MOVE_TICKS:-0}"

OUTDIR="${OUTDIR:-$BASE_DIR/output}"

# ------------- headers -------------
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

# ------------- TRAIN (only train-supported args) -------------
python3 "$BASE_DIR/ml/train_price_trend.py" \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --horizon-secs "$HORIZON_SECS" \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --device "$DEVICE" \
  --output-dir "$OUTDIR"

# ------------- SIM header -------------
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
echo "Portfolio:       topK=$PER_MARKET_TOPK budget=$PER_MARKET_BUDGET sizing=$BASKET_SIZING exit_ticks=$EXIT_ON_MOVE_TICKS"
echo "Sim device:      $SIM_DEVICE"
echo "Output dir:      $OUTDIR/stream"

# ------------- SIM args -------------
SIM_ARGS=(
  --curated "$CURATED_ROOT"
  --asof "$ASOF"
  --start-date "$START_DATE"
  --valid-days "$VALID_DAYS"
  --sport "$SPORT"
  --horizon-secs "$HORIZON_SECS"
  --preoff-max "$PREOFF_MAX"
  --commission "$COMMISSION"
  --ev-scale "${EV_SCALE:-1.0}"
  --output-dir "$OUTDIR/stream"
  # Optional performance/sampling knobs (uncomment/set envs if you want)
  # --max-files-per-day "${MAX_FILES_PER_DAY:-0}"
  # --file-sample-mode "${FILE_SAMPLE_MODE:-uniform}"
  # --row-sample-secs "${ROW_SAMPLE_SECS:-0}"
  # --polars-max-threads "${POLARS_MAX_THREADS:-0}"
  # --defs-days-back "${DEFS_DAYS_BACK:-30}"
  # --defs-days-forward "${DEFS_DAYS_FORWARD:-7}"
  # --fallback-coverage-thresh "${FALLBACK_COVERAGE_THRESH:-0.01}"
  # --force-skip-preoff   # include only if you explicitly want to skip pre-off
)


# Optional odds band
[[ -n "$ODDS_MIN" ]] && SIM_ARGS+=( --odds-min "$ODDS_MIN" )
[[ -n "$ODDS_MAX" ]] && SIM_ARGS+=( --odds-max "$ODDS_MAX" )

# Liquidity flags
if [[ "$ENFORCE_LIQUIDITY" == "1" ]]; then
  SIM_ARGS+=( --enforce-liquidity --liquidity-levels "$LIQUIDITY_LEVELS" )
fi
[[ "$REQUIRE_BOOK" == "1" ]] && SIM_ARGS+=( --require-book )

# Append any extra simulator-only args after '--'
if [[ ${#EXTRA_SIM_ARGS[@]} -gt 0 ]]; then
  SIM_ARGS+=("${EXTRA_SIM_ARGS[@]}")
fi

python3 "$BASE_DIR/ml/simulate_stream.py" "${SIM_ARGS[@]}"
