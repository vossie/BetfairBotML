#!/usr/bin/env bash
set -euo pipefail

# ---------- ASOF (default: yesterday) ----------
if [[ $# -ge 1 ]]; then
  ASOF="$1"
  TAG="${2:-automl}"
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
  TAG="automl"
fi

CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

# ---------- Paths ----------
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$BIN_DIR/.." && pwd)"
OUTDIR="${OUTDIR:-$BASE_DIR/output}"
BEST_CFG="$OUTDIR/automl/$ASOF/$TAG/best_config.json"

# ---------- Parallelism (auto; override with MAX_PARALLEL) ----------
detect_threads() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    echo 8
  fi
}
THREADS="$(detect_threads)"
# leave a few cores free for IO/OS; floor at 2
DEFAULT_MP=$(( THREADS > 8 ? THREADS - 4 : (THREADS > 2 ? THREADS - 1 : 2) ))
MAX_PARALLEL="${MAX_PARALLEL:-$DEFAULT_MP}"

echo "[run_best_trend] Detected threads: $THREADS  -> max_parallel=$MAX_PARALLEL"

# ---------- AutoML if needed ----------
if [[ ! -f "$BEST_CFG" ]]; then
  echo "[run_best_trend] No best_config.json at $BEST_CFG"
  echo "[run_best_trend] Running AutoML (train on GPU, simulate on CPU, parallel=$MAX_PARALLEL)…"

  # Rolling start date: 21 days before ASOF (override with START_DATE env if you want)
  if [[ -n "${START_DATE:-}" ]]; then
    SD="$START_DATE"
  else
    if date -d "$ASOF -21 days" +%Y-%m-%d >/dev/null 2>&1; then
      SD="$(date -d "$ASOF -21 days" +%Y-%m-%d)"
    elif command -v gdate >/dev/null 2>&1; then
      SD="$(gdate -d "$ASOF -21 days" +%Y-%m-%d)"
    else
      SD="$ASOF"
    fi
  fi

  python3 "$BASE_DIR/ml/train_price_trend_automl.py" \
    --curated "$CURATED_ROOT" \
    --asof "$ASOF" \
    --start-date "$SD" \
    --valid-days "${VALID_DAYS:-7}" \
    --sport "${SPORT:-horse-racing}" \
    --preoff-max "${PREOFF_MAX:-30}" \
    --horizon-secs "${HORIZON_SECS:-120}" \
    --commission "${COMMISSION:-0.02}" \
    --train-device "${TRAIN_DEVICE:-cuda}" \
    --sim-device cpu \
    --max-parallel "$MAX_PARALLEL" \
    --ev-mode "${EV_MODE:-mtm}" \
    --edge-grid "${EDGE_GRID:-0.0005,0.001,0.0015,0.002,0.003}" \
    --stake-grid "${STAKE_GRID:-flat,kelly}" \
    --odds-grid "${ODDS_GRID:-none,2.2:3.6,1.5:5.0}" \
    ${ENFORCE_LIQUIDITY:+--enforce-liquidity} \
    --liquidity-levels-grid "${LIQUIDITY_LEVELS_GRID:-1,3}" \
    --bankroll-nom "${BANKROLL_NOM:-5000}" \
    --kelly-cap "${KELLY_CAP:-0.02}" \
    --kelly-floor "${KELLY_FLOOR:-0.001}" \
    --tag "$TAG"

  if [[ ! -f "$BEST_CFG" ]]; then
    echo "❌ AutoML finished but best_config.json not found: $BEST_CFG" >&2
    exit 1
  fi
fi

# ---------- jq required ----------
if ! command -v jq >/dev/null 2>&1; then
  echo "❌ jq is required but not installed." >&2
  exit 1
fi

# ---------- Read best config ----------
edge=$(jq -r '.edge_thresh' "$BEST_CFG")
stake=$(jq -r '.stake_mode' "$BEST_CFG")
odds_min=$(jq -r '.odds_min' "$BEST_CFG")
odds_max=$(jq -r '.odds_max' "$BEST_CFG")
liq_enf=$(jq -r '.enforce_liquidity' "$BEST_CFG")
liq_levels=$(jq -r '.liquidity_levels' "$BEST_CFG")

export START_DATE=$(jq -r '.start_date' "$BEST_CFG")
export VALID_DAYS=$(jq -r '.valid_days' "$BEST_CFG")
export SPORT=$(jq -r '.sport' "$BEST_CFG")
export PREOFF_MAX=$(jq -r '.preoff_max' "$BEST_CFG")
export HORIZON_SECS=$(jq -r '.horizon_secs' "$BEST_CFG")
export COMMISSION=$(jq -r '.commission' "$BEST_CFG")
export DEVICE=$(jq -r '.device' "$BEST_CFG")        # training device used
export EV_MODE=$(jq -r '.ev_mode' "$BEST_CFG")
export BANKROLL_NOM=$(jq -r '.bankroll_nom' "$BEST_CFG")
export KELLY_CAP=$(jq -r '.kelly_cap' "$BEST_CFG")
export KELLY_FLOOR=$(jq -r '.kelly_floor' "$BEST_CFG")

export EDGE_THRESH="$edge"
export STAKE="$stake"

# Optional odds band
if [[ "$odds_min" != "null" ]]; then export ODDS_MIN="$odds_min"; fi
if [[ "$odds_max" != "null" ]]; then export ODDS_MAX="$odds_max"; fi

# Liquidity flags
if [[ "$liq_enf" == "true" ]]; then
  export ENFORCE_LIQUIDITY=1
else
  export ENFORCE_LIQUIDITY=0
fi
export LIQUIDITY_LEVELS="$liq_levels"

echo "[run_best_trend] ✅ Running backtest with best configuration (ASOF=$ASOF, TAG=$TAG)…"
"$BIN_DIR/train_price_trend.sh" "$ASOF"
