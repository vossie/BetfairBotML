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
  if command -v nproc >/dev/null 2>&1; then nproc
  elif command -v getconf >/dev/null 2>&1; then getconf _NPROCESSORS_ONLN
  else echo 8; fi
}
THREADS="$(detect_threads)"
# Conservative default: leave ~1/3 cores free; keep at least 2
DEFAULT_MP=$(( THREADS > 12 ? THREADS - (THREADS/3) : (THREADS > 4 ? THREADS - 2 : 2) ))
MAX_PARALLEL="${MAX_PARALLEL:-$DEFAULT_MP}"

# Per-process thread caps (good for IO-bound Polars workloads)
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Batch size for simulator (lower → less RAM per proc)
export BATCH_SIZE="${BATCH_SIZE:-75000}"

echo "[run_best_trend] Threads: total=$THREADS  max_parallel=$MAX_PARALLEL  (POLARS_MAX_THREADS=$POLARS_MAX_THREADS, BATCH_SIZE=$BATCH_SIZE)"

# ---------- AutoML if needed ----------
if [[ ! -f "$BEST_CFG" ]]; then
  echo "[run_best_trend] No best_config.json at $BEST_CFG"
  echo "[run_best_trend] Running AutoML (train on GPU, simulate on CPU)…"

  # Rolling start date: 21 days before ASOF (override via START_DATE env)
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

  # Defaults for robust sweep (override via env if desired)
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
  EV_MODE="${EV_MODE:-mtm}"
  EDGE_GRID="${EDGE_GRID:-0.0005,0.001,0.0015,0.002,0.003}"
  STAKE_GRID="${STAKE_GRID:-flat,kelly}"
  # Prefer realistic bands first; 'none' last only as diagnostic
  ODDS_GRID="${ODDS_GRID:-2.2:3.6,1.5:5.0,none}"
  # Add liquidity sweeps only if explicitly requested
  ENFORCE_LIQ="${ENFORCE_LIQUIDITY:-}"
  LIQ_GRID="${LIQUIDITY_LEVELS_GRID:-1,3}"
  # Safety filters for ranking
  MIN_TRADES="${MIN_TRADES:-20000}"
  PREFER="${PREFER:-settlement}"
  EV_CAP="${EV_CAP:-1.0}"
  VALID_DAYS="${VALID_DAYS:-7}"
  SPORT="${SPORT:-horse-racing}"
  PREOFF_MAX="${PREOFF_MAX:-30}"
  HORIZON_SECS="${HORIZON_SECS:-120}"
  COMMISSION="${COMMISSION:-0.02}"
  BANKROLL_NOM="${BANKROLL_NOM:-5000}"
  KELLY_CAP="${KELLY_CAP:-0.02}"
  KELLY_FLOOR="${KELLY_FLOOR:-0.001}"

  # Run AutoML
  set -x
  python3 "$BASE_DIR/ml/train_price_trend_automl.py" \
    --curated "$CURATED_ROOT" \
    --asof "$ASOF" \
    --start-date "$SD" \
    --valid-days "$VALID_DAYS" \
    --sport "$SPORT" \
    --preoff-max "$PREOFF_MAX" \
    --horizon-secs "$HORIZON_SECS" \
    --commission "$COMMISSION" \
    --train-device "$TRAIN_DEVICE" \
    --sim-device cpu \
    --max-parallel "$MAX_PARALLEL" \
    --ev-mode "$EV_MODE" \
    --edge-grid "$EDGE_GRID" \
    --stake-grid "$STAKE_GRID" \
    --odds-grid "$ODDS_GRID" \
    ${ENFORCE_LIQ:+--enforce-liquidity} \
    --liquidity-levels-grid "$LIQ_GRID" \
    --bankroll-nom "$BANKROLL_NOM" \
    --kelly-cap "$KELLY_CAP" \
    --kelly-floor "$KELLY_FLOOR" \
    --batch-size "$BATCH_SIZE" \
    --ev-cap "$EV_CAP" \
    --min-trades "$MIN_TRADES" \
    --prefer "$PREFER" \
    --output-dir "$OUTDIR" \
    --tag "$TAG"
  set +x
fi

# ---------- jq required ----------
if ! command -v jq >/dev/null 2>&1; then
  echo "❌ jq is required but not installed." >&2
  exit 1
fi

if [[ ! -f "$BEST_CFG" ]]; then
  echo "❌ best_config.json still not found at $BEST_CFG" >&2
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

echo "[run_best_trend] ✅ Backtesting best configuration (ASOF=$ASOF, TAG=$TAG)…"
"$BIN_DIR/train_price_trend.sh" "$ASOF"

# ---------- Cleanup old AutoML outputs ----------
KEEP_AUTOML_RUNS="${KEEP_AUTOML_RUNS:-3}"
AUTOML_ROOT="$OUTDIR/automl"

cleanup_automl() {
  local root="$1"
  local keep_n="$2"
  local dry="${3:-0}"
  [[ -d "$root" ]] || { echo "[cleanup] No automl dir at $root — skipping."; return; }
  mapfile -t ASOF_DIRS < <(ls -1 "$root" 2>/dev/null | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' | sort)
  local total="${#ASOF_DIRS[@]}"
  if (( total <= keep_n )); then
    echo "[cleanup] $total runs <= keep_n=$keep_n — nothing to remove."
    return
  fi
  local to_delete=("${ASOF_DIRS[@]:0:total-keep_n}")
  echo "[cleanup] Keeping last $keep_n runs; removing ${#to_delete[@]} older runs:"
  for d in "${to_delete[@]}"; do
    local path="$root/$d"
    if [[ "$dry" == "1" ]]; then
      echo "  DRY_RUN: rm -rf \"$path\""
    else
      echo "  rm -rf \"$path\""
      rm -rf "$path" || echo "  WARN: failed to remove $path"
    fi
  done
}

echo "[cleanup] Pruning old AutoML outputs under $AUTOML_ROOT (keep $KEEP_AUTOML_RUNS)…"
cleanup_automl "$AUTOML_ROOT" "$KEEP_AUTOML_RUNS" "${DRY_RUN:-0}"

echo "[run_best_trend] Done."
