#!/usr/bin/env bash
set -euo pipefail

# ===================== Paths =====================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$BASE_DIR/ml"
OUT_BASE="$BASE_DIR/output/stream"
MODEL_PATH="$BASE_DIR/output/models/xgb_trend_reg.json"

: "${CURATED_ROOT:?Please export CURATED_ROOT (e.g. /mnt/nvme/betfair-curated)}"

# ===================== ASOF =====================
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  ASOF="$1"
else
  ASOF="$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || python3 - <<'PY'
from datetime import datetime, timedelta
print((datetime.utcnow().date() - timedelta(days=1)).strftime("%Y-%m-%d"))
PY
)"
fi

# ===================== Config (env) =====================
START_DAYS_BACK="${START_DAYS_BACK:-34}"
VALID_DAYS="${VALID_DAYS:-7}"
USE_EARLIEST="${USE_EARLIEST:-0}"                 # rolling window by default
EARLIEST_DATE="${EARLIEST_DATE:-2025-09-05}"
START_DATE_FORCE="${START_DATE_FORCE:-}"

SPORT="${SPORT:-horse-racing}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"

TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
SIM_DEVICE="${SIM_DEVICE:-cuda}"

TAG="${TAG:-auto_strict}"
CLEAN_MODE="${CLEAN_MODE:-skip}"        # skip|auto|prompt

# Edge search targets
TPD_MIN="${TPD_MIN:-1000}"
TPD_MAX="${TPD_MAX:-2000}"
EDGE_MIN="${EDGE_MIN:-0.0005}"
EDGE_MAX="${EDGE_MAX:-0.03}"
EDGE_INIT_GRID="${EDGE_INIT_GRID:-0.003,0.005,0.01,0.015}"
MAX_EDGE_ITER="${MAX_EDGE_ITER:-7}"

# Non-edge grids
STAKE_MODES_GRID="${STAKE_MODES_GRID:-flat,kelly}"
ODDS_BANDS_GRID="${ODDS_BANDS_GRID:-2.2:3.6,1.5:5.0}"
EV_SCALE_GRID="${EV_SCALE_GRID:-0.005,0.01,0.02,0.05}"
EV_CAP_GRID="${EV_CAP_GRID:-0.05}"
EXIT_TICKS_GRID="${EXIT_TICKS_GRID:-0,1}"
TOPK_GRID="${TOPK_GRID:-1}"
BUDGET_GRID="${BUDGET_GRID:-5,10}"
LIQ_ENFORCE_GRID="${LIQ_ENFORCE_GRID:-1}"
MIN_FILL_FRAC_GRID="${MIN_FILL_FRAC_GRID:-5.0,10.0,25.0,50.0}"

# EV density
SAMPLE_EV_DENSITY="${SAMPLE_EV_DENSITY:-1}"
EV_HIST_BINS="${EV_HIST_BINS:-0,0.0005,0.001,0.002,0.003,0.005,0.01}"

# Perf
PARALLEL="${PARALLEL:-6}"
TIMEOUT_SECS="${TIMEOUT_SECS:-2400}"
TOTAL_CORES="${TOTAL_CORES:-36}"

# Tuner
TUNE_ENABLE="${TUNE_ENABLE:-1}"
TUNE_LOG_DIR="$BASE_DIR/output/tune/$ASOF"
TUNE_MAX_TRIALS="${TUNE_MAX_TRIALS:-24}"
XGB_DEPTHS="${XGB_DEPTHS:-5 6}"
XGB_N_EST="${XGB_N_EST:-400 600}"
XGB_ETAS="${XGB_ETAS:-0.05 0.10}"
XGB_MIN_CHILD_WEIGHT="${XGB_MIN_CHILD_WEIGHT:-1.0 2.0}"
XGB_SUBSAMPLE="${XGB_SUBSAMPLE:-0.8 0.9}"
XGB_COLSAMPLE="${XGB_COLSAMPLE:-0.8 0.9}"
XGB_L2="${XGB_L2:-1.0 2.0}"
XGB_L1="${XGB_L1:-0.0 0.5}"
XGB_EARLY_STOP="${XGB_EARLY_STOP:-50}"

# Optional sampling passthrough
MAX_FILES_PER_DAY="${MAX_FILES_PER_DAY:-}"
FILE_SAMPLE_MODE="${FILE_SAMPLE_MODE:-}"
ROW_SAMPLE_SECS="${ROW_SAMPLE_SECS:-}"

# Sweeper guard (ensure enough trades overall across the window)
MIN_TRADES="${MIN_TRADES:-2000}"

# ===================== Derived dirs =====================
SWEEP_ROOT="$OUT_BASE/sweeps/$ASOF/$TAG"
CONFIRM_OUT="$OUT_BASE/confirm_${ASOF}_${TAG}"
TRIALS="$SWEEP_ROOT/trials.csv"            # define early to avoid set -u issues
BEST="$SWEEP_ROOT/best_config.json"        # define early to avoid set -u issues

# ===================== Start date =====================
if [[ -n "${START_DATE_FORCE}" ]]; then
  START_DATE="$START_DATE_FORCE"
  echo "[auto] Using fixed START_DATE=${START_DATE}"
elif [[ "$USE_EARLIEST" == "1" ]]; then
  START_DATE="$EARLIEST_DATE"
  echo "[auto] USE_EARLIEST=1 → START_DATE=${START_DATE}"
else
  START_DATE="$(python3 - <<PY
from datetime import datetime, timedelta
asof=datetime.strptime("$ASOF","%Y-%m-%d").date()
print((asof - timedelta(days=int("$START_DAYS_BACK"))).strftime("%Y-%m-%d"))
PY
)"
fi

python3 - <<PY
from datetime import datetime, timedelta
asof=datetime.strptime("$ASOF","%Y-%m-%d").date()
start=datetime.strptime("$START_DATE","%Y-%m-%d").date()
cutoff=asof - timedelta(days=int("$VALID_DAYS"))
assert start <= cutoff, f"START_DATE {start} must be <= ASOF-VALID_DAYS {cutoff}"
print("[auto] Train window:", start, "→", cutoff)
PY

echo "[auto] OUT_BASE=$OUT_BASE"
echo "[auto] SWEEP_ROOT=$SWEEP_ROOT"
echo "[auto] CONFIRM_OUT=$CONFIRM_OUT"
echo "[auto] Ensuring output dirs…"

clean_dir() {
  local path="$1"
  case "$CLEAN_MODE" in
    skip)   echo "[auto] CLEAN_MODE=skip → keeping '$path'." ;;
    auto)   [[ -d "$path" ]] && { echo "[auto] Cleaning: $path"; rm -rf "$path"; } ;;
    prompt) if [[ -d "$path" ]]; then
              read -p "[auto] Found '$path'. Delete? [y/N] " R
              [[ "$R" =~ ^[Yy]$ ]] && rm -rf "$path" || { echo "[auto] Aborting."; exit 1; }
            fi ;;
    *) echo "[auto] Unknown CLEAN_MODE='$CLEAN_MODE'"; exit 1 ;;
  esac
}
mkdir -p "$OUT_BASE"
clean_dir "$SWEEP_ROOT";   mkdir -p "$SWEEP_ROOT"
clean_dir "$CONFIRM_OUT";  mkdir -p "$CONFIRM_OUT"
mkdir -p "$TUNE_LOG_DIR"

# ===================== Threading =====================
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-$TOTAL_CORES}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$TOTAL_CORES}"
export XGBOOST_NUM_THREADS="${XGBOOST_NUM_THREADS:-$TOTAL_CORES}"

# ===================== Helpers =====================
py_float_gt () { python3 - "$1" "$2" <<'PY'
import sys
a=float(sys.argv[1]); b=float(sys.argv[2])
print("yes" if a>b else "no")
PY
}

# Robust parser: extract ONLY mean= number
parse_valid_ev () {
  python3 - <<'PY'
import sys, re
txt=sys.stdin.read()
m=re.search(r'valid EV per £1:\s*mean\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', txt)
print(m.group(1) if m else "-inf")
PY
}

json_get () {
  local key="$1"; local file="$2"
  [[ -f "$file" ]] || { echo ""; return; }
  python3 - "$key" "$file" <<'PY'
import json, sys
k=sys.argv[1]; p=sys.argv[2]
try:
  with open(p) as f: d=json.load(f)
  v=d.get(k, "")
  if isinstance(v, bool): print("true" if v else "false")
  else: print(v if v is not None else "")
except Exception:
  print("")
PY
}

# ===================== Tuner =====================
BEST_SCORE="-inf"; BEST_TAG=""; BEST_ARGS=()
if [[ "${TUNE_ENABLE}" == "1" ]]; then
  echo "=== TUNE (daily XGBoost hyperparams) ${ASOF} ==="
  trial_count=0
  for d in $XGB_DEPTHS; do
    for n in $XGB_N_EST; do
      for eta in $XGB_ETAS; do
        for mc in $XGB_MIN_CHILD_WEIGHT; do
          for sub in $XGB_SUBSAMPLE; do
            for col in $XGB_COLSAMPLE; do
              for l2 in $XGB_L2; do
                for l1 in $XGB_L1; do
                  trial_count=$((trial_count+1))
                  [[ $trial_count -gt $TUNE_MAX_TRIALS ]] && { echo "[tune] Reached TUNE_MAX_TRIALS=$TUNE_MAX_TRIALS"; break 8; }
                  tag="d${d}_n${n}_eta${eta}_mc${mc}_sub${sub}_col${col}_l2${l2}_l1${l1}"
                  logf="$TUNE_LOG_DIR/${tag}.log"
                  echo "[tune] $trial_count) $tag"
                  set +e
                  python3 -u "$ML_DIR/train_price_trend.py" \
                    --curated "$CURATED_ROOT" \
                    --asof "$ASOF" \
                    --start-date "$START_DATE" \
                    --valid-days "$VALID_DAYS" \
                    --sport "$SPORT" \
                    --horizon-secs 120 \
                    --preoff-max "$PREOFF_MAX" \
                    --commission "$COMMISSION" \
                    --device "$TRAIN_DEVICE" \
                    --xgb-max-depth "$d" \
                    --xgb-num-boost-round "$n" \
                    --xgb-eta "$eta" \
                    --xgb-min-child-weight "$mc" \
                    --xgb-subsample "$sub" \
                    --xgb-colsample-bytree "$col" \
                    --xgb-reg-lambda "$l2" \
                    --xgb-reg-alpha "$l1" \
                    --xgb-early-stopping-rounds "$XGB_EARLY_STOP" \
                    | tee "$logf"
                  rc=$?
                  set -e
                  [[ $rc -ne 0 ]] && { echo "[tune] WARN: trainer rc=$rc; skipping."; continue; }

                  score="$(parse_valid_ev < "$logf")"
                  echo "    → valid_EV_per_£1=${score}"
                  if [[ "$(py_float_gt "${score}" "${BEST_SCORE}")" == "yes" ]]; then
                    BEST_SCORE="$score"; BEST_TAG="$tag"
                    BEST_ARGS=( \
                      --xgb-max-depth "$d" \
                      --xgb-num-boost-round "$n" \
                      --xgb-eta "$eta" \
                      --xgb-min-child-weight "$mc" \
                      --xgb-subsample "$sub" \
                      --xgb-colsample-bytree "$col" \
                      --xgb-reg-lambda "$l2" \
                      --xgb-reg-alpha "$l1" \
                      --xgb-early-stopping-rounds "$XGB_EARLY_STOP" )
                    echo "    ✔ new best: $BEST_SCORE ($BEST_TAG)"
                  fi
                done
              done
            done
          done
        done
      done
    done
  done
  echo "[tune] DONE. Best by valid EV/£1 = $BEST_SCORE  ($BEST_TAG)"
else
  echo "=== TUNE disabled (TUNE_ENABLE=0) ==="
fi

# ===================== Final TRAIN =====================
TRAIN_EXTRA=()
if [[ -n "" && "" != "-inf" && ${#BEST_ARGS[@]} -gt 0 ]]; then
  TRAIN_EXTRA=( "${BEST_ARGS[@]}" )
else
  echo "[tune] No valid best trial; using defaults (no extra XGB args)."
  TRAIN_EXTRA=()
fi
echo "=== TRAIN ${ASOF} ==="
python3 -u "$ML_DIR/train_price_trend.py" \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --horizon-secs 120 \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --device "$TRAIN_DEVICE" \
  ${TRAIN_EXTRA[@]}

# ===================== Build extra sim args =====================
SIM_EXTRA_ARGS=()
[[ -n "${MAX_FILES_PER_DAY}" ]] && SIM_EXTRA_ARGS+=( --max-files-per-day "$MAX_FILES_PER_DAY" )
[[ -n "${FILE_SAMPLE_MODE}"  ]] && SIM_EXTRA_ARGS+=( --file-sample-mode "$FILE_SAMPLE_MODE" )
[[ -n "${ROW_SAMPLE_SECS}"   ]] && SIM_EXTRA_ARGS+=( --row-sample-secs "$ROW_SAMPLE_SECS" )

# Split cores across workers
SIM_THREADS=$(( TOTAL_CORES / PARALLEL )); [[ $SIM_THREADS -lt 2 ]] && SIM_THREADS=2
export POLARS_MAX_THREADS="$SIM_THREADS"
export OMP_NUM_THREADS="$SIM_THREADS"
export XGBOOST_NUM_THREADS="$SIM_THREADS"
export NUMEXPR_MAX_THREADS="$SIM_THREADS"

# ===================== SWEEP =====================
echo "=== SWEEP (adaptive edge; realised ROI objective) ==="
python3 -u "$ML_DIR/sim_sweep.py" \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --device "$SIM_DEVICE" \
  --model-path "$MODEL_PATH" \
  --base-output-dir "$OUT_BASE" \
  --bankroll-nom 5000 \
  --kelly-cap 0.02 \
  --kelly-floor 0.001 \
  --stake-modes "$STAKE_MODES_GRID" \
  --odds-bands "$ODDS_BANDS_GRID" \
  --ev-scale-grid "$EV_SCALE_GRID" \
  --ev-cap-grid "$EV_CAP_GRID" \
  --exit-ticks-grid "$EXIT_TICKS_GRID" \
  --topk-grid "$TOPK_GRID" \
  --budget-grid "$BUDGET_GRID" \
  --liq-enforce-grid "$LIQ_ENFORCE_GRID" \
  --min-fill-frac-grid "$MIN_FILL_FRAC_GRID" \
  --edge-min "$EDGE_MIN" \
  --edge-max "$EDGE_MAX" \
  --edge-init-grid "$EDGE_INIT_GRID" \
  --max-edge-iterations "$MAX_EDGE_ITER" \
  --target-trades-per-day-min "$TPD_MIN" \
  --target-trades-per-day-max "$TPD_MAX" \
  --min-trades "$MIN_TRADES" \
  --parallel "$PARALLEL" \
  --timeout-secs "$TIMEOUT_SECS" \
  $( [[ "$SAMPLE_EV_DENSITY" == "1" ]] && echo --sample-ev-density ) \
  --ev-hist-bins "$EV_HIST_BINS" \
  --tag "$TAG" \
  "${SIM_EXTRA_ARGS[@]}"

# Guard: define TRIALS/BEST even if sweep failed (avoid set -u)
TRIALS="${TRIALS:-$SWEEP_ROOT/trials.csv}"
BEST="${BEST:-$SWEEP_ROOT/best_config.json}"

echo "=== BEST (pre-confirm) ==="
if [[ -f "$BEST" ]]; then
  sed -e 's/^/  /' "$BEST"
else
  echo "  (missing) $BEST"
fi

# ===================== CONFIRM =====================
# Give full cores back
export POLARS_MAX_THREADS="$TOTAL_CORES"
export OMP_NUM_THREADS="$TOTAL_CORES"
export XGBOOST_NUM_THREADS="$TOTAL_CORES"

EDGE_THR="$(json_get edge_thresh "$BEST")"
STAKE_MODE="$(json_get stake_mode "$BEST")"
ODDS_MIN="$(json_get odds_min "$BEST")"
ODDS_MAX="$(json_get odds_max "$BEST")"
ENF_LIQ="$(json_get enforce_liquidity_effective "$BEST")"
MIN_FILL_FRAC="$(json_get min_fill_frac "$BEST")"
TOPK="$(json_get per_market_topk "$BEST")"
BUDGET="$(json_get per_market_budget "$BEST")"
EXIT_TICKS="$(json_get exit_on_move_ticks "$BEST")"
EV_SCALE_USED="$(json_get ev_scale_used "$BEST")"
EV_CAP="$(json_get ev_cap "$BEST")"

CONF_ARGS=( )
[[ "$ENF_LIQ" == "true" ]] && CONF_ARGS+=( --enforce-liquidity --min-fill-frac "$MIN_FILL_FRAC" )

if [[ -n "${EDGE_THR}" && -n "${STAKE_MODE}" && -n "${ODDS_MIN}" && -n "${ODDS_MAX}" ]]; then
  echo "=== CONFIRM RUN ==="
  python3 -u "$ML_DIR/simulate_stream.py" \
    --curated "$CURATED_ROOT" \
    --asof "$ASOF" --start-date "$START_DATE" --valid-days "$VALID_DAYS" \
    --sport "$SPORT" --preoff-max "$PREOFF_MAX" --horizon-secs 120 --commission "$COMMISSION" \
    --model-path "$MODEL_PATH" \
    --edge-thresh "$EDGE_THR" \
    --stake-mode "$STAKE_MODE" --bankroll-nom 5000 --kelly-cap 0.02 --kelly-floor 0.001 \
    --odds-min "$ODDS_MIN" --odds-max "$ODDS_MAX" \
    --per-market-topk "$TOPK" --per-market-budget "$BUDGET" \
    --exit-on-move-ticks "$EXIT_TICKS" \
    --ev-scale "$EV_SCALE_USED" --ev-cap "$EV_CAP" \
    --device "$SIM_DEVICE" \
    --output-dir "$CONFIRM_OUT" \
    "${CONF_ARGS[@]}" \
    "${SIM_EXTRA_ARGS[@]}"
else
  echo "[auto] Skipping CONFIRM (incomplete BEST config)."
fi

echo "=== CONFIRM SUMMARY ==="
if [[ -f "$CONFIRM_OUT/summary_${ASOF}.json" ]]; then
  sed -e 's/^/  /' "$CONFIRM_OUT/summary_${ASOF}.json"
else
  echo "  (missing) $CONFIRM_OUT/summary_${ASOF}.json"
fi

echo "[auto] Done."
echo "  Train window:   ${START_DATE} → $(python3 - <<PY
from datetime import datetime, timedelta
asof=datetime.strptime("$ASOF","%Y-%m-%d").date()
print((asof - timedelta(days=int("$VALID_DAYS"))).strftime("%Y-%m-%d"))
PY
)"
echo "  Trials:         ${TRIALS}"
echo "  Best config:    ${BEST}"
echo "  Confirm summary:${CONFIRM_OUT}/summary_${ASOF}.json"
