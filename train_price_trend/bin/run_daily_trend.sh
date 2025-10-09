#!/usr/bin/env bash
set -euo pipefail

# === Daily end-to-end: TRAIN → ADAPTIVE SWEEP (realised ROI) → CONFIRM WINNER ===
# Usage:
#   export CURATED_ROOT=/mnt/nvme/betfair-curated
#   ./train_price_trend/bin/run_daily_trend.sh [ASOF]
# If ASOF is omitted, defaults to **yesterday** (YYYY-MM-DD).

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$BASE_DIR/ml"
OUT_BASE="$BASE_DIR/output/stream"
MODEL_PATH="$BASE_DIR/output/models/xgb_trend_reg.json"

# ── Inputs ─────────────────────────────────────────────────────────────────────
: "${CURATED_ROOT:?Please export CURATED_ROOT (e.g. /mnt/nvme/betfair-curated)}"

# ASOF default = yesterday (POSIX date; fallback to Python on systems without GNU date)
if [[ $# -ge 1 ]]; then
  ASOF="$1"
else
  ASOF="$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || python3 - <<'PY'
from datetime import datetime, timedelta
print((datetime.utcnow().date() - timedelta(days=1)).strftime("%Y-%m-%d"))
PY
)"
fi

# ── Config (override via env) ──────────────────────────────────────────────────
# Windows
START_DAYS_BACK="${START_DAYS_BACK:-34}"   # only used if START_DATE_FORCE/USE_EARLIEST not set
VALID_DAYS="${VALID_DAYS:-7}"
USE_EARLIEST="${USE_EARLIEST:-0}"          # if 1, use EARLIEST_DATE as start
EARLIEST_DATE="${EARLIEST_DATE:-2025-09-05}"  # earliest curated date
SPORT="${SPORT:-horse-racing}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"

# Devices / performance
SIM_DEVICE="${SIM_DEVICE:-cpu}"            # cpu | cuda
POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-$(nproc)}"

# Cleanup behaviour for same-day reruns
CLEAN_MODE="${CLEAN_MODE:-auto}"           # auto | prompt | skip
TAG="${TAG:-auto}"

# Optional sampling to speed dev runs (passed through to simulate_stream)
MAX_FILES_PER_DAY="${MAX_FILES_PER_DAY:-}"     # e.g. 6000
FILE_SAMPLE_MODE="${FILE_SAMPLE_MODE:-}"       # uniform|head|tail
ROW_SAMPLE_SECS="${ROW_SAMPLE_SECS:-}"         # e.g. 15

# Guardrails
MIN_TRADES="${MIN_TRADES:-5000}"
MAX_TRADES="${MAX_TRADES:-250000}"
ROI_TARGET_MIN="${ROI_TARGET_MIN:-0.0}"

# Adaptive sweep targets (used by adaptive sim_sweep.py drop-in)
TPD_MIN="${TPD_MIN:-1000}"                 # target trades/day min
TPD_MAX="${TPD_MAX:-2000}"                 # target trades/day max
EDGE_MIN="${EDGE_MIN:-0.0001}"
EDGE_MAX="${EDGE_MAX:-0.005}"
EDGE_INIT_GRID="${EDGE_INIT_GRID:-0.0005,0.001,0.002}"
MAX_EDGE_ITER="${MAX_EDGE_ITER:-6}"

# Non-edge grids still swept
STAKE_MODES_GRID="${STAKE_MODES_GRID:-flat,kelly}"
ODDS_BANDS_GRID="${ODDS_BANDS_GRID:-1.5:5.0,2.2:3.6}"
EV_SCALE_GRID="${EV_SCALE_GRID:-0.05,0.1,0.2}"
EV_CAP_GRID="${EV_CAP_GRID:-0.05}"
EXIT_TICKS_GRID="${EXIT_TICKS_GRID:-0,1}"
TOPK_GRID="${TOPK_GRID:-1}"
BUDGET_GRID="${BUDGET_GRID:-5,10}"
LIQ_ENFORCE_GRID="${LIQ_ENFORCE_GRID:-1}"
MIN_FILL_FRAC_GRID="${MIN_FILL_FRAC_GRID:-5.0}"

# EV density logging
SAMPLE_EV_DENSITY="${SAMPLE_EV_DENSITY:-1}"    # 1=on, 0=off
EV_HIST_BINS="${EV_HIST_BINS:-0,0.0005,0.001,0.002,0.003,0.005,0.01}"

# ── Compute START_DATE ─────────────────────────────────────────────────────────
if [[ -n "${START_DATE_FORCE:-}" ]]; then
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
  echo "[auto] START_DAYS_BACK=${START_DAYS_BACK} → START_DATE=${START_DATE}"
fi

# Sanity: ensure start_date <= asof - valid_days
python3 - <<PY
from datetime import datetime, timedelta, sys
asof=datetime.strptime("$ASOF","%Y-%m-%d").date()
start=datetime.strptime("$START_DATE","%Y-%m-%d").date()
cutoff=asof - timedelta(days=int("$VALID_DAYS"))
assert start <= cutoff, f"START_DATE {start} must be <= ASOF-VALID_DAYS {cutoff}"
print("[auto] Train window:", start, "→", cutoff)
PY

# ── Derived dirs ───────────────────────────────────────────────────────────────
SWEEP_ROOT="$OUT_BASE/sweeps/$ASOF/$TAG"
CONFIRM_OUT="$OUT_BASE/confirm_${ASOF}_${TAG}"

# ── Cleanup helpers ───────────────────────────────────────────────────────────
clean_dir() {
  local path="$1"
  case "$CLEAN_MODE" in
    auto)
      [[ -d "$path" ]] && { echo "[auto] Cleaning: $path"; rm -rf "$path"; }
      ;;
    prompt)
      if [[ -d "$path" ]]; then
        read -p "[auto] Found existing '$path'. Delete and rerun? [y/N] " REPLY
        [[ "$REPLY" =~ ^[Yy]$ ]] && rm -rf "$path" || { echo "[auto] Aborting."; exit 1; }
      fi
      ;;
    skip)
      echo "[auto] CLEAN_MODE=skip → not deleting existing '$path'. (May mix results.)"
      ;;
    *)
      echo "[auto] Unknown CLEAN_MODE='$CLEAN_MODE' (use auto|prompt|skip)."; exit 1;;
  esac
}

clean_dir "$SWEEP_ROOT";   mkdir -p "$SWEEP_ROOT"
clean_dir "$CONFIRM_OUT";  mkdir -p "$CONFIRM_OUT"

# ── Threading for Polars ──────────────────────────────────────────────────────
export POLARS_MAX_THREADS

# ── TRAIN ─────────────────────────────────────────────────────────────────────
echo "=== TRAIN ${ASOF} ==="
python3 "$ML_DIR/train_price_trend.py" \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --horizon-secs 120 \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --device cuda

# ── Build optional sampling args (passed to simulator via sim_sweep) ──────────
SIM_EXTRA_ARGS=()
[[ -n "$MAX_FILES_PER_DAY" ]] && SIM_EXTRA_ARGS+=( --max-files-per-day "$MAX_FILES_PER_DAY" )
[[ -n "$FILE_SAMPLE_MODE"  ]] && SIM_EXTRA_ARGS+=( --file-sample-mode "$FILE_SAMPLE_MODE" )
[[ -n "$ROW_SAMPLE_SECS"   ]] && SIM_EXTRA_ARGS+=( --row-sample-secs "$ROW_SAMPLE_SECS" )

# ── ADAPTIVE SWEEP (homes in on edge to hit ~1–2k trades/day) ─────────────────
echo "=== SWEEP (adaptive edge; realised ROI objective) ==="
python3 "$ML_DIR/sim_sweep.py" \
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
  $( [[ "$SAMPLE_EV_DENSITY" == "1" ]] && echo --sample-ev-density ) \
  --ev-hist-bins "$EV_HIST_BINS" \
  --tag "$TAG" \
  "${SIM_EXTRA_ARGS[@]}"

TRIALS="$SWEEP_ROOT/trials.csv"
BEST="$SWEEP_ROOT/best_config.json"

if [[ ! -f "$BEST" ]]; then
  echo "[auto] ERROR: best_config.json not produced. See $SWEEP_ROOT/*/stderr.log" >&2
  exit 2
fi

echo "=== BEST (pre-confirm) ==="
sed -e 's/^/  /' "$BEST"

# ── Guardrails ─────────────────────────────────────────────────────────────────
BEST_OK="$(python3 - <<PY
import json
cfg=json.load(open("$BEST"))
n=cfg.get("n_trades",0)
roi=cfg.get("overall_roi_real_mtm", cfg.get("overall_roi_exp", -1e9))
ok = (n >= int("$MIN_TRADES")) and (n <= int("$MAX_TRADES")) and (roi >= float("$ROI_TARGET_MIN"))
print("YES" if ok else "NO")
PY
)"
if [[ "$BEST_OK" != "YES" ]]; then
  echo "[auto] Best config failed guardrails (min/max trades or ROI). Review $TRIALS" >&2
  exit 3
fi

# ── CONFIRM WINNER ─────────────────────────────────────────────────────────────
echo "=== CONFIRM RUN ==="
python3 "$ML_DIR/simulate_stream.py" \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" --start-date "$START_DATE" --valid-days "$VALID_DAYS" \
  --sport "$SPORT" --preoff-max "$PREOFF_MAX" --horizon-secs 120 --commission "$COMMISSION" \
  --model-path "$MODEL_PATH" \
  --edge-thresh "$(jq -r .edge_thresh "$BEST")" \
  --stake-mode "$(jq -r .stake_mode "$BEST")" --bankroll-nom 5000 --kelly-cap 0.02 --kelly-floor 0.001 \
  --odds-min "$(jq -r .odds_min "$BEST")" --odds-max "$(jq -r .odds_max "$BEST")" \
  $(jq -r '.enforce_liquidity_effective|if .==true then "--enforce-liquidity" else "" end' "$BEST") \
  --min-fill-frac "$(jq -r .min_fill_frac "$BEST")" \
  --per-market-topk "$(jq -r .per_market_topk "$BEST")" \
  --per-market-budget "$(jq -r .per_market_budget "$BEST")" \
  --exit-on-move-ticks "$(jq -r .exit_on_move_ticks "$BEST")" \
  --ev-scale "$(jq -r .ev_scale_used "$BEST")" \
  --ev-cap "$(jq -r .ev_cap "$BEST")" \
  --device "$SIM_DEVICE" \
  --output-dir "$CONFIRM_OUT" \
  "${SIM_EXTRA_ARGS[@]}"

echo "=== CONFIRM SUMMARY ==="
sed -e 's/^/  /' "$CONFIRM_OUT/summary_${ASOF}.json"

echo "[auto] Done."
echo "  Train window:   ${START_DATE} → $(python3 - <<PY
from datetime import datetime, timedelta
asof=datetime.strptime("$ASOF","%Y-%m-%d").date()
print((asof - timedelta(days=int("$VALID_DAYS"))).strftime("%Y-%m-%d"))
PY
)"
echo "  Trials:         $TRIALS"
echo "  Best config:    $BEST"
echo "  Confirm summary:$CONFIRM_OUT/summary_${ASOF}.json"
