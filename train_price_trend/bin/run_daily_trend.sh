#!/usr/bin/env bash
set -euo pipefail

# === Daily end-to-end: train → constrained sweep → pick best by realised ROI → confirm run ===
# Usage:
#   export CURATED_ROOT=/mnt/nvme/betfair-curated
#   ./train_price_trend/bin/run_daily_trend.sh [ASOF]
# If ASOF is omitted, it defaults to **yesterday** (YYYY-MM-DD).

# ---------------- Paths ----------------
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$BASE_DIR/ml"
OUT_BASE="$BASE_DIR/output/stream"

# ---------------- Inputs ----------------
: "${CURATED_ROOT:?Please export CURATED_ROOT (e.g. /mnt/nvme/betfair-curated)}"

# If ASOF not supplied, default to yesterday
if [[ $# -ge 1 ]]; then
  ASOF="$1"
else
  ASOF="$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || python3 - <<'PY'
from datetime import datetime, timedelta
print((datetime.utcnow().date() - timedelta(days=1)).strftime("%Y-%m-%d"))
PY
)"
fi

# ---------------- Defaults you can override via env ----------------
START_DAYS_BACK="${START_DAYS_BACK:-31}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"
SIM_DEVICE="${SIM_DEVICE:-cpu}"
TAG="${TAG:-auto}"
CLEAN_MODE="${CLEAN_MODE:-auto}"   # auto | prompt | skip

# Sweep grids (realistic constraints)
EDGE_THRESH_GRID="${EDGE_THRESH_GRID:-0.00015,0.00025,0.00035,0.0005,0.0008,0.0012}"
EV_SCALE_GRID="${EV_SCALE_GRID:-0.03,0.05,0.08,0.12,0.2,0.3}"
EV_CAP_GRID="${EV_CAP_GRID:-0.03,0.05}"
STAKE_MODES_GRID="${STAKE_MODES_GRID:-flat,kelly}"
ODDS_BANDS_GRID="${ODDS_BANDS_GRID:-1.5:5.0,2.2:3.6}"
EXIT_TICKS_GRID="${EXIT_TICKS_GRID:-0,1,2}"
TOPK_GRID="${TOPK_GRID:-1}"
BUDGET_GRID="${BUDGET_GRID:-5,10}"
LIQ_ENFORCE_GRID="${LIQ_ENFORCE_GRID:-1}"
MIN_FILL_FRAC_GRID="${MIN_FILL_FRAC_GRID:-5.0}"

# Guardrails
MIN_TRADES="${MIN_TRADES:-5000}"
MAX_TRADES="${MAX_TRADES:-250000}"
ROI_TARGET_MIN="${ROI_TARGET_MIN:-0.0}"

# ---------------- Derived dates/paths ----------------
START_DATE="$(python3 - <<PY
from datetime import datetime, timedelta
asof=datetime.strptime("$ASOF","%Y-%m-%d").date()
print((asof - timedelta(days=int("$START_DAYS_BACK"))).strftime("%Y-%m-%d"))
PY
)"

MODEL_PATH="$BASE_DIR/output/models/xgb_trend_reg.json"
SWEEP_ROOT="$OUT_BASE/sweeps/$ASOF/$TAG"
CONFIRM_OUT="$OUT_BASE/confirm_${ASOF}_${TAG}"

# ---------------- Same-day cleanup (sweep + confirm) ----------------
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

# ---------------- TRAIN ----------------
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

# ---------------- SWEEP (realised ROI objective) ----------------
echo "=== SWEEP (realised ROI) ==="
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
  --batch-size 200000 \
  --min-trades "$MIN_TRADES" \
  --edge-thresh "$EDGE_THRESH_GRID" \
  --stake-modes "$STAKE_MODES_GRID" \
  --odds-bands "$ODDS_BANDS_GRID" \
  --ev-scale-grid "$EV_SCALE_GRID" \
  --ev-cap-grid "$EV_CAP_GRID" \
  --exit-ticks-grid "$EXIT_TICKS_GRID" \
  --topk-grid "$TOPK_GRID" \
  --budget-grid "$BUDGET_GRID" \
  --liq-enforce-grid "$LIQ_ENFORCE_GRID" \
  --min-fill-frac-grid "$MIN_FILL_FRAC_GRID" \
  --tag "$TAG"

TRIALS="$SWEEP_ROOT/trials.csv"
BEST="$SWEEP_ROOT/best_config.json"

if [[ ! -f "$BEST" ]]; then
  echo "[auto] ERROR: best_config.json not produced. See $SWEEP_ROOT/*/stderr.log" >&2
  exit 2
fi

echo "=== BEST (pre-confirm) ==="
sed -e 's/^/  /' "$BEST"

# ---------------- Guardrails ----------------
BEST_OK="$(python3 - <<PY
import json
cfg=json.load(open("$BEST"))
n=cfg.get("n_trades",0)
roi=cfg.get("overall_roi_real_mtm", cfg.get("overall_roi", -1e9))
ok = (n >= int("$MIN_TRADES")) and (n <= int("$MAX_TRADES")) and (roi >= float("$ROI_TARGET_MIN"))
print("YES" if ok else "NO")
PY
)"
if [[ "$BEST_OK" != "YES" ]]; then
  echo "[auto] Best config failed guardrails (min/max trades or ROI). Review $TRIALS" >&2
  exit 3
fi

# ---------------- Confirmatory re-run on the winner ----------------
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
  --output-dir "$CONFIRM_OUT"

echo "=== CONFIRM SUMMARY ==="
sed -e 's/^/  /' "$CONFIRM_OUT/summary_${ASOF}.json"

echo "[auto] Done."
echo "  Trials:   $TRIALS"
echo "  Best:     $BEST"
echo "  Confirm:  $CONFIRM_OUT/summary_${ASOF}.json"
