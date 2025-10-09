#!/usr/bin/env bash
set -euo pipefail

# ── Resolve base paths ─────────────────────────────────────────────────────────
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$BASE_DIR/ml"
OUT_BASE="$BASE_DIR/output/stream"
MODEL_PATH="$BASE_DIR/output/models/xgb_trend_reg.json"

# ── Inputs ─────────────────────────────────────────────────────────────────────
: "${CURATED_ROOT:?Please export CURATED_ROOT (e.g. /mnt/nvme/betfair-curated)}"

# ASOF: arg1 or default to yesterday (robust to systems without GNU date)
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  ASOF="$1"
else
  ASOF="$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || python3 - <<'PY'
from datetime import datetime, timedelta
print((datetime.utcnow().date() - timedelta(days=1)).strftime("%Y-%m-%d"))
PY
)"
fi
if [[ -z "${ASOF}" ]]; then
  echo "[auto] ERROR: ASOF could not be determined." >&2
  exit 1
fi

# ── Config (override via env) ──────────────────────────────────────────────────
START_DAYS_BACK="${START_DAYS_BACK:-34}"
VALID_DAYS="${VALID_DAYS:-7}"
USE_EARLIEST="${USE_EARLIEST:-1}"               # default ON per your setup
EARLIEST_DATE="${EARLIEST_DATE:-2025-09-05}"
SPORT="${SPORT:-horse-racing}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"
SIM_DEVICE="${SIM_DEVICE:-cpu}"
TAG="${TAG:-auto}"
CLEAN_MODE="${CLEAN_MODE:-skip}"   # default to skip so we never block runs
TRASH_DIR="$OUT_BASE/.trash"
POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-$(nproc || echo 8)}"

# Optional sampling (safe to leave empty)
MAX_FILES_PER_DAY="${MAX_FILES_PER_DAY:-}"
FILE_SAMPLE_MODE="${FILE_SAMPLE_MODE:-}"
ROW_SAMPLE_SECS="${ROW_SAMPLE_SECS:-}"

# Guardrails
MIN_TRADES="${MIN_TRADES:-5000}"
MAX_TRADES="${MAX_TRADES:-250000}"
ROI_TARGET_MIN="${ROI_TARGET_MIN:-0.0}"

# Adaptive sweep knobs (used by your adaptive sim_sweep.py)
TPD_MIN="${TPD_MIN:-1000}"
TPD_MAX="${TPD_MAX:-2000}"
EDGE_MIN="${EDGE_MIN:-0.0001}"
EDGE_MAX="${EDGE_MAX:-0.005}"
EDGE_INIT_GRID="${EDGE_INIT_GRID:-0.0005,0.001,0.002}"
MAX_EDGE_ITER="${MAX_EDGE_ITER:-6}"

# Non-edge grids
STAKE_MODES_GRID="${STAKE_MODES_GRID:-flat,kelly}"
ODDS_BANDS_GRID="${ODDS_BANDS_GRID:-1.5:5.0,2.2:3.6}"
EV_SCALE_GRID="${EV_SCALE_GRID:-0.05,0.1,0.2}"
EV_CAP_GRID="${EV_CAP_GRID:-0.05}"
EXIT_TICKS_GRID="${EXIT_TICKS_GRID:-0,1}"
TOPK_GRID="${TOPK_GRID:-1}"
BUDGET_GRID="${BUDGET_GRID:-5,10}"
LIQ_ENFORCE_GRID="${LIQ_ENFORCE_GRID:-1}"
MIN_FILL_FRAC_GRID="${MIN_FILL_FRAC_GRID:-5.0}"

SAMPLE_EV_DENSITY="${SAMPLE_EV_DENSITY:-1}"
EV_HIST_BINS="${EV_HIST_BINS:-0,0.0005,0.001,0.002,0.003,0.005,0.01}"

# ── Compute START_DATE ─────────────────────────────────────────────────────────
if [[ -n "${START_DATE_FORCE:-}" ]]; then
  START_DATE="$START_DATE_FORCE"
  echo "[auto] Using fixed START_DATE=${START_DATE}"
elif [[ "${USE_EARLIEST}" == "1" ]]; then
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
from datetime import datetime, timedelta
asof=datetime.strptime("$ASOF","%Y-%m-%d").date()
start=datetime.strptime("$START_DATE","%Y-%m-%d").date()
cutoff=asof - timedelta(days=int("$VALID_DAYS"))
assert start <= cutoff, f"START_DATE {start} must be <= ASOF-VALID_DAYS {cutoff}"
print("[auto] Train window:", start, "→", cutoff)
PY

# ── Derived dirs ───────────────────────────────────────────────────────────────
SWEEP_ROOT="$OUT_BASE/sweeps/$ASOF/$TAG"
CONFIRM_OUT="$OUT_BASE/confirm_${ASOF}_${TAG}"
echo "[auto] OUT_BASE=$OUT_BASE"
echo "[auto] SWEEP_ROOT=$SWEEP_ROOT"
echo "[auto] CONFIRM_OUT=$CONFIRM_OUT"

# ── Cleanup helpers ───────────────────────────────────────────────────────────
# Fast cleanup: move old dir to trash (same filesystem → instant), GC old trash dirs
clean_dir() {
  local path="$1"
  case "$CLEAN_MODE" in
    skip)
      echo "[auto] CLEAN_MODE=skip → keeping existing '$path'."
      ;;
    auto|prompt)
      if [[ -d "$path" ]]; then
        if [[ "$CLEAN_MODE" == "prompt" ]]; then
          read -p "[auto] Found '$path'. Move to trash and rerun? [y/N] " R
          [[ "$R" =~ ^[Yy]$ ]] || { echo "[auto] Aborting."; exit 1; }
        fi
        mkdir -p "$TRASH_DIR"
        local stamp; stamp="$(date +%s)"
        local dst="$TRASH_DIR/$(basename "$path").$stamp"
        echo "[auto] Moving '$path' → '$dst'"
        mv "$path" "$dst" || { echo "[auto] mv failed; falling back to rm -rf (may take time)…"; rm -rf "$path"; }
      fi
      ;;
    *)
      echo "[auto] Unknown CLEAN_MODE='$CLEAN_MODE' (use auto|prompt|skip)."; exit 1;;
  esac
}

# Optional: garbage collect trash older than 2 days (safe & quick)
if [[ -d "$TRASH_DIR" ]]; then
  echo "[auto] GC trash >2d in $TRASH_DIR …"
  find "$TRASH_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +2 -exec rm -rf {} + || true
fi

echo "[auto] Ensuring output dirs…"
clean_dir "$SWEEP_ROOT";   mkdir -p "$SWEEP_ROOT"
clean_dir "$CONFIRM_OUT";  mkdir -p "$CONFIRM_OUT"

# ── Threads for Polars ────────────────────────────────────────────────────────
export POLARS_MAX_THREADS
echo "[auto] POLARS_MAX_THREADS=$POLARS_MAX_THREADS"

echo "=== TRAIN ${ASOF} ==="
