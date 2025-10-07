#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Trend sweep runner (grid)
# -----------------------------
# Usage:
#   export CURATED_ROOT=/mnt/nvme/betfair-curated
#   ./train_price_trend/bin/sweep_price_trend.sh 2025-10-06 [tag]
#
# Override grids/params via env, e.g.:
#   EDGE_THRESH_GRID="0.001,0.002" STAKE_MODES_GRID="flat,kelly" ODDS_BANDS_GRID="1.5:5.0,none" \
#   ./train_price_trend/bin/sweep_price_trend.sh 2025-10-06 mygrid
#
# Requires:
#   - trading-aware simulate_stream.py in ml/
#   - grid-capable sim_sweep.py in ml/ (will call simulate_stream.py for each combo)
# -----------------------------

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$BASE_DIR/ml"
OUT_BASE="$BASE_DIR/output/stream"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ASOF [TAG]" >&2
  exit 1
fi

ASOF="${1}"
TAG="${2:-grid1}"

# --- required env ---
: "${CURATED_ROOT:?Please export CURATED_ROOT (e.g. /mnt/nvme/betfair-curated)}"

# --- defaults (override via env) ---
START_DATE="${START_DATE:-$(date -d "${ASOF} -31 days" +%Y-%m-%d 2>/dev/null || python3 - <<PY
from datetime import datetime, timedelta
asof=datetime.strptime("${ASOF}","%Y-%m-%d").date()
print((asof - timedelta(days=31)).strftime("%Y-%m-%d"))
PY
)}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"
SIM_DEVICE="${SIM_DEVICE:-cpu}"        # simulate_stream device (cpu/cuda); sweep itself is lightweight
EV_MODE="${EV_MODE:-mtm}"

# Model path (produced by train step)
MODEL_PATH="${MODEL_PATH:-$BASE_DIR/output/models/xgb_trend_reg.json}"

# Grids
EDGE_THRESH_GRID="${EDGE_THRESH_GRID:-0.0005,0.001,0.002,0.003,0.005}"
STAKE_MODES_GRID="${STAKE_MODES_GRID:-flat,kelly}"
ODDS_BANDS_GRID="${ODDS_BANDS_GRID:-none,2.2:3.6,1.5:5.0}"

# Trading params
BANKROLL_NOM="${BANKROLL_NOM:-5000}"
KELLY_CAP="${KELLY_CAP:-0.02}"
KELLY_FLOOR="${KELLY_FLOOR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-200000}"
MIN_TRADES="${MIN_TRADES:-10000}"

# Perf knobs for child sims
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-0}"
export XGB_FORCE_NTHREADS="${XGB_FORCE_NTHREADS:-0}"

echo "=== Trend Sweep ==="
printf "ASOF:            %s\n" "$ASOF"
printf "Start date:      %s\n" "$START_DATE"
printf "Valid days:      %s\n" "$VALID_DAYS"
printf "Pre-off max (m): %s\n" "$PREOFF_MAX"
printf "Device:          %s\n" "$SIM_DEVICE"
printf "EV mode:         %s\n" "$EV_MODE"
printf "Grids: EDGE=[%s]  STAKE=[%s]  ODDS=[%s]\n" "$EDGE_THRESH_GRID" "$STAKE_MODES_GRID" "$ODDS_BANDS_GRID"
printf "Output base:     %s\n" "$OUT_BASE"

# Ensure output root exists
mkdir -p "$OUT_BASE"

# Kick off the sweep (Python orchestrates all combos, calling simulate_stream.py per run)
python3 "$ML_DIR/sim_sweep.py" \
  --curated "$CURATED_ROOT" \
  --asof "$ASOF" \
  --start-date "$START_DATE" \
  --valid-days "$VALID_DAYS" \
  --sport "$SPORT" \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --device "$SIM_DEVICE" \
  --ev-mode "$EV_MODE" \
  --model-path "$MODEL_PATH" \
  --base-output-dir "$OUT_BASE" \
  --bankroll-nom "$BANKROLL_NOM" \
  --kelly-cap "$KELLY_CAP" \
  --kelly-floor "$KELLY_FLOOR" \
  --batch-size "$BATCH_SIZE" \
  --min-trades "$MIN_TRADES" \
  --edge-thresh "$EDGE_THRESH_GRID" \
  --stake-modes "$STAKE_MODES_GRID" \
  --odds-bands "$ODDS_BANDS_GRID" \
  --tag "$TAG"

TRIALS="$OUT_BASE/sweeps/$ASOF/$TAG/trials.csv"
BEST="$OUT_BASE/sweeps/$ASOF/$TAG/best_config.json"

if [[ -f "$TRIALS" ]]; then
  echo "[sweep] Wrote $TRIALS"
else
  echo "[sweep] Trials CSV not found at $TRIALS (check logs in run subfolders)" >&2
fi

if [[ -f "$BEST" ]]; then
  echo "[sweep] Best config → $BEST"
else
  echo "[sweep] No best_config.json produced." >&2
fi
