#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export CURATED_ROOT=/mnt/nvme/betfair-curated
#   ./train_price_trend/bin/tune_xgb_price_trend.sh [ASOF]
#
# Notes:
# - Uses your existing train script to keep data handling identical.
# - Parses the line: [trend] valid EV per £1: mean=...   to rank models
# - Saves best model as usual at output/models/xgb_trend_reg.json

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$BASE_DIR/ml"
ASOF="${1:-$(date -d 'yesterday' +%Y-%m-%d 2>/dev/null || python3 - <<'PY'
from datetime import datetime, timedelta
print((datetime.utcnow().date() - timedelta(days=1)).strftime("%Y-%m-%d"))
PY
)}"

# Fixed data window (override if you want)
START_DATE="${START_DATE_FORCE:-2025-09-05}"
VALID_DAYS="${VALID_DAYS:-7}"
SPORT="${SPORT:-horse-racing}"
PREOFF_MAX="${PREOFF_MAX:-30}"
COMMISSION="${COMMISSION:-0.02}"
DEVICE="${TRAIN_DEVICE:-cuda}"

: "${CURATED_ROOT:?Please export CURATED_ROOT (e.g. /mnt/nvme/betfair-curated)}"

# Modest, sensible grid (keeps runtime short but useful)
DEPTHS="${XGB_DEPTHS:-5 6 7}"
N_EST="${XGB_N_EST:-400 600}"
ETAS="${XGB_ETAS:-0.05 0.10}"
MCW="${XGB_MIN_CHILD_WEIGHT:-1.0 2.0}"
SUBS="${XGB_SUBSAMPLE:-0.7 0.9}"
COLS="${XGB_COLSAMPLE:-0.7 0.9}"
L2S="${XGB_L2:-1.0 2.0}"
L1S="${XGB_L1:-0.0 0.5}"

LOG_DIR="$BASE_DIR/output/tune/$ASOF"
mkdir -p "$LOG_DIR"

best_score="-inf"
best_cfg=""

rank() {
  awk -F'=' '/\[trend\] valid EV per £1: mean=/{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}' | tail -1
}

i=0
for d in $DEPTHS; do
  for n in $N_EST; do
    for eta in $ETAS; do
      for mc in $MCW; do
        for sub in $SUBS; do
          for col in $COLS; do
            for l2 in $L2S; do
              for l1 in $L1S; do
                i=$((i+1))
                tag="d${d}_n${n}_eta${eta}_mc${mc}_sub${sub}_col${col}_l2${l2}_l1${l1}"
                echo "[tune] $i) $tag"
                out="$LOG_DIR/${tag}.log"

                # Train once with this config
                python3 -u "$ML_DIR/train_price_trend.py" \
                  --curated "$CURATED_ROOT" \
                  --asof "$ASOF" \
                  --start-date "$START_DATE" \
                  --valid-days "$VALID_DAYS" \
                  --sport "$SPORT" \
                  --horizon-secs 120 \
                  --preoff-max "$PREOFF_MAX" \
                  --commission "$COMMISSION" \
                  --device "$DEVICE" \
                  --xgb-max-depth "$d" \
                  --xgb-n-estimators "$n" \
                  --xgb-learning-rate "$eta" \
                  --xgb-min-child-weight "$mc" \
                  --xgb-subsample "$sub" \
                  --xgb-colsample-bytree "$col" \
                  --xgb-reg-lambda "$l2" \
                  --xgb-reg-alpha "$l1" \
                  --xgb-early-stopping-rounds 50 \
                  | tee "$out"

                score="$(rank < "$out")"
                if [[ -z "$score" ]]; then score="-inf"; fi
                echo "    → valid_EV_per_£1=${score}"

                # Track best by validation EV (you can switch to ROI if you log it)
                python3 - <<PY
import math
s="$score"
try: v=float(s)
except: v=float("-inf")
print(v)
PY
                v=$(
                  python3 - <<PY
import math
s="$score"
try: print(float(s))
except: print(float("-inf"))
PY
                )
                # shell float compare via python
                better=$(
                  python3 - <<PY
import math
cur=float("$v"); best=float("$best_score")
print("yes" if cur>best else "no")
PY
                )
                if [[ "$better" == "yes" ]]; then
                  best_score="$v"
                  best_cfg="$tag"
                  cp "$BASE_DIR/output/models/xgb_trend_reg.json" "$LOG_DIR/xgb_${tag}.json" || true
                  echo "    ✔ new best: $best_score  ($best_cfg)"
                fi
              done
            done
          done
        done
      done
    done
  done
done

echo "[tune] DONE. Best by valid EV/£1 = $best_score  ($best_cfg)"
echo "[tune] Best model saved at: $LOG_DIR/xgb_${best_cfg}.json"
echo "[tune] You can copy it over the live model if desired:"
echo "       cp $LOG_DIR/xgb_${best_cfg}.json $BASE_DIR/output/models/xgb_trend_reg.json"
