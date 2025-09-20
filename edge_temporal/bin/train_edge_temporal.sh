#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# train_edge_temporal.sh  — local FS, CUDA default
#
# Usage:
#   train_edge_temporal.sh START_DATE [SPORT] [PREOFF_MINS] [DOWNSAMPLE_SECS]
#
# Example:
#   VALID_DAYS=3 \
#   PER_MARKET_TOPK=1 EDGE_THRESH=0.015 \
#   LTP_MIN=1.5 LTP_MAX=5.0 PM_HORIZON_SECS=300 PM_TICK_THRESHOLD=1 \
#   /opt/BetfairBotML/edge_temporal/bin/train_edge_temporal.sh 2025-09-05
#
# Notes:
# - START_DATE is the first training day.
# - We exclude today from validation. ASOF = today - 1 day.
#   So: valid = [ASOF-(VALID_DAYS-1), ASOF], train ends at ASOF-VALID_DAYS.
# - The Python trainer loads data ONCE, trains ONCE, then runs a RAM-only sweep.
#   You can narrow/widen the sweep via env vars (see SWEEP_* below).
# ------------------------------------------------------------

# Project paths
ML_ROOT="/opt/BetfairBotML/edge_temporal"
ML_PY="${ML_ROOT}/ml/train_edge_temporal.py"

# Data root (local — faster than MinIO)
CURATED_ROOT="${CURATED_ROOT:-/mnt/nvme/betfair-curated}"

# Positional args (with defaults for convenience)
SPORT="${2:-horse-racing}"
PREOFF_MINS="${3:-30}"
DOWNSAMPLE_SECS="${4:-5}"

# ---- Tunable defaults ----
COMMISSION="${COMMISSION:-0.02}"
EDGE_THRESH="${EDGE_THRESH:-0.015}"
EDGE_PROB="${EDGE_PROB:-cal}"
NO_SUM_TO_ONE="${NO_SUM_TO_ONE:-0}"          # 1 disables normalization; 0 enables
MARKET_PROB="${MARKET_PROB:-overround}"      # 'overround' recommended
PER_MARKET_TOPK="${PER_MARKET_TOPK:-1}"      # #picks per market (across sides)
SIDE="${SIDE:-back}"                         # back | lay | both

# Odds range focus
LTP_MIN="${LTP_MIN:-1.5}"
LTP_MAX="${LTP_MAX:-5.0}"

# Staking
STAKE="${STAKE:-flat}"                       # flat | kelly
KELLY_CAP="${KELLY_CAP:-0.05}"               # max Kelly fraction (if STAKE=kelly)
KELLY_FLOOR="${KELLY_FLOOR:-0.0}"            # minimum Kelly fraction floor
BANKROLL_NOM="${BANKROLL_NOM:-1000}"

# Price-move head / gating
PM_HORIZON_SECS="${PM_HORIZON_SECS:-300}"
PM_TICK_THRESHOLD="${PM_TICK_THRESHOLD:-1}"
PM_SLACK_SECS="${PM_SLACK_SECS:-3}"
PM_CUTOFF="${PM_CUTOFF:-0.0}"

# Device
DEVICE="${DEVICE:-cuda}"

# Validation window length (days)
VALID_DAYS="${VALID_DAYS:-3}"

# Optional: output dir passthrough for Python
OUTPUT_DIR="${OUTPUT_DIR:-${ML_ROOT}/output}"

# Optional RAM-only sweep grids (semicolon-separated); defaults are conservative
#   SWEEP_EDGE_THRESH="0.010;0.012;0.015;0.018"
#   SWEEP_PM_CUTOFF="0.60;0.65;0.70"
#   SWEEP_TOPK="1;2"
#   SWEEP_ODDS_WINDOWS="1.5-5.0;1.8-6.0"

# ---- args ----
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 START_DATE [SPORT] [PREOFF_MINS] [DOWNSAMPLE_SECS]" >&2
  exit 1
fi
START_DATE="$1"

# ---- basic checks ----
if [[ ! -d "${CURATED_ROOT}" ]]; then
  echo "ERROR: CURATED_ROOT not found: ${CURATED_ROOT}" >&2
  exit 2
fi

if ! [[ "${VALID_DAYS}" =~ ^[0-9]+$ ]] || [[ "${VALID_DAYS}" -lt 1 ]]; then
  echo "ERROR: VALID_DAYS must be an integer >= 1 (got '${VALID_DAYS}')." >&2
  exit 2
fi

# ---- date math (Europe/London) ----
export TZ="Europe/London"
TODAY="$(date +%F)"
ASOF="$(date -d "${TODAY} -1 day" +%F)"                         # exclude today
TRAIN_END="$(date -d "${ASOF} -${VALID_DAYS} day" +%F)"         # training ends at ASOF-VALID_DAYS
VALID_START="$(date -d "${ASOF} -$((VALID_DAYS-1)) day" +%F)"
VALID_END="${ASOF}"

if [[ "$(date -d "${START_DATE}" +%s)" -gt "$(date -d "${TRAIN_END}" +%s)" ]]; then
  echo "ERROR: START_DATE (${START_DATE}) must be on or before TRAIN_END (${TRAIN_END})." >&2
  exit 3
fi

# inclusive train-days = [START_DATE .. TRAIN_END]
epoch_s() { date -d "$1" +%s; }
SECS_START="$(epoch_s "${START_DATE}")"
SECS_END="$(epoch_s "${TRAIN_END}")"
TRAIN_DAYS=$(( (SECS_END - SECS_START) / 86400 + 1 ))

# ---- info banner ----
echo "=== Edge Temporal Training (LOCAL) ==="
echo "Curated root:         ${CURATED_ROOT}"
echo "Today:                ${TODAY}"
echo "ASOF (arg to trainer):${ASOF}   # validation excludes today"
echo "Validation window:    ${VALID_START} .. ${VALID_END}  (${VALID_DAYS} days)"
echo "Training window:      ${START_DATE} .. ${TRAIN_END}"
echo "Computed TRAIN_DAYS:  ${TRAIN_DAYS}"
echo "Sport:                ${SPORT}"
echo "Pre-off minutes:      ${PREOFF_MINS}"
echo "Downsample (secs):    ${DOWNSAMPLE_SECS}"
echo "Commission:           ${COMMISSION}"
echo "Edge threshold:       ${EDGE_THRESH}"
echo "Edge prob:            ${EDGE_PROB}"
echo "Sum-to-one:           $([[ "${NO_SUM_TO_ONE}" == "1" ]] && echo "disabled" || echo "enabled")"
echo "Market prob:          ${MARKET_PROB}"
echo "Per-market topK:      ${PER_MARKET_TOPK}"
echo "Side:                 ${SIDE}"
echo "LTP range:            [${LTP_MIN}, ${LTP_MAX}]"
echo "Stake mode:           ${STAKE} (kelly_cap=${KELLY_CAP}, floor=${KELLY_FLOOR})"
echo "Bankroll (nominal):   £${BANKROLL_NOM}"
echo "PM horizon (secs):    ${PM_HORIZON_SECS}"
echo "PM tick threshold:    ${PM_TICK_THRESHOLD}"
echo "PM slack (secs):      ${PM_SLACK_SECS}"
echo "PM cutoff:            ${PM_CUTOFF}"
echo "Output dir:           ${OUTPUT_DIR}"
echo

# ---- run Python ----
export PYTHONPATH="${ML_ROOT}/ml:${PYTHONPATH:-}"
export OUTPUT_DIR

python3 "${ML_PY}" \
  --curated "${CURATED_ROOT}" \
  --sport "${SPORT}" \
  --asof "${ASOF}" \
  --train-days "${TRAIN_DAYS}" \
  --preoff-mins "${PREOFF_MINS}" \
  --downsample-secs "${DOWNSAMPLE_SECS}" \
  --commission "${COMMISSION}" \
  --edge-thresh "${EDGE_THRESH}" \
  --pm-horizon-secs "${PM_HORIZON_SECS}" \
  --pm-tick-threshold "${PM_TICK_THRESHOLD}" \
  --pm-slack-secs "${PM_SLACK_SECS}" \
  --pm-cutoff "${PM_CUTOFF}" \
  --edge-prob "${EDGE_PROB}" \
  $([[ "${NO_SUM_TO_ONE}" == "1" ]] && echo "--no-sum-to-one") \
  --market-prob "${MARKET_PROB}" \
  --per-market-topk "${PER_MARKET_TOPK}" \
  --stake "${STAKE}" \
  --kelly-cap "${KELLY_CAP}" \
  --kelly-floor "${KELLY_FLOOR}" \
  --ltp-min "${LTP_MIN}" \
  --ltp-max "${LTP_MAX}" \
  --side "${SIDE}" \
  --device "${DEVICE}" \
  --bankroll-nom "${BANKROLL_NOM}"
