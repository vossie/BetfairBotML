#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# train_edge_temporal.sh  — local FS, CUDA default
#
# Usage:
#   train_edge_temporal.sh START_DATE [SPORT] [PREOFF_MINS] [DOWNSAMPLE_SECS]
#
# Example:
#   PER_MARKET_TOPK=1 EDGE_THRESH=0.015 \
#   LTP_MIN=1.5 LTP_MAX=5.0 PM_HORIZON_SECS=300 PM_TICK_THRESHOLD=1 \
#   /opt/BetfairBotML/edge_temporal/bin/train_edge_temporal.sh 2025-09-05
#
# Notes:
# - START_DATE is the first training day.
# - We **exclude today** from validation. ASOF = today - 1 day.
#   So:  valid = [ASOF-1, ASOF], train ends at ASOF-2.
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

# ---- Tunable defaults (reflecting our latest findings) ----
# Trading / backtest
COMMISSION="${COMMISSION:-0.02}"
EDGE_THRESH="${EDGE_THRESH:-0.015}"          # entry cutoff
EDGE_PROB="${EDGE_PROB:-cal}"                # use calibrated probs for edge calc
NO_SUM_TO_ONE="${NO_SUM_TO_ONE:-0}"          # 1 disables normalization; 0 enables

# Market probability comparator
MARKET_PROB="${MARKET_PROB:-overround}"      # 'overround' recommended

# Selection policy and flow control
PER_MARKET_TOPK="${PER_MARKET_TOPK:-1}"      # #picks per market (across sides)
SIDE="${SIDE:-back}"                         # back | lay | both

# Odds range focus
LTP_MIN="${LTP_MIN:-1.5}"
LTP_MAX="${LTP_MAX:-5.0}"

# Staking
STAKE="${STAKE:-flat}"                       # flat | kelly
KELLY_CAP="${KELLY_CAP:-0.05}"               # max Kelly fraction (if STAKE=kelly)

# Price-move head (labels only; not used for value backtest)
PM_HORIZON_SECS="${PM_HORIZON_SECS:-300}"
PM_TICK_THRESHOLD="${PM_TICK_THRESHOLD:-1}"
PM_SLACK_SECS="${PM_SLACK_SECS:-3}"

# Device
DEVICE="${DEVICE:-cuda}"

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

# ---- date math (Europe/London) ----
export TZ="Europe/London"
TODAY="$(date +%F)"
ASOF="$(date -d "${TODAY} -1 day" +%F)"     # exclude today
TRAIN_END="$(date -d "${ASOF} -2 day" +%F)" # training ends at ASOF-2
VALID0="$(date -d "${ASOF} -1 day" +%F)"    # validation: ASOF-1 ..
VALID1="${ASOF}"                             #              .. ASOF

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
echo "Validation window:    ${VALID0} .. ${VALID1}"
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
echo "Stake mode:           ${STAKE} (kelly_cap=${KELLY_CAP})"
echo "PM horizon (secs):    ${PM_HORIZON_SECS}"
echo "PM tick threshold:    ${PM_TICK_THRESHOLD}"
echo "PM slack (secs):      ${PM_SLACK_SECS}"
echo

# ---- run Python ----
export PYTHONPATH="${ML_ROOT}/ml:${PYTHONPATH:-}"

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
  --edge-prob "${EDGE_PROB}" \
  $([[ "${NO_SUM_TO_ONE}" == "1" ]] && echo "--no-sum-to-one") \
  --market-prob "${MARKET_PROB}" \
  --per-market-topk "${PER_MARKET_TOPK}" \
  --stake "${STAKE}" \
  --kelly-cap "${KELLY_CAP}" \
  --ltp-min "${LTP_MIN}" \
  --ltp-max "${LTP_MAX}" \
  --side "${SIDE}" \
  --device "${DEVICE}"
