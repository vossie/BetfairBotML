#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# train_edge_temporal.sh (LOCAL FS, CUDA default)
# Usage:
#   train_edge_temporal.sh START_DATE [SPORT] [PREOFF_MINS] [DOWNSAMPLE_SECS]
#
# Example:
#   train_edge_temporal.sh 2025-09-05 horse-racing 30 5
#
# Env overrides:
#   CURATED_ROOT   (default /mnt/nvme/betfair-curated)
#   COMMISSION     (default 0.02)
#   EDGE_THRESH    (default 0.005)
#   PM_HORIZON_SECS (default 60)
#   PM_TICK_THRESHOLD (default 1)
#   EDGE_PROB      (raw|cal, default raw)
#   NO_SUM_TO_ONE  (1 to disable sum-to-one; default 1)
#   PM_SLACK_SECS  (default 3)
#
# Relies on:
#   /opt/BetfairBotML/edge_temporal/ml/train_edge_temporal.py
# ------------------------------------------------------------

ML_ROOT="/opt/BetfairBotML/edge_temporal"
ML_PY="${ML_ROOT}/ml/train_edge_temporal.py"

CURATED_ROOT="${CURATED_ROOT:-/mnt/nvme/betfair-curated}"
SPORT="${2:-horse-racing}"
PREOFF_MINS="${3:-30}"
DOWNSAMPLE_SECS="${4:-5}"

COMMISSION="${COMMISSION:-0.02}"
EDGE_THRESH="${EDGE_THRESH:-0.005}"
PM_HORIZON_SECS="${PM_HORIZON_SECS:-60}"
PM_TICK_THRESHOLD="${PM_TICK_THRESHOLD:-1}"
PM_SLACK_SECS="${PM_SLACK_SECS:-3}"

EDGE_PROB="${EDGE_PROB:-raw}"
NO_SUM_TO_ONE="${NO_SUM_TO_ONE:-1}"   # 1 disables sum-to-one in backtest

# ---- args ----
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 START_DATE [SPORT] [PREOFF_MINS] [DOWNSAMPLE_SECS]" >&2
  exit 1
fi
START_DATE="$1"

# ---- require local curated path ----
if [[ ! -d "${CURATED_ROOT}" ]]; then
  echo "ERROR: CURATED_ROOT not found: ${CURATED_ROOT}" >&2
  exit 2
fi

# ---- date math (Europe/London) ----
export TZ="Europe/London"
ASOF="$(date +%F)"                 # today's date
TRAIN_END="$(date -d "${ASOF} -2 day" +%F)"
VALID0="$(date -d "${ASOF} -1 day" +%F)"
VALID1="${ASOF}"

# sanity check
if [[ "$(date -d "${START_DATE}" +%s)" -gt "$(date -d "${TRAIN_END}" +%s)" ]]; then
  echo "ERROR: START_DATE (${START_DATE}) must be on or before TRAIN_END (${TRAIN_END})." >&2
  exit 3
fi

# inclusive train-days
epoch_s() { date -d "$1" +%s; }
SECS_START="$(epoch_s "${START_DATE}")"
SECS_END="$(epoch_s "${TRAIN_END}")"
TRAIN_DAYS=$(( (SECS_END - SECS_START) / 86400 + 1 ))

# ---- info banner ----
echo "=== Edge Temporal Training (LOCAL) ==="
echo "Curated root:         ${CURATED_ROOT}"
echo "ASOF (today):         ${ASOF}"
echo "Validation window:    ${VALID0} .. ${VALID1}"
echo "Training window:      ${START_DATE} .. ${TRAIN_END}"
echo "Computed TRAIN_DAYS:  ${TRAIN_DAYS}"
echo "Sport:                ${SPORT}"
echo "Pre-off minutes:      ${PREOFF_MINS}"
echo "Downsample (secs):    ${DOWNSAMPLE_SECS}"
echo "Commission:           ${COMMISSION}"
echo "Edge threshold:       ${EDGE_THRESH}"
echo "PM horizon (secs):    ${PM_HORIZON_SECS}"
echo "PM tick threshold:    ${PM_TICK_THRESHOLD}"
echo "PM slack (secs):      ${PM_SLACK_SECS}"
echo "Edge prob:            ${EDGE_PROB}"
echo "Sum-to-one:           $([[ "${NO_SUM_TO_ONE}" == "1" ]] && echo "disabled" || echo "enabled")"
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
  --device cuda
