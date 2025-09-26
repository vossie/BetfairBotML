#!/usr/bin/env bash
set -euo pipefail

# === Config paths ===
CFG="/opt/BetfairBotML/edge_temporal/output/automl_real/best_config.json"
export CURATED_ROOT="/mnt/nvme/betfair-curated"
export CALIB_PATH="/opt/BetfairBotML/edge_temporal/output/automl_real/isotonic.pkl"

# === Parse JSON and export vars ===
export PREOFF_MINS=$(jq -r '.preoff_max_minutes' "$CFG")
export PM_CUTOFF=$(jq -r '.best_params.pm_cutoff' "$CFG")
export EDGE_THRESH=$(jq -r '.best_params.edge_thresh' "$CFG")
export LTP_MIN=$(jq -r '.best_params.ltp_min' "$CFG")
export LTP_MAX=$(jq -r '.best_params.ltp_max' "$CFG")
export PER_MARKET_TOPK=1

# === Kelly staking ===
export STAKE="kelly"
export KELLY_CAP=$(jq -r '.kelly_cap' "$CFG")
export KELLY_FLOOR=$(jq -r '.kelly_floor' "$CFG")
export BANKROLL_NOM=$(jq -r '.bankroll_nom' "$CFG")

# === Optional: commission ===
# export COMMISSION=$(jq -r '.commission' "$CFG")

# === Run training with best params ===
ASOF=$(jq -r '.asof' "$CFG")
echo "[run_best.sh] Running Kelly mode with tuned params..."
/opt/BetfairBotML/edge_temporal/bin/train_edge_temporal.sh "$ASOF"

# === Optional: flat staking comparison ===
echo "[run_best.sh] Running flat staking comparison..."
export STAKE="flat"
/opt/BetfairBotML/edge_temporal/bin/train_edge_temporal.sh "$ASOF"
