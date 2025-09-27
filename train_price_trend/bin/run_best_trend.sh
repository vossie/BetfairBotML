#!/usr/bin/env bash
set -euo pipefail

# ---------- ASOF date (defaults to yesterday) ----------
if [[ $# -ge 1 ]]; then
  ASOF="$1"
  TAG="${2:-automl}"
else
  # Linux (GNU date)
  if date -d "yesterday" +%Y-%m-%d >/dev/null 2>&1; then
    ASOF="$(date -d "yesterday" +%Y-%m-%d)"
  # macOS / BSD date
  elif date -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    ASOF="$(date -v-1d +%Y-%m-%d)"
  # GNU coreutils 'gdate' (mac via brew)
  elif command -v gdate >/dev/null 2>&1; then
    ASOF="$(gdate -d "yesterday" +%Y-%m-%d)"
  else
    echo "ERROR: can't compute 'yesterday' (need GNU date/BSD date). Provide ASOF explicitly." >&2
    echo "Usage: $0 YYYY-MM-DD [TAG]" >&2
    exit 1
  fi
  TAG="automl"
fi

CURATED_ROOT="${CURATED_ROOT:?must set CURATED_ROOT}"

# Optional: check for jq (required to parse best_config.json)
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: 'jq' is required but not found. Please install jq." >&2
  exit 1
fi

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$BIN_DIR/.." && pwd)"
OUTDIR="${OUTDIR:-$BASE_DIR/output}"
BEST_CFG="$OUTDIR/automl/$ASOF/$TAG/best_config.json"

if [[ ! -f "$BEST_CFG" ]]; then
  echo "Best config not found at $BEST_CFG"
  echo "Run AutoML first, e.g.:"
  echo "  python3 $BASE_DIR/ml/train_price_trend_automl.py --curated \$CURATED_ROOT --asof $ASOF --start-date 2025-09-05"
  exit 1
fi

edge=$(jq -r '.edge_thresh' "$BEST_CFG")
stake=$(jq -r '.stake_mode' "$BEST_CFG")
odds_min=$(jq -r '.odds_min' "$BEST_CFG")
odds_max=$(jq -r '.odds_max' "$BEST_CFG")
liq_enf=$(jq -r '.enforce_liquidity' "$BEST_CFG")
liq_levels=$(jq -r '.liquidity_levels' "$BEST_CFG")

export START_DATE=$(jq -r '.start_date' "$BEST_CFG")
export VALID_DAYS=$(jq -r '.valid_days' "$BEST_CFG")
export SPORT=$(jq -r '.sport' "$BEST_CFG")
export PREOFF_MAX=$(jq -r '.preoff_max' "$BEST_CFG")
export HORIZON_SECS=$(jq -r '.horizon_secs' "$BEST_CFG")
export COMMISSION=$(jq -r '.commission' "$BEST_CFG")
export DEVICE=$(jq -r '.device' "$BEST_CFG")
export EV_MODE=$(jq -r '.ev_mode' "$BEST_CFG")
export BANKROLL_NOM=$(jq -r '.bankroll_nom' "$BEST_CFG")
export KELLY_CAP=$(jq -r '.kelly_cap' "$BEST_CFG")
export KELLY_FLOOR=$(jq -r '.kelly_floor' "$BEST_CFG")

export EDGE_THRESH="$edge"
export STAKE="$stake"

# Optional odds band
if [[ "$odds_min" != "null" ]]; then export ODDS_MIN="$odds_min"; fi
if [[ "$odds_max" != "null" ]]; then export ODDS_MAX="$odds_max"; fi

# Liquidity flags
if [[ "$liq_enf" == "true" ]]; then
  export ENFORCE_LIQUIDITY=1
else
  export ENFORCE_LIQUIDITY=0
fi
export LIQUIDITY_LEVELS="$liq_levels"

# Run the main script (it also defaults ASOF to yesterday if omitted, but we pass it explicitly)
"$BIN_DIR/train_price_trend.sh" "$ASOF"
