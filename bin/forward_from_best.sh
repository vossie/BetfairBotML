#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   forward_from_best.sh --model-dir <dir> --curated <root> --start <YYYY-MM-DD> --end <YYYY-MM-DD> [--sport horse-racing] [--output <csv>]
#
# Example:
#   forward_from_best.sh \
#     --model-dir /opt/BetfairBotML/edge_temporal/output/automl_real \
#     --curated /mnt/nvme/betfair-curated \
#     --start 2025-09-19 --end 2025-09-25 \
#     --sport horse-racing

# ---------- arg parse ----------
MODEL_DIR=""
CURATED=""
START_DATE=""
END_DATE=""
SPORT="horse-racing"
OUTPUT_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)     MODEL_DIR="$2"; shift 2;;
    --curated)       CURATED="$2"; shift 2;;
    --start|--start-date) START_DATE="$2"; shift 2;;
    --end|--end-date)     END_DATE="$2"; shift 2;;
    --sport)         SPORT="$2"; shift 2;;
    --output|--output-csv) OUTPUT_CSV="$2"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$MODEL_DIR" || -z "$CURATED" || -z "$START_DATE" || -z "$END_DATE" ]]; then
  echo "Missing required args. See --help." >&2; exit 2
fi

CFG="$MODEL_DIR/best_config.json"
FWD="/opt/BetfairBotML/edge_temporal/ml/forward_test_edge_temporal.py"

if [[ ! -f "$CFG" ]]; then
  echo "best_config.json not found in $MODEL_DIR" >&2; exit 2
fi
if [[ ! -f "$FWD" ]]; then
  echo "forward_test_edge_temporal.py not found at $FWD" >&2; exit 2
fi

# ---------- JSON readers ----------
# jget <jq-filter> <default>
jget() {
  local filt="$1"; local def="${2-}"
  if command -v jq >/dev/null 2>&1; then
    local v
    v=$(jq -er "$filt // empty" "$CFG" 2>/dev/null || true)
    if [[ -z "$v" ]]; then echo "$def"; else echo "$v"; fi
  else
    # python fallback
    python3 - "$CFG" "$filt" "$def" <<'PY'
import json,sys
path = sys.argv[2]
default = sys.argv[3] if len(sys.argv)>3 else ""
with open(sys.argv[1],"r") as f:
    obj = json.load(f)
# very small jsonpath-lite: support .a.b.c
def get_path(o,p):
    cur=o
    for part in p.strip().strip('"').split('.'):
        if not part: continue
        if part.startswith('['):  # ignore complex filters
            return None
        part = part.replace('\\\"','"')
        if part in cur: cur=cur[part]
        else: return None
    return cur
# convert jq-style like ."best_params"."pm_cutoff" -> best_params.pm_cutoff
path = path.replace('."','').replace('"','').replace('[','').replace(']','').replace('// empty','').strip()
path = path.strip('|') if '|' in path else path
val = get_path(obj, path)
print(default if val is None else val)
PY
  fi
}

# ---------- load config (with sensible defaults) ----------
STAKE_MODE=$(jget '.stake_mode' 'flat')
KELLY_CAP=$(jget '.kelly_cap' '0.05')
KELLY_FLOOR=$(jget '.kelly_floor' '0.002')
BANKROLL=$(jget '.bankroll_nom' '5000')
PREOFF_MAX=$(jget '.preoff_max_minutes' '180')
COMMISSION=$(jget '.commission' '0.02')
DOWNSEC=$(jget '.downsample_secs' '0')
DEDUP=$(jget '.dedupe_mode' 'none')
HAIRCUT=$(jget '.haircut_ticks' '0')

# (model policy params are read by the forward tester from best_params.*)

# ---------- output path ----------
if [[ -z "$OUTPUT_CSV" ]]; then
  base="forward_${STAKE_MODE}_ds${DOWNSEC}_dedupe${DEDUP}_hair${HAIRCUT}_${START_DATE}_${END_DATE}.csv"
  # normalize dedupe for filename
  base="${base// /}"
  OUTPUT_CSV="$MODEL_DIR/$base"
fi

# ---------- run ----------
set -x
python3 "$FWD" \
  --curated "$CURATED" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --sport "$SPORT" \
  --model-dir "$MODEL_DIR" \
  --preoff-max "$PREOFF_MAX" \
  --commission "$COMMISSION" \
  --stake-mode "$STAKE_MODE" \
  --kelly-cap "$KELLY_CAP" \
  --kelly-floor "$KELLY_FLOOR" \
  --bankroll-nom "$BANKROLL" \
  --downsample-secs "$DOWNSEC" \
  --dedupe-mode "$DEDUP" \
  --haircut-ticks "$HAIRCUT" \
  --output-csv "$OUTPUT_CSV"
set +x

echo "Output → $OUTPUT_CSV"
