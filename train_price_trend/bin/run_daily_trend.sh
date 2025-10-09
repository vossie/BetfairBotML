# ── Compute START_DATE ─────────────────────────────────────────────────────────
if [[ -n "${START_DATE_FORCE:-}" ]]; then
  START_DATE="$START_DATE_FORCE"
  echo "[auto] Using fixed START_DATE=${START_DATE}"
elif [[ "${USE_EARLIEST:-0}" == "1" ]]; then
  START_DATE="${EARLIEST_DATE:-2025-09-05}"
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

echo "[auto] Deriving dirs…"
SWEEP_ROOT="$OUT_BASE/sweeps/$ASOF/$TAG"
CONFIRM_OUT="$OUT_BASE/confirm_${ASOF}_${TAG}"
echo "[auto] SWEEP_ROOT=$SWEEP_ROOT"
echo "[auto] CONFIRM_OUT=$CONFIRM_OUT"

# ── Cleanup helpers ───────────────────────────────────────────────────────────
clean_dir() {
  local path="$1"
  case "${CLEAN_MODE:-auto}" in
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
      echo "[auto] Unknown CLEAN_MODE='${CLEAN_MODE:-}' (use auto|prompt|skip)."; exit 1;;
  esac
}

echo "[auto] Cleaning/creating output dirs…"
clean_dir "$SWEEP_ROOT";   mkdir -p "$SWEEP_ROOT"
clean_dir "$CONFIRM_OUT";  mkdir -p "$CONFIRM_OUT"
echo "[auto] Output dirs ready."

# ── Threading for Polars ──────────────────────────────────────────────────────
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-$(nproc || echo 8)}"
echo "[auto] POLARS_MAX_THREADS=$POLARS_MAX_THREADS"

echo "=== TRAIN ${ASOF} ==="
