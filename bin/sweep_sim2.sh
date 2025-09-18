cd /opt/BetfairBotML && cat > sweep_sim2.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/BetfairBotML

# Use ml.sim2 by default; set MODULE=ml.sim2_all to target that module.
MODULE="${MODULE:-ml.sim2}"

OUT="./output/sweep_results.csv"
mkdir -p ./output
echo "combo,min_edge,kelly,n_bets,stake,pnl,roi" > "$OUT"

i=0
for MIN_EDGE in 0.05 0.07 0.10; do
  for KELLY in 0.10 0.25 0.50; do
    i=$((i+1))
    echo "=== Combo $i: min_edge=$MIN_EDGE kelly=$KELLY ==="
    LOG=$(PYTHONPATH=. python -m "$MODULE" \
      --model ./output/xgb_model.json \
      --curated /mnt/nvme/betfair-curated \
      --sport horse-racing \
      --date 2025-09-17 \
      --days-before 1 \
      --preoff-mins 30 \
      --stream-bucket-secs 5 \
      --latency-ms 300 \
      --cooldown-secs 20 \
      --max-open-per-market 2 \
      --place-until-mins 0 \
      --persistence lapse \
      --rest-secs 0 \
      --min-stake 1.0 \
      --tick-snap \
      --slip-ticks 1 \
      --min-edge "$MIN_EDGE" \
      --kelly "$KELLY" \
      --commission 0.05 \
      --side auto \
      --top-n-per-market 1 \
      --stake-cap-market 10 \
      --stake-cap-day 600 \
      --max-exposure-day 900 \
      --odds-min 1.5 \
      --odds-max 8.0 \
      --back-odds-min 1.6 \
      --back-odds-max 8.0 \
      --lay-odds-min 1.3 \
      --lay-odds-max 3.5 \
      --max-stake-per-bet 3.0 \
      --max-liability-per-bet 20.0 \
      --min-ev 0.03 2>&1)
    LINE=$(echo "$LOG" | grep -E 'Summary: n_bets=' | tail -n1) || true
    if [[ -z "${LINE:-}" ]]; then
      echo "WARN: No summary parsed for combo $i" >&2
      continue
    fi
    NBETS=$(echo "$LINE" | awk -F'[ =]+' '{for(i=1;i<=NF;i++) if($i=="n_bets"){print $(i+1); exit}}')
    STAKE=$(echo "$LINE" | awk -F'[ =]+' '{for(i=1;i<=NF;i++) if($i=="stake"){print $(i+1); exit}}' | tr -d ',')
    PNL=$(echo "$LINE" | awk -F'[ =]+' '{for(i=1;i<=NF;i++) if($i=="pnl"){print $(i+1); exit}}' | tr -d ',')
    ROI=$(echo "$LINE" | awk -F'ROI=' '{print $2}' | tr -d '% ')

    echo "$i,$MIN_EDGE,$KELLY,$NBETS,$STAKE,$PNL,$ROI" >> "$OUT"
  done
done

echo
echo "=== BEST BY ROI ==="
{ head -n1 "$OUT"; tail -n +2 "$OUT" | sort -t, -k7,7nr | head -n1; }
SH
chmod +x sweep_sim2.sh
./sweep_sim2.sh
