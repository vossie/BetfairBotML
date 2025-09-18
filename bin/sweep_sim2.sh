cd /opt/BetfairBotML
cat > sweep_sim2.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/BetfairBotML

# module to run
MODULE="${MODULE:-ml.sim2_all}"

# grid to sweep (edit to taste, or override by GRID env)
GRID="${GRID:-side=[auto], min_edge=[0.05:0.12:0.01], kelly=[0.05,0.10,0.15,0.25], min_ev=[0.02,0.03,0.05], odds_min=[1.5,1.6], odds_max=[4.0,5.0,6.0,8.0], max_stake_per_bet=[2,3], slip_ticks=[0,1,2]}"

PYTHONPATH=. python -m "$MODULE" \
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
  --commission 0.05 \
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
  --min-ev 0.03 \
  --side auto \
  --min-edge 0.08 \
  --kelly 0.10 \
  --sweep-grid "$GRID" \
  --sweep-parallel 0
SH
chmod +x sweep_sim2.sh
