# single day
python ml/sim.py --curated <root> --sport horse-racing --date 2025-09-15 --model

# date + one day before (i.e., 2025-09-14 and 2025-09-15)
python ml/sim.py --curated <root> --sport horse-racing --date 2025-09-15 --days-before 1 --model

# Single-model (e.g., 30-min model)
.venv/bin/python -m ml.sim \
  --model xgb_model.json \
  --curated /mnt/nvme/betfair-curated \
  --sport horse-racing \
  --date 2025-09-14 --days 10 \
  --preoff-mins 30 \
  --min-edge 0.02 \
  --kelly 0.25 \
  --commission 0.02 \
  --top-n-per-market 1 \
  --side auto \
  --bets-out bets_30.csv

# Dual-horizon gate (30 & 180 models)
.venv/bin/python -m ml.sim \
  --model-30 model_30.json \
  --model-180 model_180.json \
  --gate-mins 45 \
  --curated /mnt/nvme/betfair-curated \
  --sport horse-racing \
  --date 2025-09-14 --days 10 \
  --preoff-mins 180 \
  --min-edge 0.02 \
  --kelly 0.25 \
  --commission 0.02 \
  --top-n-per-market 1 \
  --side auto \
  --bets-out bets_dual.csv
