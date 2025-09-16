#!/bin/bash
# run.sh
cd /opt/BetfairBotML || exit 1

# Load environment
source "$(dirname "$0")/set-env-vars-prod.sh"

.venv/bin/python -m ml.sim   --model-30 model_30.json   --model-180 model_180.json   --gate-mins 45   --curated /mnt/nvme/betfair-curated   --sport horse-racing   --date 2025-09-14 --days 10   --preoff-mins 180   --min-edge 0.02   --kelly 0.25   --commission 0.02   --top-n-per-market 1   --side auto   --bets-out bets_dual.csv