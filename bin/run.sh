#!/bin/bash
# run.sh

cd /opt/BetfairBotML || exit

# Load environment variables from set-env-vars-prod.sh
source "$(dirname "$0")/set-env-vars-prod.sh"

git pull
.venv/bin/python -m ml.train_xgb --curated s3://betfair-curated  --sport horse-racing --date 2025-09-13 --days 8 --preoff-mins 30 --batch-markets 100 --downsample-secs 0