| Step                          | Purpose                        | Output                       |
| ----------------------------- | ------------------------------ | ---------------------------- |
| `train_price_trend.sh`        | Build prediction model         | `models/xgb_trend_reg.json`  |
| `train_price_trend_automl.py` | Search for best trading params | `automl/best_config.json`    |
| `run_best_trend.sh`           | Final performance backtest     | `stream/summary_<ASOF>.json` |


What each script does

train_price_trend.sh → trains the XGBoost model, then runs the fixed simulator to produce daily EV parquet files. You can pass a date or extra simulator flags after --. 

train_price_trend

sweep_price_trend.sh → runs sim_sweep.py over grids (edge, stake modes, odds bands) and writes results under output/stream/<ASOF>/sweeps/.... You supply ASOF as the first arg. 

sweep_price_trend

run_best_trend.sh → if no best config exists, it runs AutoML (trains if needed, sweeps many configs), picks the best (best_config.json), then backtests that config via train_price_trend.sh. Requires jq. 

run_best_trend

Quick starts
1) Train + simulate EV (you already did this)
export CURATED_ROOT=/mnt/nvme/betfair-curated
./train_price_trend/bin/train_price_trend.sh 2025-10-06
# or: ./train_price_trend/bin/train_price_trend.sh -- --defs-days-back 45


Outputs:

Model → output/models/xgb_trend_reg.json

Daily EV parquet → output/stream/sim_YYYY-MM-DD.parquet (plus logs per day) 

train_price_trend

2) Sweep trading configs (uses sim_sweep.py)
export CURATED_ROOT=/mnt/nvme/betfair-curated
./train_price_trend/bin/sweep_price_trend.sh 2025-10-06


Defaults inside the script:

Grids: EDGE_THRESH_GRID=0.0005,0.001,0.002,0.003,0.005, STAKE_MODES_GRID=flat,kelly, ODDS_BANDS_GRID=none,2.2:3.6,1.5:5.0

Outputs under: output/stream/ with a sweep subfolder & summaries per run. 

sweep_price_trend

Override anything via env, e.g.:

EDGE_THRESH_GRID=0.001,0.002 ODDS_BANDS_GRID=1.5:5.0 TAG=mygrid \
./train_price_trend/bin/sweep_price_trend.sh 2025-10-06

3) Full AutoML → best config → backtest
export CURATED_ROOT=/mnt/nvme/betfair-curated
./train_price_trend/bin/run_best_trend.sh 2025-10-06 automl


If no best_config.json at output/automl/<ASOF>/<TAG>/, it runs AutoML (GPU train, CPU sims), then backtests best params.

Needs jq installed. It also prunes old AutoML runs. 

run_best_trend

Tips / gotchas

Set CURATED_ROOT first (required by all scripts). 

train_price_trend

For faster/lighter sims: BATCH_SIZE=75000 POLARS_MAX_THREADS=3 (already the defaults in run_best_trend.sh). 

run_best_trend

To pass extra simulator-only args with train_price_trend.sh, use -- then flags (e.g., --file-sample-mode tail). 

train_price_trend

If you tell me which path you want (quick single sweep vs full AutoML), I’ll give you a one-liner with sensible overrides for your box.