| Step                          | Purpose                        | Output                       |
| ----------------------------- | ------------------------------ | ---------------------------- |
| `train_price_trend.sh`        | Build prediction model         | `models/xgb_trend_reg.json`  |
| `train_price_trend_automl.py` | Search for best trading params | `automl/best_config.json`    |
| `run_best_trend.sh`           | Final performance backtest     | `stream/summary_<ASOF>.json` |
