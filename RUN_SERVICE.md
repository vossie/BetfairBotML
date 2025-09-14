# install deps
pip install fastapi uvicorn xgboost polars joblib

# set env (example: joblib artifact with features list)
export MODEL_PATH="artifacts/xgb_preoff.pkl"
export API_KEY="supersecret"

# if using xgb JSON:
# export MODEL_PATH="xgb_model.json"
# export FEATURE_LIST="ltp,mom_10s,mom_60s,vol_10s,vol_60s,spread_ticks,imb1,traded_vol,overround,norm_prob,rank_price"
# export API_KEY="supersecret"

uvicorn ml.service:app --host 0.0.0.0 --port 8080 --workers 1
