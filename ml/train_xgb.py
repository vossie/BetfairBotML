# ml/train_xgb.py
from __future__ import annotations
import argparse, os
import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss
from xgboost import XGBClassifier

from ml.features import build_features_streaming

FEATURES = [
    "ltp_odds","best_back_odds","best_lay_odds","best_back_size","best_lay_size",
    "spreadTicks","imbalanceBest1","best1_size_imbalance","microprice_best1","tradedVolume"
]

def prepare_xy(feats: pl.DataFrame, labels: pl.DataFrame):
    df = feats.join(labels.select(["marketId","selectionId","winLabel"]), on=["marketId","selectionId"], how="inner")
    df = df.drop_nulls(["winLabel"])
    X = df.select(FEATURES).to_numpy()
    y = df["winLabel"].to_numpy().astype(np.int64)
    # group by marketId to prevent leakage
    groups = df["marketId"].to_numpy()
    return X, y, groups, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--date", required=True)  # YYYY-MM-DD
    ap.add_argument("--decision_secs", type=int, default=5)
    args = ap.parse_args()

    feats, labels = build_features_streaming(args.curated, args.sport, args.date, args.decision_secs)

    X, y, groups, df_all = prepare_xy(feats, labels)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]

    model = XGBClassifier(
        tree_method="gpu_hist", predictor="gpu_predictor",
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,
        eval_metric="logloss", random_state=42
    )
    model.fit(Xtr, ytr)

    p_val = model.predict_proba(Xva)[:,1]
    print("Val LogLoss:", log_loss(yva, p_val))
    print("Val Brier  :", brier_score_loss(yva, p_val))

    # Persist
    os.makedirs("artifacts", exist_ok=True)
    import joblib
    joblib.dump({"model": model, "features": FEATURES}, "artifacts/xgb_preoff.pkl")
    print("Saved artifacts/xgb_preoff.pkl")

if __name__ == "__main__":
    main()
