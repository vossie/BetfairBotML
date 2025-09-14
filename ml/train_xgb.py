# ml/train_xgb.py
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, UTC
from typing import List, Dict

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from . import features


# --------------------------- date helpers ---------------------------

def _daterange(end_date_str: str, days: int) -> List[str]:
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    d = start
    out: List[str] = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


# --------------------------- GPU helpers ---------------------------

def _has_nvidia_gpu() -> bool:
    try:
        import pynvml  # noqa: F401
        return True
    except Exception:
        return os.system("nvidia-smi -L >/dev/null 2>&1") == 0


def _xgb_params(n_classes: int, learning_rate: float, max_depth: int, n_estimators: int) -> Dict:
    use_gpu = _has_nvidia_gpu()
    tree_method = "gpu_hist" if use_gpu else "hist"
    predictor = "gpu_predictor" if use_gpu else "auto"

    if n_classes > 2:
        objective = "multi:softprob"
        num_class = n_classes
        eval_metric = ["mlogloss"]
    else:
        objective = "binary:logistic"
        num_class = None
        eval_metric = ["logloss", "auc"]

    params: Dict = {
        "tree_method": tree_method,
        "predictor": predictor,
        "objective": objective,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "n_estimators": n_estimators,
        "random_state": 42,
        "verbosity": 1,
        "eval_metric": eval_metric,
    }
    if num_class is not None:
        params["num_class"] = num_class
    return params


# --------------------------- training ---------------------------

def _select_feature_cols(df_feat: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {"marketId", "selectionId", "ts", "ts_ms", label_col, "publishTimeMs"}
    cols: List[str] = []
    for c, dt in zip(df_feat.columns, df_feat.dtypes):
        if c in exclude:
            continue
        if dt.is_numeric():
            cols.append(c)
    if not cols:
        raise RuntimeError("No numeric feature columns found for XGBoost.")
    return cols


def main():
    ap = argparse.ArgumentParser(description="GPU-accelerated XGBoost trainer over curated Betfair features.")
    # Source
    ap.add_argument("--curated", required=True, help="s3://bucket[/prefix] or /local/path")
    ap.add_argument("--sport", required=True, help="e.g. horse-racing")
    ap.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"), help="anchor date YYYY-MM-DD (treated as END of window)")
    ap.add_argument("--days", type=int, default=1, help="number of days back from --date (inclusive)")

    # Feature builder knobs (match run_train)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)

    # Model knobs
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--model-out", default="xgb_model.json")

    args = ap.parse_args()

    dates = _daterange(args.date, args.days)
    print(f"Building features from curated={args.curated}, sport={args.sport}, dates={dates[0]}..{dates[-1]}")

    # Build features via the streaming builder (already memory-safe)
    df_feat, total_raw = features.build_features_streaming(
        curated_root=args.curated,
        sport=args.sport,
        dates=dates,
        preoff_minutes=args.preoff_mins,
        batch_markets=args.batch_markets,
        downsample_secs=(args.downsample_secs or None),
    )
    if df_feat.is_empty():
        raise SystemExit("No features built (empty frame). Check curated paths and date range.")

    print(f"Feature rows: {df_feat.height:,} (from ~{total_raw:,} raw snapshot rows scanned)")

    # Sort by time to do a time-based split
    if "ts" in df_feat.columns:
        df_feat = df_feat.sort("ts")
    elif "publishTimeMs" in df_feat.columns:
        df_feat = df_feat.sort("publishTimeMs")

    label_col = args.label_col
    if label_col not in df_feat.columns:
        raise SystemExit(f"Label column '{label_col}' not found in features.")

    feature_cols = _select_feature_cols(df_feat, label_col)

    # Split last 20% as validation
    n = df_feat.height
    n_valid = max(1, int(n * 0.2))
    n_train = n - n_valid
    train_df = df_feat.slice(0, n_train)
    valid_df = df_feat.slice(n_train, n_valid)

    # Data to numpy (float32 for XGB)
    Xtr = train_df.select(feature_cols).fill_null(strategy="mean").to_numpy(dtype=np.float32)
    ytr = train_df.select(label_col).to_numpy().ravel()
    Xva = valid_df.select(feature_cols).fill_null(strategy="mean").to_numpy(dtype=np.float32)
    yva = valid_df.select(label_col).to_numpy().ravel()

    # Determine binary vs multi
    n_classes = int(pl.Series(ytr).unique().len())

    params = _xgb_params(
        n_classes=n_classes,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
    )

    # Train
    clf = xgb.XGBClassifier(**params)
    clf.fit(Xtr, ytr)

    # Metrics
    if n_classes > 2:
        pva = clf.predict_proba(Xva)
        metrics = {"mlogloss": float(log_loss(yva, pva))}
    else:
        pva = clf.predict_proba(Xva)[:, 1]
        pva = np.clip(pva, 1e-6, 1 - 1e-6)
        metrics = {"logloss": float(log_loss(yva, pva))}
        try:
            metrics["auc"] = float(roc_auc_score(yva, pva))
        except Exception:
            pass

    print("Metrics:", metrics)

    # Save model
    clf.save_model(args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
