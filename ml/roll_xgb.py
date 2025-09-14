# ml/roll_xgb.py
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from . import features


# --------------------------- date helpers ---------------------------

def _daterange_inclusive(start_str: str, end_str: str) -> List[str]:
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end must be >= start")
    out: List[str] = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _window_dates(valid_date: str, window_days: int) -> List[str]:
    """Return the training window dates (valid_date - window_days .. valid_date - 1), inclusive."""
    end = datetime.strptime(valid_date, "%Y-%m-%d").date() - timedelta(days=1)
    start = end - timedelta(days=window_days - 1)
    return _daterange_inclusive(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


# --------------------------- GPU helpers ---------------------------

def _has_nvidia_gpu() -> bool:
    try:
        import pynvml  # noqa: F401
        return True
    except Exception:
        return os.system("nvidia-smi -L >/dev/null 2>&1") == 0


def _xgb_params(learning_rate: float, max_depth: int, n_estimators: int) -> Dict:
    use_gpu = _has_nvidia_gpu()
    return {
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "predictor": "gpu_predictor" if use_gpu else "auto",
        "objective": "binary:logistic",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "n_estimators": n_estimators,
        "random_state": 42,
        "verbosity": 1,
        "eval_metric": ["logloss", "auc"],
    }


# --------------------------- feature selection ---------------------------

def _select_feature_cols(df_feat: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {"marketId", "selectionId", "ts", "ts_ms", "publishTimeMs", label_col}
    cols: List[str] = []
    for c, dt in zip(df_feat.columns, df_feat.dtypes):
        if c in exclude:
            continue
        if dt.is_numeric():
            cols.append(c)
    if not cols:
        raise RuntimeError("No numeric feature columns found for XGBoost.")
    return cols


# --------------------------- one-step train/eval ---------------------------

def _train_eval_once(
    curated: str,
    sport: str,
    train_dates: List[str],
    val_date: str,
    preoff_mins: int,
    batch_markets: int,
    downsample_secs: int | None,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    label_col: str,
) -> Tuple[Dict, Dict, List[str]]:
    # Build TRAIN features
    df_train, total_raw_train = features.build_features_streaming(
        curated_root=curated,
        sport=sport,
        dates=train_dates,
        preoff_minutes=preoff_mins,
        batch_markets=batch_markets,
        downsample_secs=downsample_secs,
    )
    if df_train.is_empty():
        return ({"rows": 0, "raw": int(total_raw_train)}, {"rows": 0}, train_dates)

    # Build VAL features (single day)
    df_val, total_raw_val = features.build_features_streaming(
        curated_root=curated,
        sport=sport,
        dates=[val_date],
        preoff_minutes=preoff_mins,
        batch_markets=batch_markets,
        downsample_secs=downsample_secs,
    )
    if df_val.is_empty():
        return ({"rows": int(df_train.height), "raw": int(total_raw_train)},
                {"rows": 0}, train_dates)

    # Sort by time if present
    for name, df in (("train", df_train), ("val", df_val)):
        if "ts" in df.columns:
            locals()[f"df_{name}"] = df.sort("ts")
        elif "publishTimeMs" in df.columns:
            locals()[f"df_{name}"] = df.sort("publishTimeMs")

    if label_col not in df_train.columns or label_col not in df_val.columns:
        raise RuntimeError(f"Label column '{label_col}' missing in features.")

    feat_cols = _select_feature_cols(df_train, label_col)

    # Numpy arrays (float32)
    Xtr = df_train.select(feat_cols).fill_null(strategy="mean").to_numpy(dtype=np.float32)
    ytr = df_train.select(label_col).to_numpy().ravel()
    Xva = df_val.select(feat_cols).fill_null(strategy="mean").to_numpy(dtype=np.float32)
    yva = df_val.select(label_col).to_numpy().ravel()

    # Train XGB
    params = _xgb_params(learning_rate, max_depth, n_estimators)
    model = xgb.XGBClassifier(**params)
    model.fit(Xtr, ytr)

    # Metrics on VAL
    pva = model.predict_proba(Xva)[:, 1]
    pva = np.clip(pva, 1e-6, 1 - 1e-6)
    metrics = {
        "logloss": float(log_loss(yva, pva)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(yva, pva))
    except Exception:
        pass

    info = {
        "train_rows": int(df_train.height),
        "train_raw": int(total_raw_train),
        "val_rows": int(df_val.height),
        "val_raw": int(total_raw_val),
        "n_features": len(feat_cols),
    }
    return info, metrics, feat_cols


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rolling evaluation: train on last N days, validate on next day (GPU XGBoost)."
    )
    ap.add_argument("--curated", required=True, help="s3://bucket[/prefix] or /local/path")
    ap.add_argument("--sport", required=True, help="e.g. horse-racing")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (first validation day)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (last validation day)")
    ap.add_argument("--window-days", type=int, default=7, help="training window length (days)")
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)

    # Model
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-estimators", type=int, default=500)

    # Output
    ap.add_argument("--out-csv", default="", help="Optional path to save rolling metrics CSV (e.g., artifacts/rolling_metrics.csv)")
    args = ap.parse_args()

    val_days = _daterange_inclusive(args.start, args.end)

    rows: List[Dict] = []
    last_feat_cols: List[str] = []

    for val_date in val_days:
        train_dates = _window_dates(val_date, args.window_days)
        print(f"[ROLL] Train: {train_dates[0]}..{train_dates[-1]} | Val: {val_date}")

        info, metrics, feat_cols = _train_eval_once(
            curated=args.curated,
            sport=args.sport,
            train_dates=train_dates,
            val_date=val_date,
            preoff_mins=args.preoff_mins,
            batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            label_col=args.label_col,
        )
        last_feat_cols = feat_cols or last_feat_cols

        row = {
            "val_date": val_date,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "train_rows": info.get("train_rows", 0),
            "train_raw": info.get("train_raw", 0),
            "val_rows": info.get("val_rows", 0),
            "val_raw": info.get("val_raw", 0),
            "n_features": info.get("n_features", 0),
            "logloss": metrics.get("logloss", float("nan")),
            "auc": metrics.get("auc", float("nan")),
        }
        print(f"[ROLL] {val_date} → logloss={row['logloss']:.4f}"
              + (f", auc={row['auc']:.4f}" if not np.isnan(row["auc"]) else ""))
        rows.append(row)

    df_metrics = pl.DataFrame(rows)
    print("\nRolling metrics:")
    print(df_metrics)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df_metrics.write_csv(args.out_csv)
        print(f"Saved: {args.out_csv}")

    # Optional: print the final feature list used
    if last_feat_cols:
        print("\nFeatures used:", ", ".join(last_feat_cols))


if __name__ == "__main__":
    main()
