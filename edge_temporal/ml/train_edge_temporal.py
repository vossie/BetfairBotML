#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRICT real-data trainer for Edge Temporal — rolling window only.

- Loads curated Parquet/CSV for train+valid date window only.
- Exits if required cols are missing.
- Uses GPU when --device=cuda, errors if not available.
- Forces float32 everywhere to bound memory.
"""

import argparse, os, sys, time, json, re, glob
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# ----------------------------------
# Config / schema helpers
# ----------------------------------
ID_COLS = {"marketId", "selectionId"}
LABEL_COLS = {"winLabel", "pm_label", "pm_future_ticks"}
ODDS_COLS = {"ltp"}
DATE_COL_CANDIDATES = ("asof", "eventDate", "marketDate", "date")

def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def extract_date_from_name(p: Path) -> datetime | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}", p.as_posix())
    return parse_date(m.group(0)) if m else None

def odds_to_prob(odds):
    a = np.asarray(odds, dtype=np.float32)
    return np.where(np.isfinite(a) & (a > 0), 1.0 / a, np.nan).astype(np.float32)

def safe_logloss(y_true, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

def kelly_fraction(p, odds):
    b = float(odds) - 1.0
    if b <= 0.0: return 0.0
    q = 1.0 - float(p)
    f = (b * float(p) - q) / b
    return max(0.0, float(f))

def make_xgb_params(device: str):
    if device.lower() == "cuda":
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            max_depth=7,
            eta=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            max_bin=256,
        )
    else:
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            predictor="cpu_predictor",
            max_depth=7,
            eta=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            max_bin=512,
        )

def train_xgb(params, dtrain, dvalid, num_boost_round=500, early_stopping_rounds=50, must_use_gpu=False):
    try:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        return booster
    except xgb.core.XGBoostError as e:
        if must_use_gpu:
            die(f"GPU training failed: {e}")
        raise

# ----------------------------------
# Data loading (patched for rolling window)
# ----------------------------------
def glob_files(curated_root: str, start: datetime, end: datetime) -> list[Path]:
    root = Path(curated_root)
    if not root.exists():
        die(f"CURATED root not found: {root}")

    pats = os.environ.get("FEATURE_GLOB", f"{root}/**/*.parquet").split(";")
    pats_csv = os.environ.get("FEATURE_CSV_GLOB", f"{root}/**/*.csv").split(";")
    files = []
    for pat in pats: files.extend(glob.glob(pat, recursive=True))
    for pat in pats_csv: files.extend(glob.glob(pat, recursive=True))
    files = [Path(f) for f in files if Path(f).is_file()]

    # Only keep files whose date is in [start, end]
    sel = []
    for f in files:
        dt = extract_date_from_name(f)
        if dt is not None and start <= dt <= end:
            sel.append(f)
    if not sel:
        die(f"No curated files found in range {start.date()} → {end.date()}")
    return sel

def read_frame(p: Path) -> pl.DataFrame:
    if p.suffix.lower() == ".parquet": return pl.read_parquet(p)
    if p.suffix.lower() == ".csv": return pl.read_csv(p, infer_schema_length=200000)
    raise ValueError(f"Unsupported type: {p}")

def ensure_required_cols(df: pl.DataFrame):
    for c in ("winLabel", "ltp", "marketId", "selectionId"):
        if c not in df.columns:
            die(f"Missing required col {c}")
    return True

def pick_date_col(df: pl.DataFrame):
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            try:
                return df.with_columns(pl.col(c).cast(pl.Datetime)), c
            except: continue
    return df, None

def select_feature_cols(df: pl.DataFrame):
    exclude = ID_COLS | LABEL_COLS | ODDS_COLS
    feats = []
    for c, dt in df.schema.items():
        if c in exclude:
            continue
        # works across Polars versions
        if dt in pl.NUMERIC_DTYPES:
            feats.append(c)
    if not feats:
        die("No numeric feature cols found.")
    return feats


# ----------------------------------
# Args
# ----------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--train-days", type=int, default=12)
    ap.add_argument("--valid-days", type=int, default=2)
    ap.add_argument("--edge-thresh", type=float, default=0.015)
    ap.add_argument("--pm-cutoff", type=float, default=0.65)
    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--stake", choices=["flat","kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.0)
    ap.add_argument("--bankroll-nom", type=float, default=1000.0)
    ap.add_argument("--ltp-min", type=float, default=1.5)
    ap.add_argument("--ltp-max", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()

# ----------------------------------
# Main
# ----------------------------------
def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    asof_dt = parse_date(args.asof)
    train_end = asof_dt - timedelta(days=args.valid_days)
    train_start = train_end - timedelta(days=args.train_days - 1)
    valid_start = asof_dt - timedelta(days=args.valid_days - 1)
    valid_end = asof_dt

    # Only glob the files we need for [train_start, valid_end]
    files = glob_files(args.curated, train_start, valid_end)
    dfs = [read_frame(p) for p in files]
    df_all = pl.concat(dfs, how="diagonal_relaxed")

    ensure_required_cols(df_all)
    df_all, date_col = pick_date_col(df_all)

    # Partition
    df_train = df_all.filter((pl.col(date_col) >= train_start) & (pl.col(date_col) <= train_end))
    df_valid = df_all.filter((pl.col(date_col) >= valid_start) & (pl.col(date_col) <= valid_end))

    feat_cols = select_feature_cols(df_all)

    # Force float32
    X_train = df_train.select(feat_cols).to_numpy().astype(np.float32)
    y_train = df_train.get_column("winLabel").to_numpy().astype(np.float32)
    X_valid = df_valid.select(feat_cols).to_numpy().astype(np.float32)
    y_valid = df_valid.get_column("winLabel").to_numpy().astype(np.float32)
    odds_valid = df_valid.get_column("ltp").to_numpy().astype(np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    params = make_xgb_params(args.device)
    booster = train_xgb(params, dtrain, dvalid, must_use_gpu=(args.device=="cuda"))
    preds = booster.predict(dvalid)

    print(f"[Value head] logloss={safe_logloss(y_valid,preds):.4f} auc={roc_auc_score(y_valid,preds):.3f}")

if __name__ == "__main__":
    main()
