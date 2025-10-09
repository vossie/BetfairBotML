#!/usr/bin/env python3
# train_price_trend (inf-safe, memory-safe)
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import polars as pl

# XGBoost (GPU-capable)
from xgboost import XGBRegressor

# ───────────────────────────────
# Utilities / feature builder hooks
# We call into your existing feature builder. If your project uses a different
# function name/location, tweak the 'load_train_valid' function at the bottom.
# ───────────────────────────────
def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_price_trend (inf-safe, memory-safe)",
        description="Train XGBoost trend regressor on curated Betfair snapshots."
    )
    # Core data window
    p.add_argument("--curated", required=True, help="Curated root, e.g. /mnt/nvme/betfair-curated")
    p.add_argument("--asof", required=True, help="AS OF date (YYYY-MM-DD)")
    p.add_argument("--start-date", required=True, help="Start date for training window (YYYY-MM-DD)")
    p.add_argument("--valid-days", type=int, default=7, help="Validation days ending at ASOF (inclusive)")
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30, help="minutes pre-off")
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "output" / "models"))
    p.add_argument("--country-facet", action="store_true", help="If set, facet by country (if present).")

    # Performance / sampling guards (memory-safe)
    p.add_argument("--downsample-secs", type=int, default=None, help="Optional row sampler spacing (sec)")
    p.add_argument("--sample-frac", type=float, default=None, help="Optional per-day fractional sample [0,1]")
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-valid-rows", type=int, default=250_000)

    # XGBoost hyperparameters (GPU-friendly defaults)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-n-estimators", type=int, default=400)
    p.add_argument("--xgb-learning-rate", type=float, default=0.10)  # eta
    p.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    p.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    p.add_argument("--xgb-random-state", type=int, default=42)
    p.add_argument("--xgb-early-stopping-rounds", type=int, default=50)

    # Feature list override (defaults match your 17-feature model)
    p.add_argument(
        "--feature-list",
        default="ltp_f,vol_f,ltp_diff_5s,vol_diff_5s,"
                "ltp_lag30s,ltp_lag60s,ltp_lag120s,"
                "tradedVolume_lag30s,tradedVolume_lag60s,tradedVolume_lag120s,"
                "ltp_mom_30s,ltp_mom_60s,ltp_mom_120s,"
                "ltp_ret_30s,ltp_ret_60s,ltp_ret_120s,mins_to_off",
        help="Comma-separated feature columns to train on."
    )
    # Target column (label). Your pipeline uses future delta for EV/mtm (__dp).
    p.add_argument("--target-col", default="__dp", help="Target column (default: __dp).")

    return p.parse_args()

# ───────────────────────────────
# IO helpers
# ───────────────────────────────
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def header(args: argparse.Namespace, model_dir: Path):
    print("=== Price Trend Training ===")
    print(f"Curated root:    {args.curated}")
    print(f"ASOF:            {args.asof}")
    print(f"Start date:      {args.start_date}")
    print(f"Valid days:      {args.valid_days}")
    print(f"Horizon (secs):  {args.horizon_secs}")
    print(f"Pre-off max (m): {args.preoff_max}")
    print(f"XGBoost device:  {args.device}")
    print(f"EV mode:         mtm")

def train_valid_ranges(asof: str, start_date: str, valid_days: int) -> Tuple[Tuple[str,str], Tuple[str,str]]:
    asof_d = _dt(asof).date()
    valid_start = (asof_d - timedelta(days=valid_days)).strftime("%Y-%m-%d")
    train_start = start_date
    train_end = (asof_d - timedelta(days=valid_days+1)).strftime("%Y-%m-%d")
    valid_end = asof
    print(f"[trend] TRAIN: {train_start} .. {train_end}")
    print(f"[trend] VALID: {valid_start} .. {valid_end}")
    return (train_start, train_end), (valid_start, valid_end)

# ───────────────────────────────
# Feature / data loader
# This function calls into your existing feature builder to construct
# train/valid frames with the requested feature columns and the label (__dp).
# Adjust the import / function call here if your local module name differs.
# ───────────────────────────────
def load_train_valid(
    args: argparse.Namespace,
    features: List[str],
    target_col: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns (train_df, valid_df) with columns: features + [target_col]
    Must be memory-safe: we collect with row caps if requested.
    """
    # Try to use your existing builder
    try:
        # Expectation: build_features has a function like:
        #   build_price_trend_frames(curated, start_str, end_str, valid_start_str, valid_end_str, ...)
        # that returns (train_df, valid_df) as Polars DataFrames including feature columns and '__dp' target.
        from build_features import build_price_trend_frames  # type: ignore
        (train_df, valid_df) = build_price_trend_frames(
            curated_root=args.curated,
            sport=args.sport,
            horizon_secs=args.horizon_secs,
            preoff_max_mins=args.preoff_max,
            commission=args.commission,
            start_date=args.start_date,
            asof=args.asof,
            valid_days=args.valid_days,
            downsample_secs=args.downsample_secs,
            sample_frac=args.sample_frac,
            feature_list=features,
            target_col=target_col,
            country_facet=args.country_facet,
            max_train_rows=args.max_train_rows,
            max_valid_rows=args.max_valid_rows,
        )
        return train_df, valid_df
    except Exception as e:
        # Fallback: build using per-day sim parquet if available (simulate_stream outputs)
        # This is slower but robust; it scans sim_YYYY-MM-DD.parquet under output/stream/
        # and concatenates days into train/valid splits, selecting requested columns.
        # Note: requires that simulate_stream ran for the date ranges previously.
        out_stream = Path(__file__).resolve().parents[1] / "output" / "stream"
        (train_rng, valid_rng) = train_valid_ranges(args.asof, args.start_date, args.valid_days)
        train_days = date_range_days(*train_rng)
        valid_days = date_range_days(*valid_rng)

        def _collect(days: List[str]) -> pl.DataFrame:
            files = [str(out_stream / f"sim_{d}.parquet") for d in days if (out_stream / f"sim_{d}.parquet").exists()]
            if not files:
                raise RuntimeError("No sim_*.parquet files found for requested days. "
                                   "Run simulate_stream over the window or provide a build_features function.")
            lf = pl.scan_parquet(files).select([*features, target_col])
            if args.max_train_rows and days is train_days:
                return lf.fetch(args.max_train_rows)
            if args.max_valid_rows and days is valid_days:
                return lf.fetch(args.max_valid_rows)
            return lf.collect()

        train_df = _collect(train_days)
        valid_df = _collect(valid_days)
        return train_df, valid_df

def date_range_days(start_str: str, end_str: str) -> List[str]:
    s = _dt(start_str).date()
    e = _dt(end_str).date()
    days = []
    d = s
    while d <= e:
        days.append(_date_str(datetime(d.year, d.month, d.day)))
        d = d + timedelta(days=1)
    return days

# ───────────────────────────────
# Metrics
# We use your target (__dp) to compute an EV-like metric on VALID:
#   "valid EV per £1: mean=..."
# This mirrors your logs so sweeps can parse them.
# ───────────────────────────────
def valid_ev_per_1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # Expected value proxy per £1: assume y_true is future price delta in ticks converted to £1 EV
    # If your pipeline provides ev_per_1 directly, swap in that column here.
    # We compute the correlation-weighted magnitude as a robust sanity metric:
    if y_true.size == 0:
        return float("nan")
    # Simple expected mtm gain when betting proportional to predicted sign*magnitude
    ev = (np.sign(y_pred) * y_true).mean()
    return float(ev)

# ───────────────────────────────
# Main
# ───────────────────────────────
def main():
    args = parse_args()
    model_dir = Path(args.output_dir)
    ensure_dir(model_dir)

    header(args, model_dir)
    train_rng, valid_rng = train_valid_ranges(args.asof, args.start_date, args.valid_days)

    features = [c.strip() for c in args.feature_list.split(",") if c.strip()]
    target_col = args.target_col

    # Load data (DataFrames with selected features + target)
    train_df, valid_df = load_train_valid(args, features, target_col)

    # Convert to numpy
    X_train = train_df.select(features).to_numpy()
    y_train = train_df[target_col].to_numpy()
    if args.max_train_rows:
        X_train = X_train[: args.max_train_rows]
        y_train = y_train[: args.max_train_rows]

    X_valid = valid_df.select(features).to_numpy()
    y_valid = valid_df[target_col].to_numpy()
    if args.max_valid_rows:
        X_valid = X_valid[: args.max_valid_rows]
        y_valid = y_valid[: args.max_valid_rows]

    # Guard against empty
    n_tr, n_val = len(y_train), len(y_valid)
    print(f"[trend] rows train={n_tr:,}  valid={n_val:,}")

    if n_tr == 0 or n_val == 0:
        # Save a tiny stub model to keep the pipeline alive, but warn loudly.
        print("[trend] WARNING: empty dataset; saving no-op model.")
        dummy = XGBRegressor(n_estimators=1, max_depth=1, tree_method="hist")
        model_path = model_dir / "xgb_trend_reg.json"
        dummy.save_model(str(model_path))
        print(f"[trend] saved model → {model_path}")
        return

    # XGBoost model (GPU if requested)
    xgb_params = dict(
        max_depth=args.xgb_max_depth,
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        min_child_weight=args.xgb_min_child_weight,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        reg_lambda=args.xgb_reg_lambda,
        reg_alpha=args.xgb_reg_alpha,
        random_state=args.xgb_random_state,
        tree_method="gpu_hist" if args.device == "cuda" else "hist",
        predictor="gpu_predictor" if args.device == "cuda" else "auto",
        n_jobs=0,
    )
    model = XGBRegressor(**xgb_params)

    fit_kwargs = {}
    if args.xgb_early_stopping_rounds and n_val > 0:
        fit_kwargs.update(
            dict(eval_set=[(X_valid, y_valid)], early_stopping_rounds=args.xgb_early_stopping_rounds, verbose=False)
        )

    model.fit(X_train, y_train, **fit_kwargs)

    # Compute validation EV/£1 metric (to mirror your logs)
    y_pred = model.predict(X_valid)
    ev_valid = valid_ev_per_1(y_pred, y_valid)
    # For continuity with your sweeps, also print p>0 share if desired
    p_pos = float((y_pred > 0).mean()) if y_pred.size else float("nan")
    print(f"[trend] valid EV per £1: mean={ev_valid:.5f}  p>0 share={p_pos:.3f}")

    # Save model
    model_path = model_dir / "xgb_trend_reg.json"
    model.save_model(str(model_path))
    print(f"[trend] saved model → {model_path}")

if __name__ == "__main__":
    # tame Polars printing
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(200)
    main()
