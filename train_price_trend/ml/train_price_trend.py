#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import xgboost as xgb

from utils import (
    parse_date, to_ms, read_snapshots, filter_preoff,
    implied_prob_from_ltp_expr, time_to_off_minutes_expr,
    kelly_fraction_back, kelly_fraction_lay
)

UTC = timezone.utc

# feature set (match simulate_stream)
NUMERIC_FEATS = [
    "ltp","tradedVolume","spreadTicks","imbalanceBest1","mins_to_off","__p_now",
    "ltp_diff_5s","vol_diff_5s",
    "ltp_lag30s","ltp_lag60s","ltp_lag120s",
    "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
    "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
    "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
]

def parse_args():
    p = argparse.ArgumentParser(description="Train price-trend delta model (regression)")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    return p.parse_args()

# simple rolling feature builder on a sorted DataFrame
def build_features(df: pl.DataFrame) -> pl.DataFrame:
    # assume sorted by marketId, selectionId, publishTimeMs
    w = ["marketId","selectionId"]
    # 5s cadence → 30/60/120s are 6/12/24 steps
    df = (
        df.sort(["marketId","selectionId","publishTimeMs"])
          .with_columns([
              pl.col("ltp").diff().over(w).alias("ltp_diff_5s"),
              pl.col("tradedVolume").diff().over(w).alias("vol_diff_5s"),

              pl.col("ltp").shift(6).over(w).alias("ltp_lag30s"),
              pl.col("ltp").shift(12).over(w).alias("ltp_lag60s"),
              pl.col("ltp").shift(24).over(w).alias("ltp_lag120s"),

              pl.col("tradedVolume").shift(6).over(w).alias("tradedVolume_lag30s"),
              pl.col("tradedVolume").shift(12).over(w).alias("tradedVolume_lag60s"),
              pl.col("tradedVolume").shift(24).over(w).alias("tradedVolume_lag120s"),
          ])
          .with_columns([
              (pl.col("ltp") - pl.col("ltp_lag30s")).alias("ltp_mom_30s"),
              (pl.col("ltp") - pl.col("ltp_lag60s")).alias("ltp_mom_60s"),
              (pl.col("ltp") - pl.col("ltp_lag120s")).alias("ltp_mom_120s"),

              ((pl.col("ltp") - pl.col("ltp_lag30s")) / pl.when(pl.col("ltp_lag30s")==0).then(1e-12).otherwise(pl.col("ltp_lag30s"))).alias("ltp_ret_30s"),
              ((pl.col("ltp") - pl.col("ltp_lag60s")) / pl.when(pl.col("ltp_lag60s")==0).then(1e-12).otherwise(pl.col("ltp_lag60s"))).alias("ltp_ret_60s"),
              ((pl.col("ltp") - pl.col("ltp_lag120s"))/ pl.when(pl.col("ltp_lag120s")==0).then(1e-12).otherwise(pl.col("ltp_lag120s"))).alias("ltp_ret_120s"),
          ])
    )
    return df

def main():
    args = parse_args()
    curated = Path(args.curated)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    asof_dt = parse_date(args.asof)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)
    start_dt = parse_date(args.start_date)

    print("=== Price Trend Training ===")
    print(f"Curated root:    {curated}")
    print(f"ASOF:            {args.asof}")
    print(f"Start date:      {args.start_date}")
    print(f"Valid days:      {args.valid_days}")
    print(f"Horizon (secs):  {args.horizon_secs}")
    print(f"Pre-off max (m): {args.preoff_max}")
    print(f"Stake mode:      {args.stake_mode} (cap={args.kelly_cap} floor={args.kelly_floor})")

    print(f"[trend] TRAIN: {start_dt.date()} .. {(valid_start - timedelta(days=1)).date()}")
    print(f"[trend] VALID: {valid_start.date()} .. {valid_end.date()}")
    print(f"[trend] horizon={args.horizon_secs}s  preoff≤{args.preoff_max}m  device={args.device}")

    # Load snapshots + defs (join inside) then filter to pre-off window
    lf = read_snapshots(curated, start_dt, valid_end, args.sport)
    df = lf.collect()
    df = filter_preoff(df, args.preoff_max)

    # Split windows by publishTimeMs
    ms_train_lo = to_ms(start_dt)
    ms_train_hi = to_ms(valid_start)  # exclusive
    ms_valid_lo = to_ms(valid_start)
    ms_valid_hi = to_ms(valid_end + timedelta(days=1))

    df_train = df.filter((pl.col("publishTimeMs") >= ms_train_lo) & (pl.col("publishTimeMs") < ms_train_hi))
    df_valid = df.filter((pl.col("publishTimeMs") >= ms_valid_lo) & (pl.col("publishTimeMs") < ms_valid_hi))

    # Build rolling feats
    df_train = build_features(df_train)
    df_valid = build_features(df_valid)

    # Drop rows without enough history for lags
    for col in ["ltp_lag120s", "tradedVolume_lag120s"]:
        df_train = df_train.filter(pl.col(col).is_not_null())
        df_valid = df_valid.filter(pl.col(col).is_not_null())

    # TARGET: next-120s probability delta ≈ (p(t+H) - p(t))
    # Build approximate label with forward shift over group
    g = ["marketId","selectionId"]
    df_train = df_train.with_columns(
        ( (1.0 / pl.when(pl.col("ltp").shift(-24).over(g) < 1e-12).then(1e-12).otherwise(pl.col("ltp").shift(-24).over(g)))
          - pl.col("__p_now")
        ).alias("__delta_label")
    ).drop_nulls(["__delta_label"])

    df_valid = df_valid.with_columns(
        ( (1.0 / pl.when(pl.col("ltp").shift(-24).over(g) < 1e-12).then(1e-12).otherwise(pl.col("ltp").shift(-24).over(g)))
          - pl.col("__p_now")
        ).alias("__delta_label")
    ).drop_nulls(["__delta_label"])

    X_tr = df_train.select(NUMERIC_FEATS).to_numpy()
    y_tr = df_train["__delta_label"].to_numpy()
    X_va = df_valid.select(NUMERIC_FEATS).to_numpy()
    y_va = df_valid["__delta_label"].to_numpy()

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=NUMERIC_FEATS)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=NUMERIC_FEATS)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
    }
    if args.device == "cuda":
        params["device"] = "cuda"

    evallist = [(dtr, "train"), (dva, "valid")]
    bst = xgb.train(params, dtr, num_boost_round=500, evals=evallist, verbose_eval=False)

    # quick validation EV snapshot (not a full sim)
    p_now = df_valid["__p_now"].to_numpy()
    dp = bst.predict(dva)
    p_pred = np.clip(p_now + dp, 0.0, 1.0)
    ltp = df_valid["ltp"].to_numpy()

    commission = float(args.commission)
    ev_back = p_pred * (ltp - 1.0) * (1.0 - commission)
    ev_lay = (1.0 - p_pred) - p_pred * (ltp - 1.0)
    ev = np.where(dp >= 0.0, ev_back, ev_lay)

    print(f"[trend] rows train={len(y_tr):,}  valid={len(y_va):,}")
    print(f"[trend] valid EV per £1: mean={ev.mean():.5f}  p>0 share={(ev>0).mean():.3f}")

    # save model
    model_dir = Path(args.output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgb_trend_reg.json"
    bst.save_model(str(model_path))
    print(f"[trend] saved model → {model_path}")

if __name__ == "__main__":
    main()
