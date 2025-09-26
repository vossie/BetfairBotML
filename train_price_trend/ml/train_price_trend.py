#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

from utils import parse_date, to_ms, read_snapshots, implied_prob_from_ltp_expr, time_to_off_minutes_expr, filter_preoff, write_json, kelly_fraction_back, kelly_fraction_lay, clip_float
from build_features import add_core_features, add_label_delta_prob

UTC = timezone.utc

def parse_args():
    p = argparse.ArgumentParser(description="Price Trend Trainer (edge-based, stake scaling)")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True, help="YYYY-MM-DD (inclusive in valid end)")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD (training start)")
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--horizon-secs", type=int, default=120, help="label horizon in seconds")
    p.add_argument("--preoff-max", type=int, default=30, help="minutes to off upper bound")
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--edge-thresh", type=float, default=0.0, help="EV per £1 threshold to place trade")
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    p.add_argument("--downsample-secs", type=int, default=5, help="assumed snapshot step for lags (5s default)")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    curated = Path(args.curated)

    asof_dt = parse_date(args.asof)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)
    train_start = parse_date(args.start_date)
    train_end = valid_start - timedelta(days=1)

    print(f"[trend] TRAIN: {train_start.date()} .. {train_end.date()}")
    print(f"[trend] VALID: {valid_start.date()} .. {valid_end.date()}")
    print(f"[trend] horizon={args.horizon_secs}s  preoff≤{args.preoff_max}m  device={args.device}")

    # ---- Load
    lf = read_snapshots(curated, train_start, valid_end, args.sport)
    lf = lf.with_columns([
        implied_prob_from_ltp_expr("ltp"),
        time_to_off_minutes_expr()
    ])

    # Collect once, then feature/label
    df = lf.collect()

    # Restrict to pre-off window
    df = filter_preoff(df, args.preoff_max)

    # Build features and label
    df = add_core_features(df)
    df = add_label_delta_prob(df, args.horizon_secs)

    # Drop rows without label/ltp
    df = df.filter(pl.col("delta_p").is_finite() & pl.col("__p_now").is_finite())

    # Split train/valid by publishTimeMs
    train_lo, train_hi = to_ms(train_start), to_ms(train_end + timedelta(days=1))
    valid_lo, valid_hi = to_ms(valid_start), to_ms(valid_end + timedelta(days=1))
    df_train = df.filter((pl.col("publishTimeMs") >= train_lo) & (pl.col("publishTimeMs") < train_hi))
    df_valid = df.filter((pl.col("publishTimeMs") >= valid_lo) & (pl.col("publishTimeMs") < valid_hi))

    # Feature selection: keep numeric engineered cols + some microstructure
    exclude = {"sport","marketId","__p_future"}
    feats = [c for c, dt in zip(df.columns, df.dtypes)
             if c not in exclude and c not in ("delta_p",) and dt.is_numeric()]
    target = "delta_p"

    print(f"[trend] rows: train={df_train.height:,}  valid={df_valid.height:,}  features={len(feats)}")

    # ---- Train XGB regressor (predict delta_p)
    params = {
        "device": args.device,
        "tree_method": "hist",
        "eval_metric": "rmse",
        "max_depth": 8,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
    }

    dtrain = xgb.DMatrix(df_train.select(feats).to_arrow(), label=df_train[target].to_numpy())
    dvalid = xgb.DMatrix(df_valid.select(feats).to_arrow(), label=df_valid[target].to_numpy())
    bst = xgb.train(params, dtrain, num_boost_round=400, evals=[(dtrain,"train"),(dvalid,"valid")], verbose_eval=False)

    pred_valid = bst.predict(dvalid)
    rmse = mean_squared_error(df_valid[target].to_numpy(), pred_valid, squared=False)
    print(f"[trend] RMSE(delta_p)={rmse:.6f}")

    # ---- Edge-based backtest (instantaneous fill approximation)
    # Use p_pred = p_now + delta_pred; decide back/lay and size stake.
    valid = df_valid.select(["marketId","selectionId","publishTimeMs","ltp","__p_now"]).with_columns(
        pl.Series("__delta_pred", pred_valid)
    )
    valid = valid.with_columns((pl.col("__p_now") + pl.col("__delta_pred")).alias("__p_pred"))
    valid = valid.with_columns([
        pl.when(pl.col("__delta_pred") > 0).then(pl.lit("back"))
         .when(pl.col("__delta_pred") < 0).then(pl.lit("lay"))
         .otherwise(pl.lit("none")).alias("__dir")
    ])

    # Expected value per £1 (approx)
    # back EV: p_pred*(o-1)*(1-c) - (1-p_pred)*0
    # lay EV per £ backer stake: (1-p_pred)*1 - p_pred*(o-1)
    commission = float(args.commission)
    valid = valid.with_columns([
        ((pl.col("__p_pred") * (pl.col("ltp") - 1.0) * (1.0 - commission))).alias("__ev_back"),
        (((1.0 - pl.col("__p_pred")) - pl.col("__p_pred") * (pl.col("ltp") - 1.0))).alias("__ev_lay")
    ])

    # Choose EV based on direction
    valid = valid.with_columns([
        pl.when(pl.col("__dir")=="back").then(pl.col("__ev_back"))
          .when(pl.col("__dir")=="lay").then(pl.col("__ev_lay"))
          .otherwise(pl.lit(0.0)).alias("__ev_per_1")
    ])

    # Threshold on EV
    valid = valid.filter(pl.col("__dir")!="none")
    if args.edge_thresh > 0:
        valid = valid.filter(pl.col("__ev_per_1") >= args.edge_thresh)

    # Stake sizing
    if args.stake_mode == "flat":
        stake = 1.0
        valid = valid.with_columns(pl.lit(stake).alias("__stake"))
    else:
        # Kelly-like, capped/floored
        def kelly_col():
            return pl.when(pl.col("__dir")=="back").then(
                pl.map_rows(lambda p, o: kelly_fraction_back(p, o, commission), return_dtype=pl.Float64, exprs=["__p_pred","ltp"])
            ).when(pl.col("__dir")=="lay").then(
                pl.map_rows(lambda p, o: kelly_fraction_lay(p, o, commission), return_dtype=pl.Float64, exprs=["__p_pred","ltp"])
            ).otherwise(pl.lit(0.0))
        valid = valid.with_columns(kelly_col().alias("__kelly_raw"))
        valid = valid.with_columns(
            (pl.col("__kelly_raw").clip_min(args.kelly_floor).clip_max(args.kelly_cap) * float(args.bankroll_nom)).alias("__stake")
        )

    # PnL approximation using p_pred as proxy of true (for offline selection check)
    # back realized EV per £stake: __ev_back ; lay: __ev_lay
    valid = valid.with_columns((pl.col("__ev_per_1") * pl.col("__stake")).alias("__pnl_expect"))

    # Aggregate
    summary = valid.select([
        pl.count().alias("n_trades"),
        pl.sum("__pnl_expect").alias("total_exp_profit"),
        pl.mean("__ev_per_1").alias("avg_ev_per_1"),
        pl.mean("__stake").alias("avg_stake"),
        pl.median("__stake").alias("med_stake")
    ]).collect()

    print(summary)

    # Save artifacts
    model_dir = Path(args.output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(model_dir / "xgb_trend_reg.json"))

    write_json(Path(args.output_dir) / "run_meta.json", {
        "asof": args.asof,
        "start_date": args.start_date,
        "valid_days": args.valid_days,
        "horizon_secs": args.horizon_secs,
        "preoff_max_minutes": args.preoff_max,
        "commission": args.commission,
        "edge_thresh": args.edge_thresh,
        "stake_mode": args.stake_mode,
        "kelly_cap": args.kelly_cap,
        "kelly_floor": args.kelly_floor,
        "bankroll_nom": args.bankroll_nom,
        "features": feats
    })

    # Daily breakdown
    daily = valid.with_columns(
        (pl.from_epoch((pl.col("publishTimeMs")/1000).cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC").dt.date().cast(pl.Utf8)).alias("__day")
    ).group_by("__day").agg([
        pl.count().alias("n_trades"),
        pl.sum("__pnl_expect").alias("exp_profit"),
        pl.mean("__ev_per_1").alias("avg_ev")
    ]).sort("__day")
    daily_path = Path(args.output_dir) / f"trend_daily_{args.asof}.csv"
    daily.collect().write_csv(daily_path)
    print(f"[trend] daily saved → {daily_path}")

if __name__ == "__main__":
    main()
