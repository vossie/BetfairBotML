#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from utils import (
    parse_date, to_ms, read_snapshots,
    implied_prob_from_ltp_expr, time_to_off_minutes_expr, filter_preoff,
    write_json, kelly_fraction_back, kelly_fraction_lay
)
from build_features import add_core_features, add_label_delta_prob

UTC = timezone.utc
NUMERIC_DT = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

def parse_args():
    p = argparse.ArgumentParser(description="Price Trend Trainer (edge-based, stake scaling)")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True, help="YYYY-MM-DD (inclusive valid end)")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD (training start)")
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--edge-thresh", type=float, default=0.0)
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
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

    # Load
    lf = read_snapshots(curated, train_start, valid_end, args.sport).with_columns([
        implied_prob_from_ltp_expr("ltp"),
        time_to_off_minutes_expr(),
    ])
    df = lf.collect()

    # 0–preoff_max min window
    df = filter_preoff(df, args.preoff_max)

    # Features + label
    df = add_core_features(df)
    df = add_label_delta_prob(df, args.horizon_secs)
    df = df.filter(pl.col("delta_p").is_finite() & pl.col("__p_now").is_finite())

    # Split by time
    train_lo, train_hi = to_ms(train_start), to_ms(train_end + timedelta(days=1))
    valid_lo, valid_hi = to_ms(valid_start), to_ms(valid_end + timedelta(days=1))
    df_train = df.filter((pl.col("publishTimeMs") >= train_lo) & (pl.col("publishTimeMs") < train_hi))
    df_valid = df.filter((pl.col("publishTimeMs") >= valid_lo) & (pl.col("publishTimeMs") < valid_hi))

    # Feature list (numeric only)
    exclude = {"sport","marketId","__p_future"}
    feats = [
        c for c, dt in zip(df.columns, df.dtypes)
        if c not in exclude and c != "delta_p" and dt in NUMERIC_DT
    ]
    target = "delta_p"

    print(f"[trend] rows: train={df_train.height:,}  valid={df_valid.height:,}  features={len(feats)}")

    # Train regressor on delta_p
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

    # Validate
    pred_valid = bst.predict(dvalid)
    rmse = mean_squared_error(df_valid[target].to_numpy(), pred_valid, squared=False)
    print(f"[trend] RMSE(delta_p)={rmse:.6f}")

    # Edge-based decisions
    valid = df_valid.select(["marketId","selectionId","publishTimeMs","ltp","__p_now"]).with_columns(
        pl.Series("__delta_pred", pred_valid)
    )
    valid = valid.with_columns((pl.col("__p_now") + pl.col("__delta_pred")).alias("__p_pred"))
    valid = valid.with_columns([
        pl.when(pl.col("__delta_pred") > 0).then(pl.lit("back"))
         .when(pl.col("__delta_pred") < 0).then(pl.lit("lay"))
         .otherwise(pl.lit("none")).alias("__dir")
    ])

    commission = float(args.commission)
    valid = valid.with_columns([
        # EV per £1 stake (approx)
        (pl.col("__p_pred") * (pl.col("ltp") - 1.0) * (1.0 - commission)).alias("__ev_back"),
        (((1.0 - pl.col("__p_pred")) - pl.col("__p_pred") * (pl.col("ltp") - 1.0))).alias("__ev_lay"),
    ])
    valid = valid.with_columns([
        pl.when(pl.col("__dir")=="back").then(pl.col("__ev_back"))
          .when(pl.col("__dir")=="lay").then(pl.col("__ev_lay"))
          .otherwise(pl.lit(0.0)).alias("__ev_per_1")
    ])
    valid = valid.filter(pl.col("__dir")!="none")
    if args.edge_thresh > 0.0:
        valid = valid.filter(pl.col("__ev_per_1") >= args.edge_thresh)

    # Stake sizing
    if args.stake_mode == "flat":
        valid = valid.with_columns(pl.lit(1.0).alias("__stake"))
    else:
        # Kelly-like (capped/floored) using map_rows to mix back/lay formulas
        def _kelly_map(p: float, o: float, side: str) -> float:
            return (kelly_fraction_back(p, o, commission) if side == "back"
                    else kelly_fraction_lay(p, o, commission))

        valid = valid.with_columns(
            pl.map_elements(
                ["__p_pred","ltp","__dir"],
                lambda p,o,d: _kelly_map(p,o,d),
                return_dtype=pl.Float64
            ).alias("__kelly_raw")
        )
        valid = valid.with_columns(
            (pl.when(pl.col("__kelly_raw") < args.kelly_floor).then(args.kelly_floor)
               .when(pl.col("__kelly_raw") > args.kelly_cap).then(args.kelly_cap)
               .otherwise(pl.col("__kelly_raw")) * float(args.bankroll_nom)).alias("__stake")
        )

    valid = valid.with_columns((pl.col("__ev_per_1") * pl.col("__stake")).alias("__pnl_expect"))

    summary = valid.select([
        pl.count().alias("n_trades"),
        pl.sum("__pnl_expect").alias("total_exp_profit"),
        pl.mean("__ev_per_1").alias("avg_ev_per_1"),
        pl.mean("__stake").alias("avg_stake"),
        pl.median("__stake").alias("med_stake"),
    ]).collect()
    print(summary)

    # Save
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
        "features": feats,
    })

    # Daily breakdown
    daily = valid.with_columns(
        pl.from_epoch(pl.col("publishTimeMs").cast(pl.Int64), time_unit="ms")
          .dt.replace_time_zone("UTC").dt.date().cast(pl.Utf8).alias("__day")
    ).group_by("__day").agg([
        pl.count().alias("n_trades"),
        pl.sum("__pnl_expect").alias("exp_profit"),
        pl.mean("__ev_per_1").alias("avg_ev"),
    ]).sort("__day")
    dpath = Path(args.output_dir) / f"trend_daily_{args.asof}.csv"
    daily.collect().write_csv(dpath)
    print(f"[trend] daily saved → {dpath}")

if __name__ == "__main__":
    main()
