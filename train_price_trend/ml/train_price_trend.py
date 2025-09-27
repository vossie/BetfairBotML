#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import timedelta
import numpy as np
import polars as pl
import xgboost as xgb

from utils import (
    parse_date, read_snapshots,
    time_to_off_minutes_expr, filter_preoff,
)

UTC_TZ = "UTC"

LAG_30S = 6
LAG_60S = 12
LAG_120S = 24

BASE_FEATS = [
    "ltp","tradedVolume","spreadTicks","imbalanceBest1","mins_to_off","__p_now",
    "ltp_diff_5s","vol_diff_5s",
    "ltp_lag30s","ltp_lag60s","ltp_lag120s",
    "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
    "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
    "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
]

def parse_args():
    p = argparse.ArgumentParser(description="Train price-trend delta model (regression) with country facets")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    p.add_argument("--country-facet", type=int, default=1, help="1=include country facets (is_gb, country_freq)")
    return p.parse_args()

def _build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["marketId","selectionId","publishTimeMs"])
    def add_group_feats(g: pl.DataFrame) -> pl.DataFrame:
        g = g.with_columns([
            (pl.col("ltp") - pl.col("ltp").shift(1)).alias("ltp_diff_5s"),
            (pl.col("tradedVolume") - pl.col("tradedVolume").shift(1)).alias("vol_diff_5s"),
            pl.col("ltp").shift(LAG_30S).alias("ltp_lag30s"),
            pl.col("ltp").shift(LAG_60S).alias("ltp_lag60s"),
            pl.col("ltp").shift(LAG_120S).alias("ltp_lag120s"),
            pl.col("tradedVolume").shift(LAG_30S).alias("tradedVolume_lag30s"),
            pl.col("tradedVolume").shift(LAG_60S).alias("tradedVolume_lag60s"),
            pl.col("tradedVolume").shift(LAG_120S).alias("tradedVolume_lag120s"),
            (pl.col("ltp") - pl.col("ltp").shift(LAG_30S)).alias("ltp_mom_30s"),
            (pl.col("ltp") - pl.col("ltp").shift(LAG_60S)).alias("ltp_mom_60s"),
            (pl.col("ltp") - pl.col("ltp").shift(LAG_120S)).alias("ltp_mom_120s"),
            ((pl.col("ltp") - pl.col("ltp").shift(LAG_30S)) / pl.col("ltp").shift(LAG_30S)).alias("ltp_ret_30s"),
            ((pl.col("ltp") - pl.col("ltp").shift(LAG_60S)) / pl.col("ltp").shift(LAG_60S)).alias("ltp_ret_60s"),
            ((pl.col("ltp") - pl.col("ltp").shift(LAG_120S)) / pl.col("ltp").shift(LAG_120S)).alias("ltp_ret_120s"),
            (1.0 / pl.col("ltp")).alias("__p_now"),
        ])
        return g
    return df.group_by(["marketId","selectionId"], maintain_order=True).apply(add_group_feats)

def _future_join(df: pl.DataFrame, horizon_secs: int) -> pl.DataFrame:
    df2 = df.with_columns([
        (pl.col("publishTimeMs").cast(pl.Int64) + horizon_secs * 1000).alias("ts_exit_ms"),
        pl.from_epoch(pl.col("publishTimeMs"), time_unit="ms").dt.replace_time_zone(UTC_TZ).alias("ts_dt"),
        pl.from_epoch(pl.col("publishTimeMs") + horizon_secs * 1000, time_unit="ms").dt.replace_time_zone(UTC_TZ).alias("ts_exit_dt"),
    ]).sort(["marketId","selectionId","ts_exit_dt"])
    exit_df = (
        df.select([
            pl.col("marketId"), pl.col("selectionId"),
            pl.from_epoch(pl.col("publishTimeMs"), time_unit="ms").dt.replace_time_zone(UTC_TZ).alias("ts_dt"),
            pl.col("ltp").alias("ltp_exit_proxy"),
        ])
        .filter(pl.col("ltp_exit_proxy").is_not_null() & (pl.col("ltp_exit_proxy") > 1.01))
        .sort(["marketId","selectionId","ts_dt"])
    )
    out = df2.join_asof(
        exit_df, left_on="ts_exit_dt", right_on="ts_dt",
        by=["marketId","selectionId"], strategy="forward",
        tolerance=timedelta(minutes=10),
    ).with_columns(
        pl.col("ltp_exit_proxy").alias("ltp_future")
    ).drop(["ts_dt","ts_exit_dt","ltp_exit_proxy"])
    return out

def _ev_mtm_per1(p_now: np.ndarray, p_pred: np.ndarray, commission: float, side_back_mask: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p_now_c = np.clip(p_now, eps, 1.0 - eps)
    p_pred_c = np.clip(p_pred, eps, 1.0 - eps)
    ev_back = (p_pred_c / p_now_c - 1.0) * (1.0 - commission)
    ev_lay  = (1.0 - (p_now_c / p_pred_c)) * (1.0 - commission)
    return np.where(side_back_mask, ev_back, ev_lay)

def main():
    args = parse_args()
    curated = Path(args.curated)

    asof_dt = parse_date(args.asof)
    start_dt = parse_date(args.start_date)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)

    print("=== Price Trend Training ===")
    print(f"Curated root:    {curated}")
    print(f"ASOF:            {args.asof}")
    print(f"Start date:      {args.start_date}")
    print(f"Valid days:      {args.valid_days}")
    print(f"Horizon (secs):  {args.horizon_secs}")
    print(f"Pre-off max (m): {args.preoff_max}")
    print(f"XGBoost device:  {args.device}")
    print(f"EV mode:         mtm")

    cols = ["marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","marketStartMs","countryCode"]
    lf = read_snapshots(curated, start_dt, valid_end, args.sport)
    schema = lf.collect_schema().names()
    present = [c for c in cols if c in schema]
    lf = lf.select(present).with_columns([ time_to_off_minutes_expr() ])

    df = lf.collect()
    df = df.filter(pl.col("ltp").is_not_null() & (pl.col("ltp") > 1.01))
    df = filter_preoff(df, args.preoff_max)

    df = df.with_columns(
        pl.from_epoch(pl.col("publishTimeMs"), time_unit="ms").dt.replace_time_zone(UTC_TZ).dt.date().alias("__date")
    )
    print(f"[trend] TRAIN: {start_dt} .. {valid_start}")
    print(f"[trend] VALID: {valid_start} .. {valid_end}")

    df_tr = df.filter(pl.col("__date") < valid_start)
    df_va = df.filter((pl.col("__date") >= valid_start) & (pl.col("__date") <= valid_end))

    def prep(df_in: pl.DataFrame) -> pl.DataFrame:
        f = _build_features(df_in)
        f = _future_join(f, args.horizon_secs)
        f = f.filter(pl.col("ltp_future").is_not_null() & (pl.col("ltp_future") > 1.01))
        f = f.with_columns([
            (1.0 / pl.col("ltp")).alias("p_now"),
            (1.0 / pl.col("ltp_future")).alias("p_future"),
            (pl.col("p_future") - pl.col("p_now")).alias("dp"),
        ])
        return f

    df_tr_f = prep(df_tr)
    df_va_f = prep(df_va)

    feat_extra: list[str] = []
    if args.country_facet and "countryCode" in df_tr_f.columns:
        df_tr_f = df_tr_f.with_columns((pl.col("countryCode") == "GB").cast(pl.Int8).alias("is_gb"))
        df_va_f = df_va_f.with_columns((pl.col("countryCode") == "GB").cast(pl.Int8).alias("is_gb"))
        freq = (
            df_tr_f.group_by("countryCode").agg(pl.len().alias("cnt"))
            .with_columns((pl.col("cnt") / pl.col("cnt").sum()).alias("freq"))
            .select(["countryCode","freq"])
        )
        df_tr_f = df_tr_f.join(freq, on="countryCode", how="left").with_columns(pl.col("freq").fill_null(0.0).alias("country_freq"))
        df_va_f = df_va_f.join(freq, on="countryCode", how="left").with_columns(pl.col("freq").fill_null(0.0).alias("country_freq"))
        feat_extra = ["is_gb","country_freq"]

    feature_cols = BASE_FEATS + feat_extra

    def to_xy(df_feat: pl.DataFrame):
        X = df_feat.select([pl.col(c).cast(pl.Float64) for c in feature_cols]).to_numpy()
        y = df_feat["dp"].to_numpy()
        return X, y

    X_tr, y_tr = to_xy(df_tr_f)
    X_va, y_va = to_xy(df_va_f)

    params = {
        "max_depth": 6,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": args.device,
        "predictor": "auto",
        "objective": "reg:squarederror",
    }
    bst = xgb.XGBRegressor(**params)
    bst.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    dp_pred = bst.predict(X_va)
    p_now = df_va_f["p_now"].to_numpy()
    p_pred = np.clip(p_now + dp_pred, 1e-6, 1.0-1e-6)
    side_back = dp_pred >= 0.0
    ev = _ev_mtm_per1(p_now, p_pred, args.commission, side_back)

    print(f"[trend] rows train={len(y_tr):,}  valid={len(y_va):,}")
    print(f"[trend] valid EV per £1: mean={ev.mean():.5f}  p>0 share={(ev>0).mean():.3f}")
    model_dir = Path(args.output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgb_trend_reg.json"
    bst.get_booster().save_model(str(model_path))
    print(f"[trend] saved model → {model_path}")

if __name__ == "__main__":
    main()
