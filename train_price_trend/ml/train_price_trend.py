#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train price-trend model (dp = p(t+H) - p(t)) for pre-off racing.

Highlights
- Skips missing days cleanly (no crash if yesterday isn't landed yet).
- Uses schema cache helper (schema_loader) to pin requested columns and
  avoid expensive schema discovery on weird days.
- Polars 1.33+ compatible; joins & features are fully vectorized (no .apply).
- XGBoost 2.x with GPU: {"tree_method":"hist","device":"cuda"} when requested.
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("train_price_trend (pre-off)")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    p.add_argument("--country-facet", action="store_true",
                   help="include countryCode as string facet column")
    return p.parse_args()

# ---------- schema helper ----------
def load_schema(kind: str) -> list[str]:
    # Lazy import so this file works even if helper missing
    try:
        from ml.schema_loader import load_schema as _load
        return _load(kind)
    except Exception:
        # minimal safe fallbacks
        if kind == "orderbook":
            return ["sport","marketId","selectionId","publishTimeMs",
                    "ltp","ltpTick","tradedVolume","spreadTicks","imbalanceBest1",
                    "backTicks","backSizes","layTicks","laySizes"]
        if kind == "marketdef":
            return ["sport","marketId","marketStartMs","countryCode"]
        if kind == "results":
            return ["sport","marketId","selectionId","runnerStatus","winLabel"]
        return []

def intersect_existing(requested: list[str], have: list[str]) -> list[str]:
    hs = set(have)
    return [c for c in requested if c in hs]

# ---------- date helpers ----------
def daterange(start: str, end: str) -> list[str]:
    sd = datetime.strptime(start, "%Y-%m-%d").date()
    ed = datetime.strptime(end, "%Y-%m-%d").date()
    return [(sd + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((ed - sd).days + 1)]

def curated_dirs(root: str, sport: str, date: str):
    base = Path(root)
    sdir = base / "orderbook_snapshots_5s" / f"sport={sport}" / f"date={date}"
    ddir = base / "market_definitions"      / f"sport={sport}" / f"date={date}"
    rdir = base / "results"                  / f"sport={sport}" / f"date={date}"
    return sdir, ddir, rdir

def list_parquets(dirpath: Path) -> list[str]:
    return sorted(glob.glob(str(dirpath / "*.parquet")))

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    # Polars >=1.25 prefers engine="streaming"
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

def scan_many(file_lists: list[list[str]], requested_cols: list[str], label: str) -> pl.DataFrame:
    files = [f for sub in file_lists for f in sub]
    if not files:
        return pl.DataFrame({c: [] for c in requested_cols})
    lf = pl.scan_parquet(files)
    have = lf.collect_schema().names()
    cols = intersect_existing(requested_cols, have)
    missing = [c for c in requested_cols if c not in have]
    if missing:
        print(f"[schema:{label}] missing {len(missing)}: {missing[:10]}{' …' if len(missing)>10 else ''}")
    return collect_streaming(lf.select([pl.col(c) for c in cols]))

# ---------- feature builder ----------
def build_features(df: pl.DataFrame, defs: pl.DataFrame,
                   preoff_max_m: int, horizon_s: int,
                   include_country: bool) -> pl.DataFrame:
    if df.is_empty():
        return df

    out = (
        df.join(defs, on=["sport","marketId"], how="left")
          .with_columns([
              ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off"),
              pl.col("ltp").cast(pl.Float64).alias("ltp_f"),
              pl.col("tradedVolume").cast(pl.Float64).alias("vol_f"),
          ])
          .filter(
              pl.col("mins_to_off").is_not_null()
              & (pl.col("mins_to_off") >= 0.0)
              & (pl.col("mins_to_off") <= float(preoff_max_m))
          )
          .sort(["marketId","selectionId","publishTimeMs"])
    )

    grp = ["marketId","selectionId"]
    out = out.with_columns([
        pl.col("ltp_f").diff().over(grp).alias("ltp_diff_5s"),
        pl.col("vol_f").diff().over(grp).alias("vol_diff_5s"),
        pl.col("ltp_f").shift(6).over(grp).alias("ltp_lag30s"),
        pl.col("ltp_f").shift(12).over(grp).alias("ltp_lag60s"),
        pl.col("ltp_f").shift(24).over(grp).alias("ltp_lag120s"),
        pl.col("vol_f").shift(6).over(grp).alias("tradedVolume_lag30s"),
        pl.col("vol_f").shift(12).over(grp).alias("tradedVolume_lag60s"),
        pl.col("vol_f").shift(24).over(grp).alias("tradedVolume_lag120s"),
    ]).with_columns([
        (pl.col("ltp_f") - pl.col("ltp_lag30s")).alias("ltp_mom_30s"),
        (pl.col("ltp_f") - pl.col("ltp_lag60s")).alias("ltp_mom_60s"),
        (pl.col("ltp_f") - pl.col("ltp_lag120s")).alias("ltp_mom_120s"),
        ((pl.col("ltp_f") / pl.col("ltp_lag30s")) - 1.0).alias("ltp_ret_30s"),
        ((pl.col("ltp_f") / pl.col("ltp_lag60s")) - 1.0).alias("ltp_ret_60s"),
        ((pl.col("ltp_f") / pl.col("ltp_lag120s")) - 1.0).alias("ltp_ret_120s"),
        (pl.col("publishTimeMs") + horizon_s * 1000).alias("ts_exit_ms"),
    ])

    if include_country:
        if "countryCode" not in out.columns:
            out = out.with_columns(pl.lit("UNK").alias("countryCode"))
    else:
        if "countryCode" not in out.columns:
            out = out.with_columns(pl.lit("UNK").alias("countryCode"))

    # dynamic horizon -> steps = horizon/5s
    steps = max(1, int(round(horizon_s / 5)))
    out = out.with_columns([
        pl.col("ltp_f").shift(-steps).over(grp).alias("ltp_future")
    ]).with_columns([
        (pl.col("ltp_future") - pl.col("ltp_f")).alias("dp_target")
    ])
    out = out.filter(pl.col("dp_target").is_not_null())
    return out

# ---------- data prep ----------
def prepare_train_valid(curated: str, sport: str,
                        start_date: str, asof: str, valid_days: int,
                        include_country: bool, preoff_max: int, horizon_s: int):
    ad = datetime.strptime(asof, "%Y-%m-%d").date()
    vstart = (ad - timedelta(days=valid_days - 1))
    train_end = (vstart - timedelta(days=1))
    tr_dates = daterange(start_date, train_end.strftime("%Y-%m-%d"))
    va_dates = daterange(vstart.strftime("%Y-%m-%d"), asof)

    want_order = load_schema("orderbook")
    want_defs  = load_schema("marketdef")

    def collect_set(dates: list[str], label: str):
        snap_files, def_files, missing = [], [], []
        for d in dates:
            sdir, ddir, _ = curated_dirs(curated, sport, d)
            sfiles = list_parquets(sdir)
            dfiles = list_parquets(ddir)
            if not sfiles:
                missing.append(d); continue
            snap_files.append(sfiles)
            def_files.append(dfiles)
        if missing:
            print(f"[{label}] Skipping {len(missing)} missing day(s): {', '.join(missing[:6])}{' …' if len(missing)>6 else ''}")
        snaps = scan_many(snap_files, want_order, "orderbook")
        defs  = scan_many(def_files,  want_defs,  "marketdef")
        # ensure columns exist
        if "marketStartMs" not in defs.columns: defs = defs.with_columns(pl.lit(None).alias("marketStartMs"))
        if "countryCode"  not in defs.columns:  defs = defs.with_columns(pl.lit("UNK").alias("countryCode"))
        return snaps, defs

    snaps_tr, defs_tr = collect_set(tr_dates, "TRAIN")
    snaps_va, defs_va = collect_set(va_dates, "VALID")

    if snaps_tr.is_empty() and snaps_va.is_empty():
        raise SystemExit("No snapshot files found in requested window. Choose an earlier ASOF or check curated folders.")

    f_tr = build_features(snaps_tr, defs_tr, preoff_max, horizon_s, include_country) if not snaps_tr.is_empty() else pl.DataFrame()
    f_va = build_features(snaps_va, defs_va, preoff_max, horizon_s, include_country) if not snaps_va.is_empty() else pl.DataFrame()

    if not f_tr.is_empty() and not f_va.is_empty():
        print(f"[trend] TRAIN: {tr_dates[0]} .. {tr_dates[-1]}")
        print(f"[trend] VALID: {va_dates[0]} .. {va_dates[-1]}")
    elif not f_tr.is_empty():
        print("[trend] VALID window empty after skipping missing days; training only.")
    return f_tr, f_va

# ---------- model ----------
def fit_xgb_reg(train_df: pl.DataFrame, device: str):
    if xgb is None or train_df.is_empty():
        return None, []
    # drop non-feature cols
    drop = {
        "sport","marketId","selectionId","publishTimeMs",
        "marketStartMs","countryCode","ts_exit_ms","dp_target","ltp_future"
    }
    features = [c for c in train_df.columns if c not in drop]
    X = train_df.select(features).to_pandas()
    y = train_df["dp_target"].to_pandas()

    params = {
        "max_depth": 6,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }
    if device == "cuda":
        params["device"] = "cuda"

    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(params, dtrain, num_boost_round=300)
    return booster, features

def eval_valid(booster, features: list[str], valid_df: pl.DataFrame):
    if booster is None or valid_df.is_empty():
        return 0.0, 0.0
    Xv = valid_df.select([c for c in features if c in valid_df.columns]).to_pandas()
    dv = xgb.DMatrix(Xv)
    dp_pred = booster.predict(dv)
    ev = dp_pred.mean() if len(dp_pred) else 0.0
    p_pos = float((dp_pred > 0).mean()) if len(dp_pred) else 0.0
    return ev, p_pos

# ---------- main ----------
def main():
    args = parse_args()
    print("=== Price Trend Training ===")
    print(f"Curated root:    {args.curated}")
    print(f"ASOF:            {args.asof}")
    print(f"Start date:      {args.start_date}")
    print(f"Valid days:      {args.valid_days}")
    print(f"Horizon (secs):  {args.horizon_secs}")
    print(f"Pre-off max (m): {args.preoff_max}")
    print(f"XGBoost device:  {args.device}")
    print(f"EV mode:         mtm")

    f_tr, f_va = prepare_train_valid(args.curated, args.sport,
                                     args.start_date, args.asof, args.valid_days,
                                     args.country_facet, args.preoff_max, args.horizon_secs)

    print(f"[trend] horizon={args.horizon_secs}s  preoff≤{args.preoff_max}m  device={args.device}")
    print(f"[trend] rows train={(f_tr.height if not f_tr.is_empty() else 0):,d}  valid={(f_va.height if not f_va.is_empty() else 0):,d}")

    booster, feats = fit_xgb_reg(f_tr, args.device)
    if booster is not None:
        ev_mean, p_pos = eval_valid(booster, feats, f_va)
        print(f"[trend] valid EV per £1: mean={ev_mean:.5f}  p>0 share={p_pos:.3f}")

    outdir = Path(args.output_dir) / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    mp = outdir / "xgb_trend_reg.json"

    if booster is not None:
        booster.save_model(str(mp))
        print(f"[trend] saved model → {mp}")
    else:
        mp.write_text('{"note":"no-xgb; proxy-only"}')
        print(f"[trend] WARNING: xgboost not available; wrote sentinel model to {mp}")

if __name__ == "__main__":
    main()
