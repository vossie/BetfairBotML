#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-safe trainer for price trend (dp = ltp(t+H) - ltp(t)).

Key features to avoid OOM:
- Per-day streaming feature build (no giant in-memory concat).
- Optional downsample of snapshot grid (e.g., keep every 10s).
- Optional sampling and row cap for train/valid.
- Float32 everywhere.
- GPU: QuantileDMatrix for hist device; CPU: standard DMatrix.

Polars >=1.33; XGBoost >=2.x recommended.
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import polars as pl

try:
    import xgboost as xgb
except Exception:
    xgb = None

# --------------- CLI ---------------
def parse_args():
    p = argparse.ArgumentParser("train_price_trend (memory-safe)")
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
    p.add_argument("--country-facet", action="store_true", help="include countryCode")

    # NEW: memory/runtime control
    p.add_argument("--downsample-secs", type=int, default=0,
                   help="Downsample snapshots: keep >= this step (0=off, e.g., 10 keeps 10s grid).")
    p.add_argument("--sample-frac", type=float, default=1.0, help="Random sample fraction for training (0<frac<=1).")
    p.add_argument("--max-train-rows", type=int, default=0, help="Hard cap on train rows (0=off).")
    p.add_argument("--max-valid-rows", type=int, default=250_000, help="Hard cap on valid rows.")
    return p.parse_args()

# --------------- schema helpers ---------------
def load_schema(kind: str) -> list[str]:
    try:
        from ml.schema_loader import load_schema as _load
        return _load(kind)
    except Exception:
        if kind == "orderbook":
            return ["sport","marketId","selectionId","publishTimeMs","ltp","ltpTick",
                    "tradedVolume","spreadTicks","imbalanceBest1"]
        if kind == "marketdef":
            return ["sport","marketId","marketStartMs","countryCode"]
        return []

def intersect_existing(requested: list[str], have: list[str]) -> list[str]:
    hs = set(have)
    return [c for c in requested if c in hs]

# --------------- dates/paths ---------------
def daterange(start: str, end: str) -> list[str]:
    sd = datetime.strptime(start, "%Y-%m-%d").date()
    ed = datetime.strptime(end, "%Y-%m-%d").date()
    return [(sd + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((ed - sd).days + 1)]

def curated_dirs(root: str, sport: str, date: str) -> Tuple[Path,Path]:
    base = Path(root)
    sdir = base / "orderbook_snapshots_5s" / f"sport={sport}" / f"date={date}"
    ddir = base / "market_definitions"      / f"sport={sport}" / f"date={date}"
    return sdir, ddir

def list_parquets(dirpath: Path) -> list[str]:
    return sorted(glob.glob(str(dirpath / "*.parquet")))

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

# --------------- IO per day ---------------
def read_day(curated: str, sport: str, date: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    want_order = load_schema("orderbook")
    want_defs  = load_schema("marketdef")
    sdir, ddir = curated_dirs(curated, sport, date)

    # snapshots
    snaps = pl.DataFrame({})
    sfiles = list_parquets(sdir)
    if sfiles:
        lf = pl.scan_parquet(sfiles)
        have = lf.collect_schema().names()
        cols = intersect_existing(want_order, have)
        if cols:
            snaps = collect_streaming(lf.select([pl.col(c) for c in cols]))

    # market defs
    defs = pl.DataFrame({})
    dfiles = list_parquets(ddir)
    if dfiles:
        lf = pl.scan_parquet(dfiles)
        have = lf.collect_schema().names()
        cols = intersect_existing(want_defs, have)
        if cols:
            defs = collect_streaming(lf.select([pl.col(c) for c in cols]))
    if "marketStartMs" not in defs.columns:
        defs = defs.with_columns(pl.lit(None).alias("marketStartMs"))
    if "countryCode" not in defs.columns:
        defs = defs.with_columns(pl.lit("UNK").alias("countryCode"))
    return snaps, defs

# --------------- features ---------------
def build_features(df: pl.DataFrame, defs: pl.DataFrame,
                   preoff_max_m: int, horizon_s: int,
                   include_country: bool,
                   downsample_secs: int) -> pl.DataFrame:
    if df.is_empty(): return df

    out = (
        df.join(defs, on=["sport","marketId"], how="left")
          .with_columns([
              ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off"),
              pl.col("ltp").cast(pl.Float32).alias("ltp_f"),
              pl.col("tradedVolume").cast(pl.Float32).alias("vol_f"),
          ])
          .filter(pl.col("mins_to_off").is_not_null() & (pl.col("mins_to_off") >= 0.0) & (pl.col("mins_to_off") <= float(preoff_max_m)))
          .sort(["marketId","selectionId","publishTimeMs"])
    )

    # optional downsample: keep rows near a coarser grid
    if downsample_secs and downsample_secs > 0:
        step = max(1, int(round(downsample_secs / 5)))
        out = out.with_columns(((pl.col("publishTimeMs") // 5000) % step).alias("__mod"))
        out = out.filter(pl.col("__mod")==0).drop("__mod")

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

    if include_country and "countryCode" not in out.columns:
        out = out.with_columns(pl.lit("UNK").alias("countryCode"))

    steps = max(1, int(round(horizon_s / 5)))
    out = out.with_columns([pl.col("ltp_f").shift(-steps).over(grp).alias("ltp_future")])
    out = out.with_columns([(pl.col("ltp_future") - pl.col("ltp_f")).alias("dp_target")])
    out = out.filter(pl.col("dp_target").is_not_null())

    # keep only needed columns for XGB
    keep = [
        "marketId","selectionId","publishTimeMs","marketStartMs","countryCode",
        "ltp_f","vol_f","ltp_diff_5s","vol_diff_5s",
        "ltp_lag30s","ltp_lag60s","ltp_lag120s",
        "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
        "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
        "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
        "mins_to_off","ltp_future","dp_target"
    ]
    present = [c for c in keep if c in out.columns]
    out = out.select(present)
    return out

# --------------- prep sets (bounded) ---------------
def prepare_sets(args):
    ad = datetime.strptime(args.asof, "%Y-%m-%d").date()
    vstart = (ad - timedelta(days=args.valid_days - 1))
    train_end = (vstart - timedelta(days=1))
    tr_dates = daterange(args.start_date, train_end.strftime("%Y-%m-%d"))
    va_dates = daterange(vstart.strftime("%Y-%m-%d"), args.asof)

    def collect_dates(dates: List[str], label: str) -> pl.DataFrame:
        parts: List[pl.DataFrame] = []
        missing = []
        for d in dates:
            snaps, defs = read_day(args.curated, args.sport, d)
            if snaps.is_empty():
                missing.append(d); continue
            feat = build_features(snaps, defs, args.preoff_max, args.horizon_secs,
                                  args.country_facet, args.downsample_secs)
            if not feat.is_empty():
                parts.append(feat)
        if missing:
            print(f"[{label}] skipped {len(missing)} empty day(s): {', '.join(missing[:6])}{' …' if len(missing)>6 else ''}")
        if not parts:
            return pl.DataFrame({})
        # vertical concat in streaming engine
        try:
            df = pl.concat(parts, how="vertical_relaxed").collect(engine="streaming")  # if parts are LazyFrames
        except Exception:
            df = pl.concat(parts, how="vertical_relaxed")
        return df

    df_tr = collect_dates(tr_dates, "TRAIN")
    df_va = collect_dates(va_dates, "VALID")

    # sampling / caps
    if not df_tr.is_empty():
        if args.sample_frac < 1.0:
            df_tr = df_tr.sample(frac=args.sample_frac, with_replacement=False, shuffle=True, seed=42)
        if args.max_train_rows and df_tr.height > args.max_train_rows:
            df_tr = df_tr.sample(n=args.max_train_rows, with_replacement=False, shuffle=True, seed=42)

    if not df_va.is_empty():
        if args.max_valid_rows and df_va.height > args.max_valid_rows:
            df_va = df_va.sample(n=args.max_valid_rows, with_replacement=False, shuffle=True, seed=42)

    if not df_tr.is_empty() and not df_va.is_empty():
        print(f"[trend] TRAIN: {tr_dates[0]} .. {tr_dates[-1]}")
        print(f"[trend] VALID: {va_dates[0]} .. {va_dates[-1]}")

    return df_tr, df_va

# --------------- XGB ---------------
def to_pandas_float32(df: pl.DataFrame, cols: list[str]):
    pdf = df.select(cols).to_pandas()
    for c in pdf.columns:
        if pdf[c].dtype == "float64":
            pdf[c] = pdf[c].astype("float32")
        elif str(pdf[c].dtype).startswith("int"):
            pdf[c] = pdf[c].astype("float32")
    return pdf

def fit_xgb(df_tr: pl.DataFrame, device: str):
    if xgb is None or df_tr.is_empty():
        return None, []

    drop = {"marketId","selectionId","publishTimeMs","marketStartMs","countryCode","ltp_future","dp_target"}
    feats = [c for c in df_tr.columns if c not in drop]

    X32 = to_pandas_float32(df_tr, feats)
    y32 = df_tr["dp_target"].to_pandas().astype("float32")

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
    booster = None
    if device == "cuda":
        params["device"] = "cuda"
        # QuantileDMatrix -> memory efficient for GPU
        try:
            dtrain = xgb.QuantileDMatrix(X32, label=y32)
            booster = xgb.train(params, dtrain, num_boost_round=300)
        except Exception:
            # fallback
            dtrain = xgb.DMatrix(X32, label=y32)
            booster = xgb.train(params, dtrain, num_boost_round=300)
    else:
        dtrain = xgb.DMatrix(X32, label=y32)
        booster = xgb.train(params, dtrain, num_boost_round=250)

    return booster, feats

def eval_xgb(booster, feats: list[str], df_va: pl.DataFrame):
    if booster is None or df_va.is_empty():
        return 0.0, 0.0
    take = [c for c in feats if c in df_va.columns]
    Xv32 = to_pandas_float32(df_va, take)
    try:
        dv = xgb.QuantileDMatrix(Xv32) if booster.attributes().get("device","") == "cuda" else xgb.DMatrix(Xv32)
    except Exception:
        dv = xgb.DMatrix(Xv32)
    preds = booster.predict(dv)
    if len(preds) == 0:
        return 0.0, 0.0
    return float(preds.mean()), float((preds > 0).mean())

# --------------- main ---------------
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

    df_tr, df_va = prepare_sets(args)

    n_tr = df_tr.height if not df_tr.is_empty() else 0
    n_va = df_va.height if not df_va.is_empty() else 0
    print(f"[trend] rows train={n_tr:,d}  valid={n_va:,d}")

    booster, feats = fit_xgb(df_tr, args.device)
    if booster is not None:
        ev_mean, p_pos = eval_xgb(booster, feats, df_va)
        print(f"[trend] valid EV per £1: mean={ev_mean:.5f}  p>0 share={p_pos:.3f}")

    outdir = Path(args.output_dir) / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    mp = outdir / "xgb_trend_reg.json"
    if booster is not None:
        booster.save_model(str(mp))
        print(f"[trend] saved model → {mp}")
    else:
        mp.write_text('{"note":"no-xgb; proxy-only"}')
        print(f"[trend] WARNING: xgboost unavailable; wrote sentinel model to {mp}")

if __name__ == "__main__":
    main()
