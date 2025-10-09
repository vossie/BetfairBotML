#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

import polars as pl
import numpy as np

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ---------------- args ----------------
def parse_args():
    p = argparse.ArgumentParser(
        prog="train_price_trend (inf-safe, memory-safe)",
        description="Train XGBoost trend regressor on curated Betfair snapshots."
    )
    p.add_argument("--curated", required=True, help="Curated root, e.g. /mnt/nvme/betfair-curated")
    p.add_argument("--asof", required=True, help="AS OF date (YYYY-MM-DD)")
    p.add_argument("--start-date", required=True, help="Training start date (YYYY-MM-DD)")
    p.add_argument("--valid-days", type=int, default=7, help="Validation days ending at ASOF (inclusive)")
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30, help="minutes pre-off")
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "output"))

    # existing options you already had
    p.add_argument("--country-facet", action="store_true")
    p.add_argument("--downsample-secs", type=int, default=0)
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--max-train-rows", type=int, default=0, help="cap train rows (0=off)")
    p.add_argument("--max-valid-rows", type=int, default=250_000, help="cap valid rows")

    # NEW/Preferred flags (already present)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-eta", type=float, default=0.08)  # learning_rate
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    p.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    p.add_argument("--xgb-num-boost-round", type=int, default=300)
    p.add_argument("--xgb-early-stopping-rounds", type=int, default=0)

    # --- LEGACY ALIASES (accept both; map below) ---
    p.add_argument("--xgb-learning-rate", type=float, default=None)  # alias for --xgb-eta
    p.add_argument("--xgb-n-estimators", type=int, default=None)  # alias for --xgb-num-boost-round

    return p.parse_args()

# ---------------- schema helpers ----------------
def load_schema(kind: str) -> list[str]:
    # Try project-provided loader if present
    try:
        from ml.schema_loader import load_schema as _load
        return _load(kind)
    except Exception:
        if kind == "orderbook":
            return ["sport","marketId","selectionId","publishTimeMs",
                    "ltp","ltpTick","tradedVolume","spreadTicks","imbalanceBest1",
                    "backTicks","backSizes","layTicks","laySizes"]
        if kind == "marketdef":
            return ["sport","marketId","marketStartMs","countryCode"]
        return []

def intersect_existing(requested: list[str], have: list[str]) -> list[str]:
    hs = set(have)
    return [c for c in requested if c in hs]

# ---------------- dates/paths ----------------
def daterange(start: str, end: str) -> list[str]:
    sd = datetime.strptime(start, "%Y-%m-%d").date()
    ed = datetime.strptime(end, "%Y-%m-%d").date()
    return [(sd + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((ed - sd).days + 1)]

def curated_dirs(root: str, sport: str, date: str) -> Tuple[Path,Path]:
    base = Path(root)
    ob = base / sport / date / "orderbook"
    md = base / sport / date / "market-definitions"
    return ob, md

def read_day(root: str, sport: str, date: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    ob_dir, md_dir = curated_dirs(root, sport, date)
    ob_files = sorted(glob.glob(str(ob_dir / "*.parquet")))
    md_files = sorted(glob.glob(str(md_dir / "*.parquet")))
    if not ob_files or not md_files:
        return pl.DataFrame({}), pl.DataFrame({})

    # scan lazily to be memory-safe
    ob_cols = load_schema("orderbook")
    ob = pl.scan_parquet(ob_files).select([c for c in ob_cols if c in pl.scan_parquet(ob_files).columns])
    md_cols = load_schema("marketdef")
    md = pl.scan_parquet(md_files).select([c for c in md_cols if c in pl.scan_parquet(md_files).columns])

    # materialize now; engine='streaming' keeps memory frugal
    return ob.collect(streaming=True), md.collect(streaming=True)

# ---------------- features ----------------
def _safe_ret(now: pl.Expr, past: pl.Expr) -> pl.Expr:
    return (now - past) / pl.when(past.abs() < 1e-12).then(1e-12).otherwise(past)

def build_features(snaps: pl.DataFrame, defs: pl.DataFrame,
                   preoff_max_mins: int, horizon_secs: int,
                   include_country: bool, downsample_secs: int) -> pl.DataFrame:
    if snaps.is_empty() or defs.is_empty():
        return pl.DataFrame({})

    # join defs (marketStartMs) to get minutes to off
    want_defs = intersect_existing(["marketId","marketStartMs","countryCode","sport"], defs.columns)
    defs_s = defs.select(want_defs).unique(subset=["marketId"])
    out = snaps.join(defs_s, on="marketId", how="left")

    # pre-off window
    out = out.with_columns([
        ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off")
    ]).filter(pl.col("mins_to_off").is_not_null() & (pl.col("mins_to_off") >= 0) & (pl.col("mins_to_off") <= preoff_max_mins))

    # light sanitization + derived floats
    out = out.with_columns([
        pl.col("ltp").cast(pl.Float64).alias("ltp_f"),
        pl.col("tradedVolume").cast(pl.Float64).alias("vol_f"),
    ]).sort(["marketId","selectionId","publishTimeMs"])

    # Optional downsample to reduce rows (keep every N seconds)
    if downsample_secs and downsample_secs > 0:
        step = max(1, int(round(downsample_secs / 5)))  # base grid is 5s
        out = out.with_columns(((pl.col("publishTimeMs") // 5000) % step).alias("__mod"))
        out = out.filter(pl.col("__mod") == 0).drop("__mod")

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
        _safe_ret(pl.col("ltp_f"), pl.col("ltp_lag30s")).alias("ltp_ret_30s"),
        _safe_ret(pl.col("ltp_f"), pl.col("ltp_lag60s")).alias("ltp_ret_60s"),
        _safe_ret(pl.col("ltp_f"), pl.col("ltp_lag120s")).alias("ltp_ret_120s"),
        (pl.col("publishTimeMs") + horizon_secs * 1000).alias("ts_exit_ms"),
    ])

    if include_country and "countryCode" not in out.columns:
        out = out.with_columns(pl.lit("UNK").alias("countryCode"))

    # future ltp (forward fill)
    out = out.with_columns([
        pl.when(pl.col("ltp_f").is_not_null()).then(pl.col("ltp_f")).otherwise(None).alias("__ltp_tmp")
    ])
    out = out.with_columns([
        pl.col("__ltp_tmp").shift(-max(1, int(round(horizon_secs / 5.0)))).over(grp).alias("ltp_future")
    ]).drop("__ltp_tmp")

    out = out.with_columns([(pl.col("ltp_future") - pl.col("ltp_f")).alias("dp_target")])
    out = out.filter(pl.col("dp_target").is_not_null())

    # Select columns used by model
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
    return out.select(present)

def prepare_sets(args):
    asof_d = datetime.strptime(args.asof, "%Y-%m-%d").date()
    vstart = asof_d - timedelta(days=args.valid_days)
    train_end = vstart - timedelta(days=1)

    tr_dates = daterange(args.start_date, train_end.strftime("%Y-%m-%d"))
    va_dates = daterange(vstart.strftime("%Y-%m-%d"), args.asof)

    def collect_dates(dates: List[str], label: str) -> pl.DataFrame:
        parts: List[pl.DataFrame] = []
        missing = []
        for d in dates:
            snaps, defs = read_day(args.curated, args.sport, d)
            if snaps.is_empty():
                missing.append(d)
                continue
            feat = build_features(snaps, defs, args.preoff_max, args.horizon_secs,
                                  args.country_facet, args.downsample_secs)
            if not feat.is_empty():
                parts.append(feat)
        if missing:
            print(f"[{label}] skipped {len(missing)} empty day(s): {', '.join(missing[:6])}{' …' if len(missing)>6 else ''}")
        if not parts:
            return pl.DataFrame({})
        try:
            df = pl.concat(parts, how="vertical_relaxed").collect(engine="streaming")
        except Exception:
            df = pl.concat(parts, how="vertical_relaxed")
        # sample/cap
        if label == "TRAIN":
            if args.sample_frac < 1.0:
                df = df.sample(frac=args.sample_frac, with_replacement=False, shuffle=True, seed=42)
            if args.max_train_rows and df.height > args.max_train_rows:
                df = df.sample(n=args.max_train_rows, with_replacement=False, shuffle=True, seed=42)
        else:
            if args.max_valid_rows and df.height > args.max_valid_rows:
                df = df.sample(n=args.max_valid_rows, with_replacement=False, shuffle=True, seed=42)
        return df

    df_tr = collect_dates(tr_dates, "TRAIN")
    df_va = collect_dates(va_dates, "VALID")

    # ensure finite targets
    if not df_tr.is_empty():
        df_tr = df_tr.with_columns(pl.when(pl.col("dp_target").is_finite()).then(pl.col("dp_target")).otherwise(None).alias("dp_target")).drop_nulls(["dp_target"])
    if not df_va.is_empty():
        df_va = df_va.with_columns(pl.when(pl.col("dp_target").is_finite()).then(pl.col("dp_target")).otherwise(None).alias("dp_target")).drop_nulls(["dp_target"])
    return df_tr, df_va

# ---------------- numpy helper ----------------
def to_float32_pandas(df: pl.DataFrame, cols: List[str]):
    # NOTE: older Polars doesn't accept copy= kw
    pdf = df.select(cols).to_pandas()
    # Replace ±inf with NaN; XGBoost will treat NaN as missing
    for c in pdf.columns:
        col = pdf[c].to_numpy(copy=False)
        # Convert to float32
        if np.issubdtype(col.dtype, np.floating):
            pdf[c] = pdf[c].astype("float32", copy=False)
            col = pdf[c].to_numpy(copy=False)
        else:
            pdf[c] = pdf[c].astype("float32")
            col = pdf[c].to_numpy(copy=False)
        # sanitize infs
        m_inf = ~np.isfinite(col)
        if m_inf.any():
            col[m_inf] = np.nan
    return pdf

# ---------------- XGB fit/eval ----------------
def fit_xgb(df_tr: pl.DataFrame, df_va: pl.DataFrame, device: str, args):
    if xgb is None or df_tr.is_empty():
        return None, []

    drop = {"marketId","selectionId","publishTimeMs","marketStartMs","countryCode","ltp_future","dp_target"}
    feats = [c for c in df_tr.columns if c not in drop]
    X32 = to_float32_pandas(df_tr, feats)
    y32 = df_tr["dp_target"].to_pandas().astype("float32")
    y_arr = y32.to_numpy()
    y_arr[~np.isfinite(y_arr)] = np.nan

    params = {
        "max_depth": int(args.xgb_max_depth),
        "eta": float(args.xgb_eta),
        "subsample": float(args.xgb_subsample),
        "colsample_bytree": float(args.xgb_colsample_bytree),
        "min_child_weight": float(args.xgb_min_child_weight),
        "lambda": float(args.xgb_reg_lambda),
        "alpha": float(args.xgb_reg_alpha),
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }

    booster = None
    dtrain = None
    dvalid = None

    # Build DMatrices
    if device == "cuda":
        params["device"] = "cuda"
        try:
            dtrain = xgb.QuantileDMatrix(X32, label=y_arr, missing=np.nan)
        except Exception:
            dtrain = xgb.DMatrix(X32, label=y_arr, missing=np.nan)
    else:
        dtrain = xgb.DMatrix(X32, label=y_arr, missing=np.nan)

    evals = None
    if not df_va.is_empty():
        Xv32 = to_float32_pandas(df_va, feats)
        yv = df_va["dp_target"].to_pandas().astype("float32").to_numpy()
        yv[~np.isfinite(yv)] = np.nan
        try:
            dvalid = xgb.QuantileDMatrix(Xv32, label=yv, missing=np.nan) if device=="cuda" else xgb.DMatrix(Xv32, label=yv, missing=np.nan)
        except Exception:
            dvalid = xgb.DMatrix(Xv32, label=yv, missing=np.nan)
        evals = [(dtrain, "train"), (dvalid, "valid")]

    num_round = int(args.xgb_num_boost_round)
    early_stop = int(args.xgb_early_stopping_rounds)
    if early_stop > 0 and evals is not None:
        booster = xgb.train(params, dtrain, num_boost_round=num_round, evals=evals,
                            early_stopping_rounds=early_stop, verbose_eval=False)
    else:
        booster = xgb.train(params, dtrain, num_boost_round=num_round)

    return booster, feats

def eval_xgb(booster, feats: list[str], df_va: pl.DataFrame):
    if booster is None or df_va.is_empty():
        return 0.0, 0.0
    take = [c for c in feats if c in df_va.columns]
    Xv32 = to_float32_pandas(df_va, take)
    try:
        dv = xgb.QuantileDMatrix(Xv32, missing=np.nan)
    except Exception:
        dv = xgb.DMatrix(Xv32, missing=np.nan)
    preds = booster.predict(dv)
    if len(preds) == 0:
        return 0.0, 0.0
    # mirror your earlier logging: mean predicted dp, and share > 0
    return float(np.nanmean(preds)), float(np.mean(preds > 0))

# ---------------- main ----------------
def main():
    args = parse_args()

    # Map legacy flags to new names if provided
    if getattr(args, "xgb_learning_rate", None) is not None:
        args.xgb_eta = args.xgb_learning_rate
    if getattr(args, "xgb_n_estimators", None) is not None:
        args.xgb_num_boost_round = args.xgb_n_estimators


    print("=== Price Trend Training ===")
    print(f"Curated root:    {args.curated}")
    print(f"ASOF:            {args.asof}")
    print(f"Start date:      {args.start_date}")
    print(f"Valid days:      {args.valid_days}")
    print(f"Horizon (secs):  {args.horizon_secs}")
    print(f"Pre-off max (m): {args.preoff_max}")
    print(f"XGBoost device:  {args.device}")
    print(f"EV mode:         mtm")

    # windows
    asof_d = datetime.strptime(args.asof, "%Y-%m-%d").date()
    vstart = asof_d - timedelta(days=args.valid_days)
    train_start = args.start_date
    train_end = (vstart - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[trend] TRAIN: {train_start} .. {train_end}")
    print(f"[trend] VALID: {vstart.strftime('%Y-%m-%d')} .. {args.asof}")

    df_tr, df_va = prepare_sets(args)

    n_tr = df_tr.height if not df_tr.is_empty() else 0
    n_va = df_va.height if not df_va.is_empty() else 0
    print(f"[trend] rows train={n_tr:,d}  valid={n_va:,d}")

    booster, feats = fit_xgb(df_tr, df_va, args.device, args)
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
