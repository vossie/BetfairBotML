#!/usr/bin/env python3
# Calibration diagnostics: reliability curve, deciles, Brier, AUC/logloss.
# Polars 1.33 compatible. Matches the train_edge_temporal.py feature pipeline.

import os, sys, glob
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- helpers ----------

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def die(msg, code=2):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def compute_windows(start_date_str: str, asof_str: str, valid_days: int):
    start_date = parse_date(start_date_str)
    asof       = parse_date(asof_str)
    valid_end   = asof
    valid_start = asof - timedelta(days=valid_days - 1)
    train_start = start_date
    train_end   = valid_start - timedelta(days=1)
    return train_start, train_end, valid_start, valid_end

def epoch_date_expr(colname: str) -> pl.Expr:
    # Detect ms vs s; output UTC date (Polars 1.33)
    return (
        pl.when(pl.col(colname) >= 100_000_000_000)
        .then(pl.from_epoch(pl.col(colname), time_unit="ms"))
        .otherwise(pl.from_epoch(pl.col(colname), time_unit="s"))
    ).dt.replace_time_zone("UTC").dt.date()


# ---------- data loading / features (mirror training) ----------

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files = []
    cur = start
    while cur <= end:
        d = cur.strftime("%Y-%m-%d")
        p = root / sub / f"date={d}"
        files.extend(glob.glob(str(p / "*.parquet")))
        cur += timedelta(days=1)
    return files

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str):
    files = list_parquet_between(curated, f"orderbook_snapshots_5s/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","ltpTick"]
    names = lf.collect_schema().names()
    cols = [c for c in keep if c in names]
    return lf.select(cols).collect()

def load_results(curated: Path, start: datetime, end: datetime, sport: str):
    files = list_parquet_between(curated, f"results/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport","marketId","selectionId","winLabel"]
    names = lf.collect_schema().names()
    cols = [c for c in keep if c in names]
    return lf.select(cols).collect()

def load_defs(curated: Path, start: datetime, end: datetime, sport: str):
    files = list_parquet_between(curated, f"market_definitions/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files)
    names = lf.collect_schema().names()
    if "runners" in names:
        lf = lf.explode("runners").select([
            "sport","marketId",
            pl.col("runners").struct.field("selectionId").alias("selectionId"),
            pl.col("marketStartMs"),
            pl.col("marketType").alias("marketType_def"),
            pl.col("countryCode"),
            pl.col("runners").struct.field("handicap"),
            pl.col("runners").struct.field("sortPriority"),
            pl.col("runners").struct.field("reductionFactor"),
        ])
    else:
        have = [c for c in ["sport","marketId","marketStartMs","marketType","countryCode"] if c in names]
        lf = lf.select(have)
    return lf.collect()

def join_all(snap_df: pl.DataFrame, res_df: pl.DataFrame, defs_df: pl.DataFrame):
    if snap_df.is_empty() or res_df.is_empty(): return pl.DataFrame()
    df = snap_df.join(
        res_df.select(["sport","marketId","selectionId","winLabel"]),
        on=["sport","marketId","selectionId"], how="inner"
    )
    if not defs_df.is_empty():
        on_cols = [c for c in ["sport","marketId","selectionId"] if c in defs_df.columns and c in df.columns]
        df = df.join(defs_df, on=on_cols, how="left")
    return df

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    for col in ["marketType_def","countryCode"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null("UNK"))
            df = df.to_dummies(columns=[col])
    return df

def add_preoff_filter(df: pl.DataFrame, preoff_mins: int) -> pl.DataFrame:
    if "marketStartMs" not in df.columns:
        die("marketStartMs missing after join (market_definitions not found/loaded)")
    df = df.filter(pl.col("marketStartMs").is_not_null())
    df = df.with_columns([
        (pl.col("marketStartMs") - pl.col("publishTimeMs")).alias("secs_to_start"),
        ((pl.col("marketStartMs") - pl.col("publishTimeMs"))/60000).alias("mins_to_start"),
    ])
    return df.filter((pl.col("mins_to_start") >= 0) & (pl.col("mins_to_start") <= preoff_mins))

def apply_pm_gate(df: pl.DataFrame, pm_cutoff: float) -> pl.DataFrame:
    if "pm_label" in df.columns:
        df = df.filter(pl.col("pm_label").is_not_null())
        df = df.filter(pl.col("pm_label") >= pm_cutoff)
    return df

def numeric_only(df: pl.DataFrame, exclude: set) -> pl.DataFrame:
    df = df.drop([c for c in exclude if c in df.columns])
    drop_ls = [c for c, dt in zip(df.columns, df.dtypes) if isinstance(dt, (pl.List, pl.Struct))]
    if drop_ls: df = df.drop(drop_ls)
    keep = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric() or dt == pl.Boolean]
    return df.select(keep)


# ---------- model ----------

def make_params(device="cuda"):
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc"],
        "tree_method": "hist",
        "device": "cuda" if device != "cpu" else "cpu",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

def train(params, dtrain, dvalid):
    return xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain,"train"),(dvalid,"valid")],
        early_stopping_rounds=30,
        verbose_eval=False
    )


# ---------- calibration ----------

def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    return float(np.mean((y_pred - y_true)**2))

def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 20) -> pl.DataFrame:
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    bins = np.linspace(0.0, 1.0, n_bins+1, dtype=np.float64)
    idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins-1)
    recs = []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            recs.append(dict(bin=b, left=bins[b], right=bins[b+1], n=0, mean_pred=np.nan, win_rate=np.nan, gap=np.nan))
            continue
        yp, yt = y_pred[m], y_true[m]
        mean_pred = float(np.mean(yp))
        win_rate  = float(np.mean(yt))
        gap = float(win_rate - mean_pred)
        recs.append(dict(bin=b, left=bins[b], right=bins[b+1], n=int(m.sum()),
                         mean_pred=mean_pred, win_rate=win_rate, gap=gap))
    return pl.DataFrame(recs)


# ---------- CLI ----------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Calibration diagnostics for Edge Temporal model")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--device", default="cuda")
    # env-like backtest filters to mirror training split (only pre-off matters here)
    p.add_argument("--preoff-mins", type=int, default=int(os.environ.get("PREOFF_MINS","30")))
    p.add_argument("--pm-cutoff", type=float, default=float(os.environ.get("PM_CUTOFF","0.65")))
    return p.parse_args()


# ---------- main ----------

def main():
    args = parse_args()
    outdir = Path(os.environ.get("OUTPUT_DIR","/opt/BetfairBotML/edge_temporal/output"))
    outdir.mkdir(parents=True, exist_ok=True)

    train_start, train_end, valid_start, valid_end = compute_windows(args.start_date, args.asof, args.valid_days)
    print(f"[calib] Training window:   {train_start.date()} .. {train_end.date()}")
    print(f"[calib] Validation window: {valid_start.date()} .. {valid_end.date()}")

    curated = Path(args.curated)
    snap_df = load_snapshots(curated, train_start, valid_end, args.sport)
    res_df  = load_results(curated,  train_start, valid_end, args.sport)
    defs_df = load_defs(curated,     train_start, valid_end, args.sport)
    if snap_df.is_empty() or res_df.is_empty():
        die("no snapshots or results loaded")

    df_all = join_all(snap_df, res_df, defs_df)
    if df_all.is_empty(): die("empty after join")
    if "winLabel" not in df_all.columns: die("winLabel missing")
    if "ltp" not in df_all.columns: die("ltp missing")

    df_all = df_all.filter(pl.col("winLabel").is_not_null()).with_columns(
        pl.when(pl.col("winLabel")>0).then(1).otherwise(0).alias("winLabel")
    ).filter(pl.col("ltp").is_not_null())

    # pre-off filter + optional PM gate to mirror training/backtest
    df_all = add_preoff_filter(df_all, args.preoff_mins)
    df_all = apply_pm_gate(df_all, args.pm_cutoff)
    df_all = encode_categoricals(df_all)

    # split
    df_train = df_all.filter((pl.col("publishTimeMs") >= to_ms(train_start)) & (pl.col("publishTimeMs") < to_ms(train_end + timedelta(days=1))))
    df_valid = df_all.filter((pl.col("publishTimeMs") >= to_ms(valid_start)) & (pl.col("publishTimeMs") < to_ms(valid_end + timedelta(days=1))))
    if df_train.is_empty() or df_valid.is_empty(): die("empty train/valid")

    exclude = {"winLabel","sport","marketId","selectionId","marketStartMs","secs_to_start"}
    X_train = numeric_only(df_train, exclude)
    X_valid = numeric_only(df_valid, exclude)
    y_train = df_train["winLabel"].to_numpy().astype(np.float32)
    y_valid = df_valid["winLabel"].to_numpy().astype(np.float32)

    print(f"[calib] rows: train={df_train.height:,}  valid={df_valid.height:,}  features={X_train.width}")

    # train & predict
    params = make_params(args.device)
    dtrain = xgb.DMatrix(X_train.to_arrow(), label=y_train)
    dvalid = xgb.DMatrix(X_valid.to_arrow(), label=y_valid)
    booster = train(params, dtrain, dvalid)

    p_valid = booster.predict(dvalid).astype(np.float32)
    # metrics
    auc = roc_auc_score(y_valid, p_valid)
    ll  = log_loss(y_valid, np.clip(p_valid, 1e-15, 1-1e-15))
    br  = brier_score(y_valid, p_valid)
    print(f"[calib] AUC={auc:.3f}  logloss={ll:.4f}  Brier={br:.5f}")

    # table
    table = calibration_table(y_valid, p_valid, n_bins=20)
    tpath = outdir / f"calib_table_{args.asof}.csv"
    table.write_csv(str(tpath))
    print(f"[calib] table saved → {tpath}")

    # plot
    fig = plt.figure(figsize=(8,6))
    ax1 = plt.gca()
    # reliability curve
    mp = table["mean_pred"].fill_null(strategy="zero").to_numpy()
    wr = table["win_rate"].fill_null(strategy="zero").to_numpy()
    ax1.plot([0,1],[0,1], linestyle="--")
    ax1.plot(mp, wr, marker="o")
    ax1.set_xlabel("Mean predicted probability (bin)")
    ax1.set_ylabel("Empirical win rate (bin)")
    ax1.set_title(f"Reliability Curve (valid up to {args.asof})")

    # twin histogram of predictions
    ax2 = ax1.twinx()
    ax2.hist(np.clip(p_valid,0,1), bins=30, alpha=0.3)
    ax2.set_ylabel("Count (predictions)")

    ppath = outdir / f"calib_plot_{args.asof}.png"
    fig.tight_layout()
    fig.savefig(ppath, dpi=150)
    print(f"[calib] plot saved → {ppath}")

if __name__ == "__main__":
    main()
