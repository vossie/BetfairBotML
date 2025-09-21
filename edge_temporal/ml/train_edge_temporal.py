#!/usr/bin/env python3
import os, sys, glob, itertools
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

# --------------------------- utils ---------------------------

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def safe_logloss(y_true, y_pred):
    try:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_pred)
    except ValueError:
        return float("nan")


def die(msg, code=2):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def compute_windows(start_date_str: str, asof_str: str, valid_days: int):
    start_date = parse_date(start_date_str)             # fixed anchor (e.g., 2025-09-05)
    asof       = parse_date(asof_str)                   # yesterday from .sh
    valid_end   = asof
    valid_start = asof - timedelta(days=valid_days - 1)
    train_start = start_date
    train_end   = valid_start - timedelta(days=1)
    return train_start, train_end, valid_start, valid_end

# --------------------------- data ---------------------------

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files=[]
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
    # Select only safe, scalar columns (no lists)
    keep = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","ltpTick"]
    cols = [c for c in keep if c in lf.collect_schema().names()]
    return lf.select(cols).collect()

def load_results(curated: Path, start: datetime, end: datetime, sport: str):
    files = list_parquet_between(curated, f"results/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport","marketId","selectionId","winLabel","settledTimeMs","eventId","runnerStatus","marketType"]
    cols = [c for c in keep if c in lf.collect_schema().names()]
    return lf.select(cols).collect()

def load_defs(curated: Path, start: datetime, end: datetime, sport: str):
    files = list_parquet_between(curated, f"market_definitions/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files)
    if "runners" in lf.collect_schema().names():
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
        # minimal if no runners struct
        have = [c for c in ["sport","marketId","marketStartMs","marketType","countryCode"] if c in lf.collect_schema().names()]
        lf = lf.select(have)
    return lf.collect()

def join_all(snap_df: pl.DataFrame, res_df: pl.DataFrame, defs_df: pl.DataFrame):
    if snap_df.is_empty() or res_df.is_empty():
        return pl.DataFrame()
    df = snap_df.join(
        res_df.select([c for c in ["sport","marketId","selectionId","winLabel","marketType"] if c in res_df.columns]),
        on=["sport","marketId","selectionId"], how="inner"
    )
    if not defs_df.is_empty():
        on_cols = [c for c in ["sport","marketId","selectionId"] if c in defs_df.columns and c in df.columns]
        df = df.join(defs_df, on=on_cols, how="left")
    return df

# ---------------------- feature/prepare ----------------------

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    for col in ["marketType","marketType_def","countryCode","runnerStatus"]:
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

def numeric_only(df: pl.DataFrame, exclude: set) -> pl.DataFrame:
    # drop excluded
    df = df.drop([c for c in exclude if c in df.columns])
    # drop List/Struct columns if any slipped in
    drop_ls = [c for c, dt in zip(df.columns, df.dtypes) if isinstance(dt, (pl.List, pl.Struct))]
    if drop_ls:
        df = df.drop(drop_ls)
    # keep only numeric/bool
    keep = [c for c, dt in zip(df.columns, df.dtypes)
            if (dt.is_numeric() or isinstance(dt, pl.Boolean))]
    return df.select(keep)

# --------------------------- model ---------------------------

def make_params(device="cuda"):
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc"],
        "tree_method": "hist" if device=="cpu" else "gpu_hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

def train(params, dtrain, dvalid, must_gpu=False):
    return xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain,"train"),(dvalid,"valid")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

def evaluate(df_valid, odds, y, p_model, p_market, edge_thresh, topk, lo, hi,
             stake_mode, cap, floor_, bank, commission):
    edge = p_model - p_market
    mask = (odds >= lo) & (odds <= hi) & np.isfinite(edge)
    if not mask.any(): return dict(roi=0.0, profit=0.0, n_trades=0)
    df = pl.DataFrame({
        "marketId": df_valid["marketId"].to_numpy()[mask],
        "selectionId": df_valid["selectionId"].to_numpy()[mask],
        "ltp": odds[mask],
        "edge": edge[mask],
        "y": y[mask],
        "p_model": p_model[mask],
    }).filter(pl.col("edge") >= edge_thresh)
    if df.height == 0: return dict(roi=0.0, profit=0.0, n_trades=0)
    df = df.with_columns(pl.rank("dense", descending=True).over("marketId").alias("rk")).filter(pl.col("rk")<=topk).drop("rk")
    outcomes = df["y"].to_numpy().astype(np.float32)
    odds_sel = df["ltp"].to_numpy().astype(np.float32)
    p_model_sel = df["p_model"].to_numpy().astype(np.float32)
    if stake_mode == "kelly":
        def kf(p,o):
            b=o-1.0
            if b<=0: return 0.0
            q=1.0-p
            return max(0.0,(b*p-q)/b)
        f = np.array([max(floor_,min(cap,kf(pi,oi))) for pi,oi in zip(p_model_sel,odds_sel)],dtype=np.float32)
        stakes = f * bank
    else:
        stakes = np.full_like(odds_sel, 10.0, dtype=np.float32)
    gross = outcomes*(odds_sel-1.0)*stakes
    net   = gross*(1.0-commission)
    loss  = (1.0-outcomes)*stakes
    profit = net - loss
    pnl = float(profit.sum())
    staked = float(stakes.sum())
    roi = pnl/staked if staked>0 else 0.0
    return dict(roi=roi, profit=pnl, n_trades=int(outcomes.size))

# ---------------------------- main ----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--valid-days", type=int, default=3)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--bankroll-nom", type=float, default=1000.0)
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.0)
    ap.add_argument("--preoff-mins", type=int, default=30)
    args = ap.parse_args()

    outdir = Path(os.environ.get("OUTPUT_DIR","/opt/BetfairBotML/edge_temporal/output"))
    outdir.mkdir(parents=True, exist_ok=True)

    train_start, train_end, valid_start, valid_end = compute_windows(args.start_date, args.asof, args.valid_days)
    print(f"Training window:   {train_start.date()} .. {train_end.date()}")
    print(f"Validation window: {valid_start.date()} .. {valid_end.date()}")

    curated = Path(args.curated)
    snap_df = load_snapshots(curated, train_start, valid_end, args.sport)
    res_df  = load_results(curated,  train_start, valid_end, args.sport)
    defs_df = load_defs(curated,     train_start, valid_end, args.sport)
    if snap_df.is_empty() or res_df.is_empty():
        die("no snapshots or results loaded")

    df_all = join_all(snap_df, res_df, defs_df)
    if df_all.is_empty():
        die("empty after join")

    # label + odds hygiene
    if "winLabel" not in df_all.columns: die("winLabel missing")
    if "ltp" not in df_all.columns: die("ltp missing")
    df_all = df_all.filter(pl.col("winLabel").is_not_null()).with_columns(
        pl.when(pl.col("winLabel")>0).then(1).otherwise(0).alias("winLabel")
    ).filter(pl.col("ltp").is_not_null())

    # pre-off 0..N minutes filter
    df_all = add_preoff_filter(df_all, args.preoff_mins)

    # encode categoricals
    df_all = encode_categoricals(df_all)

    # time slice by publishTimeMs with [start, end) bounds
    train_end_excl = to_ms(train_end + timedelta(days=1))
    valid_end_excl = to_ms(valid_end + timedelta(days=1))

    df_train = df_all.filter((pl.col("publishTimeMs") >= to_ms(train_start)) & (pl.col("publishTimeMs") < train_end_excl))
    df_valid = df_all.filter((pl.col("publishTimeMs") >= to_ms(valid_start)) & (pl.col("publishTimeMs") < valid_end_excl))

    if df_train.is_empty() or df_valid.is_empty():
        die("empty train/valid")

    # features -> numeric only (drop strings, lists, structs)
    exclude = {"winLabel","sport","marketId","selectionId","marketStartMs"}
    X_train = numeric_only(df_train, exclude)
    X_valid = numeric_only(df_valid, exclude)

    # y
    y_train = df_train["winLabel"].to_numpy().astype(np.float32)
    y_valid = df_valid["winLabel"].to_numpy().astype(np.float32)

    # DMatrix
    dtrain = xgb.DMatrix(X_train.to_arrow(), label=y_train)
    dvalid = xgb.DMatrix(X_valid.to_arrow(), label=y_valid)

    # train
    booster = train(make_params(args.device), dtrain, dvalid, must_gpu=(args.device=="cuda"))

    # predict
    p = booster.predict(dvalid).astype(np.float32)
    odds = df_valid["ltp"].to_numpy().astype(np.float32)
    p_market = (1.0/np.clip(odds,1e-12,None)).astype(np.float32)

    print(f"[Value] logloss={safe_logloss(y_valid,p):.4f}  auc={roc_auc_score(y_valid,p):.3f}")

    # sweep: flat + kelly variants
    EDGE_T = [0.010, 0.012, 0.015, 0.020]
    TOPK   = [1, 2, 3]
    LTP_W  = [(1.5, 5.0), (1.5, 10.0)]
    STAKE  = [("flat", None, None, args.bankroll_nom)]
    for cap, floor_ in [(0.05,0.001),(0.05,0.002),(0.10,0.001)]:
        STAKE.append(("kelly", cap, floor_, args.bankroll_nom))

    recs=[]
    for e,t,(lo,hi),(sm,cap,floor_,bank) in itertools.product(EDGE_T,TOPK,LTP_W,STAKE):
        m = evaluate(df_valid, odds, y_valid, p, p_market, e, t, lo, hi,
                     sm, cap or 0.0, floor_ or 0.0, bank or 1000.0, float(args.commission))
        m.update(dict(edge_thresh=e, topk=t, ltp_min=lo, ltp_max=hi,
                      stake_mode=("flat" if sm=="flat" else f"kelly_cap{cap}_floor{floor_}")))
        recs.append(m)

    sweep = pl.DataFrame(recs).sort(["roi","n_trades"], descending=[True, True])
    out   = outdir / f"edge_sweep_{args.asof}.csv"
    sweep.write_csv(str(out))
    print(f"sweep saved → {out}")
    print(sweep.head(15))

if __name__ == "__main__":
    main()
