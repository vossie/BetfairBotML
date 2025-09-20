#!/usr/bin/env python3
import os, sys, glob, itertools
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

# ---------------------------
# Utilities
# ---------------------------

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def safe_logloss(y_true, y_pred):
    try:
        return log_loss(y_true, y_pred, eps=1e-15)
    except ValueError:
        return float("nan")

def compute_windows(start_date_str: str, asof_str: str, valid_days: int):
    """Compute train/valid split windows."""
    start_date = parse_date(start_date_str)
    asof = parse_date(asof_str)

    valid_end = asof
    valid_start = asof - timedelta(days=valid_days - 1)

    train_start = start_date
    train_end = valid_start - timedelta(days=1)

    return train_start, train_end, valid_start, valid_end

# ---------------------------
# Data loading
# ---------------------------

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str):
    files = []
    cur = start
    while cur <= end:
        day = cur.strftime("%Y-%m-%d")
        path = curated / "orderbook_snapshots_5s" / f"sport={sport}" / f"date={day}"
        files.extend(glob.glob(str(path / "*.parquet")))
        cur += timedelta(days=1)
    if not files:
        return pl.DataFrame()
    return pl.read_parquet(files)

def load_results(curated: Path, start: datetime, end: datetime, sport: str):
    files = []
    cur = start
    while cur <= end:
        day = cur.strftime("%Y-%m-%d")
        path = curated / "results" / f"sport={sport}" / f"date={day}"
        files.extend(glob.glob(str(path / "*.parquet")))
        cur += timedelta(days=1)
    if not files:
        return pl.DataFrame()
    return pl.read_parquet(files)

def join_snapshots_results(snap_df: pl.DataFrame, res_df: pl.DataFrame) -> pl.DataFrame:
    if snap_df.is_empty() or res_df.is_empty():
        return pl.DataFrame()
    return snap_df.join(
        res_df.select(["sport","marketId","selectionId","winLabel"]),
        on=["sport","marketId","selectionId"],
        how="inner"
    )

# ---------------------------
# Training / Evaluation
# ---------------------------

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
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain,"train"),(dvalid,"valid")],
        early_stopping_rounds=20,
        verbose_eval=50
    )
    return booster

def evaluate(df_valid, odds, y, p_model, p_market, edge_thresh, topk, lo, hi,
             stake_mode, cap, floor_, bank, commission):
    mask = (odds >= lo) & (odds <= hi)
    edges = p_model - p_market
    sel = (edges >= edge_thresh) & mask
    n_trades = sel.sum()
    if n_trades == 0:
        return dict(roi=0.0, profit=0.0, n_trades=0)

    if stake_mode == "flat":
        stake = np.ones_like(y[sel])
    else:  # kelly
        kelly_fraction = (odds[sel] * p_model[sel] - 1) / (odds[sel] - 1)
        kelly_fraction = np.clip(kelly_fraction, floor_, cap)
        stake = bank * kelly_fraction

    pnl = (y[sel] * (odds[sel]-1) - (1-y[sel])) * stake
    pnl *= (1.0 - commission)
    profit = pnl.sum()
    roi = profit / (stake.sum() + 1e-12)
    return dict(roi=roi, profit=profit, n_trades=int(n_trades))

# ---------------------------
# Main
# ---------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=3)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--device", default="cuda")
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--bankroll-nom", type=float, default=1000.0)
    p.add_argument("--kelly-cap", type=float, default=0.05)
    p.add_argument("--kelly-floor", type=float, default=0.0)
    args = p.parse_args()

    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR","/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_start, train_end, valid_start, valid_end = compute_windows(
        args.start_date, args.asof, args.valid_days
    )

    print("=== Edge Temporal Training (LOCAL) ===")
    print(f"Curated root:        {args.curated}")
    print(f"ASOF:                {args.asof}")
    print(f"Training window:     {train_start.date()} .. {train_end.date()}")
    print(f"Validation window:   {valid_start.date()} .. {valid_end.date()}")

    snap_df = load_snapshots(Path(args.curated), train_start, valid_end, args.sport)
    res_df  = load_results(Path(args.curated), train_start, valid_end, args.sport)
    df_all  = join_snapshots_results(snap_df, res_df)

    if df_all.is_empty():
        print("[ERROR] empty train/valid")
        sys.exit(1)

    # split
    df_train = df_all.filter(
        (pl.col("publishTimeMs") >= to_ms(train_start)) &
        (pl.col("publishTimeMs") <= to_ms(train_end+timedelta(days=1)))
    )
    df_valid = df_all.filter(
        (pl.col("publishTimeMs") >= to_ms(valid_start)) &
        (pl.col("publishTimeMs") <= to_ms(valid_end+timedelta(days=1)))
    )

    feats = [c for c in df_all.columns if c not in ("winLabel","sport","marketId","selectionId")]
    dtrain = xgb.DMatrix(df_train.select(feats).to_arrow(),
                         label=df_train["winLabel"].to_numpy())
    dvalid = xgb.DMatrix(df_valid.select(feats).to_arrow(),
                         label=df_valid["winLabel"].to_numpy())

    booster = train(make_params(args.device), dtrain, dvalid, must_gpu=(args.device=="cuda"))
    preds = booster.predict(dvalid)

    y = df_valid["winLabel"].to_numpy()
    odds = df_valid["ltp"].fill_null(1.0).to_numpy()
    p_model = preds.astype(np.float32)
    p_market = (1.0/np.clip(odds,1e-12,None)).astype(np.float32)

    print(f"[Value] logloss={safe_logloss(y,preds):.4f} auc={roc_auc_score(y,preds):.3f}")

    EDGE_T=[0.010,0.012,0.015]; TOPK=[1,2]; LTP_W=[(1.5,5.0)]
    STAKE=[("flat",None,None,args.bankroll_nom),
           ("kelly",args.kelly_cap,args.kelly_floor,args.bankroll_nom)]
    recs=[]
    for e,t,(lo,hi),(sm,cap,floor_,bank) in itertools.product(EDGE_T,TOPK,LTP_W,STAKE):
        m=evaluate(df_valid,odds,y,p_model,p_market,e,t,lo,hi,sm,cap or 0.0,floor_ or 0.0,bank or 1000.0,args.commission)
        m.update(dict(edge_thresh=e,topk=t,ltp_min=lo,ltp_max=hi,stake_mode=sm))
        recs.append(m)

    sweep=pl.DataFrame(recs).sort(["roi","n_trades"],descending=[True,True])
    out=OUTPUT_DIR/f"edge_sweep_{args.asof}.csv"
    sweep.write_csv(str(out))
    print(f"sweep saved → {out}")
    print(sweep.select(["roi","profit","n_trades","edge_thresh","stake_mode"]).head(10))

if __name__=="__main__":
    main()
