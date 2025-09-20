#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge Temporal Trainer (stable, normalized time units)

- Reads snapshots, results, market definitions
- Joins on (sport, marketId, selectionId)
- Normalizes publishTimeMs/marketStartMs to milliseconds
- Filters strictly to 0..preoff-mins before off
- Adds runner features (handicap, sortPriority, reductionFactor)
- Adds secs_to_start/mins_to_start as features
- Cleans labels (0/1 floats only) and drops missing odds
- Encodes categoricals then drops any leftover non-numeric cols before XGBoost
- GPU XGBoost training (CUDA forced via shell script)
- Expanded sweep: flat + multiple Kelly variants; saves ROI/PnL
"""

import argparse, os, sys, itertools
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# ------------------------------- utils --------------------------------

def die(msg, code=2):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def safe_logloss(y_true, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return float(-(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean())

def files_for_range(root: Path, sub: str, start: datetime, end: datetime):
    files=[]
    for i in range((end - start).days + 1):
        d = start + timedelta(days=i)
        dd = d.strftime("%Y-%m-%d")
        ddir = root / sub / f"date={dd}"
        if ddir.exists():
            files.extend(ddir.glob("*.parquet"))
    return files

def normalize_ms_expr(expr: pl.Expr) -> pl.Expr:
    """
    Normalize timestamp-like integers to milliseconds since epoch.
    Heuristic:
      - >1e17  : assume nanoseconds -> //1_000_000
      - >1e14  : assume microseconds -> //1_000
      - <=1e14 : assume already ms (or s); if <1e12 assume seconds -> *1000
    All cast to Int64.
    """
    return (
        pl.when(expr.is_null()).then(None)
        .when(expr > 1_000_000_000_000_000_00).then((expr // 1_000_000).cast(pl.Int64))  # ns -> ms
        .when(expr > 100_000_000_000_000).then((expr // 1_000).cast(pl.Int64))           # µs -> ms
        .when(expr < 1_000_000_000_000).then((expr * 1_000).cast(pl.Int64))              # s -> ms
        .otherwise(expr.cast(pl.Int64))                                                  # assume ms
    )

# ------------------------------ loaders -------------------------------

def load_snapshots(curated, start, end, sport):
    files = files_for_range(curated, f"orderbook_snapshots_5s/sport={sport}", start, end)
    if not files: die("no snapshot files found")
    lf = pl.concat([pl.scan_parquet(f) for f in files])
    keep = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1"]
    lf = lf.select([c for c in keep if c in lf.collect_schema().names()])
    return lf

def load_results(curated, start, end, sport):
    files = files_for_range(curated, f"results/sport={sport}", start, end)
    if not files: die("no results files found")
    lf = pl.concat([pl.scan_parquet(f) for f in files])
    keep = ["sport","marketId","selectionId","winLabel","settledTimeMs","eventId","runnerStatus","marketType"]
    lf = lf.select([c for c in keep if c in lf.collect_schema().names()])
    return lf

def load_defs(curated, start, end, sport):
    files = files_for_range(curated, f"market_definitions/sport={sport}", start, end)
    if not files: return None
    lf = pl.concat([pl.scan_parquet(f) for f in files]).explode("runners")
    lf = lf.select([
        "sport","marketId",
        pl.col("runners").struct.field("selectionId").alias("selectionId"),
        pl.col("marketStartMs"),
        pl.col("marketType").alias("marketType_def"),
        pl.col("countryCode"),
        pl.col("runners").struct.field("handicap"),
        pl.col("runners").struct.field("sortPriority"),
        pl.col("runners").struct.field("reductionFactor"),
    ])
    return lf

def join_all(snap_lf, res_lf, defs_lf):
    df = snap_lf.join(res_lf, on=["sport","marketId","selectionId"], how="inner")
    if defs_lf is not None:
        df = df.join(defs_lf, on=["sport","marketId","selectionId"], how="left")
    return df.collect()

# --------------------------- xgboost helpers --------------------------

def make_params(device):
    if device.lower() == "cuda":
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            max_depth=7, eta=0.08, subsample=0.9, colsample_bytree=0.9
        )
    else:
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            predictor="cpu_predictor",
            max_depth=7, eta=0.08, subsample=0.9, colsample_bytree=0.9
        )

def train(params, dtrain, dvalid, must_gpu=False):
    try:
        return xgb.train(params, dtrain, 500, [(dtrain,"train"),(dvalid,"valid")],
                         early_stopping_rounds=50, verbose_eval=False)
    except xgb.core.XGBoostError as e:
        if must_gpu: die(f"GPU training failed: {e}")
        raise

# -------------------------------- args --------------------------------

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--curated",required=True)
    ap.add_argument("--asof",required=True)
    ap.add_argument("--train-days",type=int,default=12)
    ap.add_argument("--valid-days",type=int,default=3)
    ap.add_argument("--edge-thresh",type=float,default=0.015)
    ap.add_argument("--pm-cutoff",type=float,default=0.65)
    ap.add_argument("--per-market-topk",type=int,default=1)
    ap.add_argument("--stake",choices=["flat","kelly"],default="flat")
    ap.add_argument("--kelly-cap",type=float,default=0.05)
    ap.add_argument("--kelly-floor",type=float,default=0.0)
    ap.add_argument("--bankroll-nom",type=float,default=1000.0)
    ap.add_argument("--ltp-min",type=float,default=1.5)
    ap.add_argument("--ltp-max",type=float,default=5.0)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--sport",default="horse-racing")
    ap.add_argument("--preoff-mins",type=int,default=30)
    ap.add_argument("--commission",type=float,default=0.02)
    return ap.parse_args()

# --------------------------- evaluation logic -------------------------

def evaluate_per_market_topk(df_valid: pl.DataFrame,
                             odds: np.ndarray, y: np.ndarray,
                             p_model: np.ndarray, p_market: np.ndarray,
                             edge_thresh: float, topk: int,
                             lo: float, hi: float,
                             stake_mode: str, kelly_cap: float, kelly_floor: float, bankroll: float,
                             commission: float):
    edge = p_model - p_market
    base = (odds >= lo) & (odds <= hi) & np.isfinite(edge)
    if not base.any():
        return dict(n_trades=0, roi=0.0, profit=0.0)

    df = pl.DataFrame({
        "marketId": df_valid["marketId"].to_numpy()[base],
        "selectionId": df_valid["selectionId"].to_numpy()[base],
        "ltp": odds[base],
        "edge_back": edge[base],
        "y": y[base],
        "p_model": p_model[base],
    }).filter(pl.col("edge_back") >= edge_thresh)

    if df.height == 0:
        return dict(n_trades=0, roi=0.0, profit=0.0)

    df = (
        df.with_columns(pl.rank("dense", descending=True).over("marketId").alias("rk"))
          .filter(pl.col("rk") <= topk)
          .drop("rk")
    )

    outcomes = df["y"].to_numpy().astype(np.float32)
    odds_sel = df["ltp"].to_numpy().astype(np.float32)
    p_model_sel = df["p_model"].to_numpy().astype(np.float32)

    if stake_mode == "kelly":
        def kf(p,o):
            b = o - 1.0
            if b <= 0: return 0.0
            q = 1.0 - p
            return max(0.0, (b*p - q)/b)
        f = np.array([max(kelly_floor, min(kelly_cap, kf(pi, oi))) for pi, oi in zip(p_model_sel, odds_sel)],
                     dtype=np.float32)
        stakes = f * bankroll
    else:
        stakes = np.full_like(odds_sel, 10.0, dtype=np.float32)

    gross = outcomes * (odds_sel - 1.0) * stakes
    net_wins = gross * (1.0 - commission)
    losses = (1.0 - outcomes) * stakes
    profit = net_wins - losses
    pnl = float(profit.sum())
    staked = float(stakes.sum())
    roi = pnl / staked if staked > 0 else 0.0

    return dict(n_trades=int(outcomes.size), roi=roi, profit=pnl)

# -------------------------------- main --------------------------------

def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR","/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    asof_dt = parse_date(args.asof)
    train_end   = asof_dt - timedelta(days=args.valid_days)
    train_start = train_end - timedelta(days=args.train_days - 1)
    valid_start = asof_dt - timedelta(days=args.valid_days - 1)
    valid_end   = asof_dt

    curated = Path(args.curated)

    # Load and join
    snap_lf = load_snapshots(curated, train_start, valid_end, args.sport)
    res_lf  = load_results(curated, train_start, valid_end, args.sport)
    defs_lf = load_defs(curated,  train_start, valid_end, args.sport)
    df_all  = join_all(snap_lf, res_lf, defs_lf)

    # Normalize time columns to ms and compute secs/mins to start
    cols_to_norm = []
    if "publishTimeMs" in df_all.columns: cols_to_norm.append("publishTimeMs")
    if "marketStartMs" in df_all.columns: cols_to_norm.append("marketStartMs")
    if "settledTimeMs" in df_all.columns: cols_to_norm.append("settledTimeMs")

    if cols_to_norm:
        df_all = df_all.with_columns([normalize_ms_expr(pl.col(c)).alias(c) for c in cols_to_norm])

    if "marketStartMs" in df_all.columns:
        df_all = df_all.filter(pl.col("marketStartMs").is_not_null())
        df_all = df_all.with_columns([
            (pl.col("marketStartMs") - pl.col("publishTimeMs")).alias("secs_to_start"),
            ((pl.col("marketStartMs") - pl.col("publishTimeMs"))/60000).alias("mins_to_start"),
        ])
        df_all = df_all.filter(
            (pl.col("mins_to_start") >= 0) &
            (pl.col("mins_to_start") <= args.preoff_mins)
        )

    # Clean labels and odds
    if "winLabel" not in df_all.columns:
        die("winLabel column missing after join (check results load/join)")
    df_all = df_all.filter(pl.col("winLabel").is_not_null())
    df_all = df_all.with_columns(pl.when(pl.col("winLabel") > 0).then(1).otherwise(0).alias("winLabel"))
    if "ltp" not in df_all.columns:
        die("ltp column missing after join (check snapshots load/join)")
    df_all = df_all.filter(pl.col("ltp").is_not_null())

    # Encode categoricals
    for col in ["marketType","marketType_def","countryCode"]:
        if col in df_all.columns:
            df_all = df_all.with_columns(pl.col(col).fill_null("UNK"))
            df_all = df_all.to_dummies(columns=[col])

    # Time split
    ms = lambda dt: int(dt.timestamp() * 1000)
    df_train = df_all.filter(
        (pl.col("publishTimeMs") >= ms(train_start)) &
        (pl.col("publishTimeMs") <= ms(train_end))
    )
    df_valid = df_all.filter(
        (pl.col("publishTimeMs") >= ms(valid_start)) &
        (pl.col("publishTimeMs") <= ms(valid_end))
    )
    if df_train.height == 0 or df_valid.height == 0:
        die("empty train/valid")

    # Feature set
    exclude = {"winLabel","sport","marketId","selectionId","marketStartMs"}
    feats = [c for c in df_all.columns if c not in exclude]

    # Numeric-only for XGBoost
    df_train_num = df_train.select(feats).select(pl.all().exclude(pl.Utf8, pl.Categorical))
    df_valid_num = df_valid.select(feats).select(pl.all().exclude(pl.Utf8, pl.Categorical))

    # Build DMatrix
    dtrain = xgb.DMatrix(df_train_num.to_arrow(), label=df_train["winLabel"].to_numpy().astype(np.float32))
    dvalid = xgb.DMatrix(df_valid_num.to_arrow(), label=df_valid["winLabel"].to_numpy().astype(np.float32))

    # Train
    booster = train(make_params(args.device), dtrain, dvalid, must_gpu=(args.device=="cuda"))

    # Predict
    preds = booster.predict(dvalid)
    y     = df_valid["winLabel"].to_numpy().astype(np.float32)
    odds  = df_valid["ltp"].to_numpy().astype(np.float32)

    p_model  = preds.astype(np.float32)
    p_market = (1.0 / np.clip(odds, 1e-12, None)).astype(np.float32)

    print(f"[Value] logloss={safe_logloss(y, preds):.4f}  auc={roc_auc_score(y, preds):.3f}")

    # Sweep (flat + multiple Kelly variants)
    EDGE_T = [0.010, 0.012, 0.015, 0.020, 0.025]
    TOPK   = [1, 2, 3]
    LTP_W  = [(1.5, 5.0), (1.5, 10.0), (2.0, 6.0)]
    STAKE  = [("flat", None, None, args.bankroll_nom)]
    for cap, floor_ in [(0.05,0.001),(0.05,0.002),(0.10,0.001),(0.10,0.003)]:
        STAKE.append(("kelly", cap, floor_, args.bankroll_nom))

    recs = []
    for e, t, (lo, hi), (sm, cap, floor_, bank) in itertools.product(EDGE_T, TOPK, LTP_W, STAKE):
        m = evaluate_per_market_topk(
            df_valid, odds, y, p_model, p_market,
            edge_thresh=e, topk=t, lo=lo, hi=hi,
            stake_mode=sm, kelly_cap=(cap or 0.0), kelly_floor=(floor_ or 0.0),
            bankroll=(bank or 1000.0), commission=float(args.commission)
        )
        m.update(dict(edge_thresh=e, topk=t, ltp_min=lo, ltp_max=hi,
                      stake_mode=("flat" if sm=="flat" else f"kelly_cap{cap}_floor{floor_}")))
        recs.append(m)

    sweep = pl.DataFrame(recs).sort(["roi","n_trades"], descending=[True, True])
    out   = OUTPUT_DIR / f"edge_sweep_{args.asof}.csv"
    sweep.write_csv(str(out))

    print(f"sweep saved → {out}")
    print(sweep.select(["roi","profit","n_trades","edge_thresh","topk","ltp_min","ltp_max","stake_mode"]).head(15))

if __name__ == "__main__":
    main()
