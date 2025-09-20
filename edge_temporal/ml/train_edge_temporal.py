#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge Temporal — rolling window, RAM-safe loader, GPU-first XGBoost, and RAM-only sweep (Flat + Kelly with P&L).

- Loads ONLY files that contain the required cols.
- Uses Polars scan() + column projection to avoid RAM blow-ups.
- Casts numerics to float32 early.
- Builds XGBoost DMatrix from Arrow (no giant NumPy copies).
- Trains once on GPU when --device=cuda (errors if GPU not usable).
- Sweeps selection/staking configs entirely in memory and writes a leaderboard CSV with ROI & P&L.
"""

import argparse, os, sys, re, glob
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import itertools

ID_COLS = {"marketId", "selectionId"}
LABEL_COLS = {"winLabel", "pm_label", "pm_future_ticks"}
ODDS_COLS = {"ltp"}
DATE_COL_CANDIDATES = ("asof", "eventDate", "marketDate", "date")
REQUIRED_COLS = {"winLabel", "ltp", "marketId", "selectionId"}

def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def extract_date_from_name(p: Path) -> datetime | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}", p.as_posix())
    return parse_date(m.group(0)) if m else None

def safe_logloss(y_true, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

def glob_files(curated_root: str, start: datetime, end: datetime) -> list[Path]:
    root = Path(curated_root)
    if not root.exists():
        die(f"CURATED root not found: {root}")
    pats = os.environ.get("FEATURE_GLOB", f"{root}/**/*.parquet").split(";")
    pats_csv = os.environ.get("FEATURE_CSV_GLOB", f"{root}/**/*.csv").split(";")
    files = []
    for pat in pats: files.extend(glob.glob(pat, recursive=True))
    for pat in pats_csv: files.extend(glob.glob(pat, recursive=True))
    files = [Path(f) for f in files if Path(f).is_file()]
    sel = []
    for f in files:
        dt = extract_date_from_name(f)
        if dt is not None and start <= dt <= end:
            sel.append(f)
    if not sel:
        die(f"No curated files found in range {start.date()} → {end.date()}")
    return sel

def filter_files_by_required_cols(files: list[Path], required: set[str]) -> list[Path]:
    out = []
    for f in files:
        try:
            lf = pl.scan_parquet(f) if f.suffix==".parquet" else pl.scan_csv(f, infer_schema_length=200_000)
            schema = lf.collect_schema()
            if required.issubset(set(schema.names())):
                out.append(f)
        except:
            continue
    return out

def infer_schema_from_scans(files: list[Path]) -> dict:
    schema: dict[str, pl.DataType] = {}
    for f in files:
        try:
            lf = pl.scan_parquet(f) if f.suffix.lower()==".parquet" else pl.scan_csv(f, infer_schema_length=200_000)
            for k, v in lf.collect_schema().items():
                if k not in schema: schema[k] = v
        except Exception:
            continue
    if not schema:
        die("Failed to infer schema from curated files")
    return schema

def select_feature_cols_from_schema(schema: dict) -> list[str]:
    exclude = ID_COLS | LABEL_COLS | ODDS_COLS | set(DATE_COL_CANDIDATES)
    feats = []
    for c, dt in schema.items():
        if c in exclude: continue
        if dt in pl.NUMERIC_DTYPES: feats.append(c)
    if not feats:
        die("No numeric feature cols found in schema.")
    return feats

def pick_date_col(schema: dict) -> str | None:
    for c in DATE_COL_CANDIDATES:
        if c in schema: return c
    return None

def scan_project_collect(files: list[Path], cols: list[str], casts_float32: set[str], date_col_name: str | None) -> pl.DataFrame:
    lfs = []
    for f in files:
        if f.suffix.lower()==".parquet":
            lf = pl.scan_parquet(f)
        elif f.suffix.lower()==".csv":
            lf = pl.scan_csv(f, infer_schema_length=200_000)
        else:
            continue
        select_cols = [c for c in cols if c in lf.collect_schema().names()]
        lf = lf.select([
            (pl.col(c).cast(pl.Float32) if c in casts_float32 else pl.col(c))
            for c in select_cols
        ])
        if date_col_name is None:
            fd = extract_date_from_name(f)
            if fd is None: continue
            lf = lf.with_columns(pl.lit(fd).alias("__fileDate").cast(pl.Datetime))
        lfs.append(lf)
    if not lfs:
        die("No usable files after projection")
    return pl.concat(lfs).collect()

def make_xgb_params(device: str):
    if device.lower() == "cuda":
        return dict(objective="binary:logistic", eval_metric="logloss",
                    tree_method="gpu_hist", predictor="gpu_predictor",
                    max_depth=7, eta=0.08, subsample=0.9,
                    colsample_bytree=0.9, max_bin=256)
    else:
        return dict(objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", predictor="cpu_predictor",
                    max_depth=7, eta=0.08, subsample=0.9,
                    colsample_bytree=0.9, max_bin=512)

def train_xgb(params, dtrain, dvalid, must_use_gpu=False):
    try:
        booster = xgb.train(params, dtrain, num_boost_round=500,
                            evals=[(dtrain, "train"), (dvalid, "valid")],
                            early_stopping_rounds=50, verbose_eval=False)
        return booster
    except xgb.core.XGBoostError as e:
        if must_use_gpu: die(f"GPU training failed: {e}")
        raise

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--train-days", type=int, default=12)
    ap.add_argument("--valid-days", type=int, default=3)
    ap.add_argument("--edge-thresh", type=float, default=0.015)
    ap.add_argument("--pm-cutoff", type=float, default=0.65)
    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--stake", choices=["flat","kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.0)
    ap.add_argument("--bankroll-nom", type=float, default=1000.0)
    ap.add_argument("--ltp-min", type=float, default=1.5)
    ap.add_argument("--ltp-max", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--commission", type=float, default=0.02)
    return ap.parse_args()

def evaluate_selection(df_valid: pl.DataFrame, odds, y, p_model, p_market,
                       edge_thresh, pm_cutoff, topk, lo, hi,
                       stake_mode, kelly_cap, kelly_floor, bankroll, commission):
    edge = p_model - p_market
    base = (odds >= lo) & (odds <= hi) & np.isfinite(edge)
    if not base.any():
        return dict(n_trades=0, roi=0.0, profit=0.0)
    df = pl.DataFrame({
        "marketId": df_valid["marketId"].to_numpy()[base],
        "selectionId": df_valid["selectionId"].to_numpy()[base],
        "ltp": odds[base], "edge_back": edge[base],
        "y": y[base], "p_model": p_model[base],
    }).filter(pl.col("edge_back") >= edge_thresh)
    if df.height == 0: return dict(n_trades=0, roi=0.0, profit=0.0)
    df = (df.with_columns(pl.rank("dense", descending=True).over("marketId").alias("rk"))
            .filter(pl.col("rk") <= topk).drop("rk"))
    outcomes = df["y"].to_numpy().astype(np.float32)
    odds_sel = df["ltp"].to_numpy().astype(np.float32)
    p_model_sel = df["p_model"].to_numpy().astype(np.float32)
    if stake_mode == "kelly":
        def kf(p, o): b=o-1.0; q=1.0-p; return max(0.0,(b*p-q)/b) if b>0 else 0.0
        f = np.array([max(kelly_floor, min(kelly_cap, kf(pi, oi))) for pi,oi in zip(p_model_sel, odds_sel)], dtype=np.float32)
        stakes = f * bankroll
    else:
        stakes = np.full_like(odds_sel, 10.0, dtype=np.float32)
    gross = outcomes * (odds_sel - 1.0) * stakes
    net_wins = gross * (1.0 - commission)
    losses = (1.0 - outcomes) * stakes
    profit = net_wins - losses
    pnl = float(profit.sum()); staked = float(stakes.sum())
    roi = pnl/staked if staked>0 else 0.0
    return dict(n_trades=int(outcomes.size), roi=roi, profit=pnl)

def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR","/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asof_dt = parse_date(args.asof)
    train_end = asof_dt - timedelta(days=args.valid_days)
    train_start = train_end - timedelta(days=args.train_days-1)
    valid_start = asof_dt - timedelta(days=args.valid_days-1)
    valid_end = asof_dt
    files = glob_files(args.curated, train_start, valid_end)
    files = filter_files_by_required_cols(files, REQUIRED_COLS)
    schema = infer_schema_from_scans(files)
    date_col = pick_date_col(schema)
    feat_cols = select_feature_cols_from_schema(schema)
    needed_cols = list(REQUIRED_COLS | set(feat_cols))
    if date_col: needed_cols.append(date_col)
    casts_float32 = set(feat_cols) | {"winLabel","ltp"}
    df_all = scan_project_collect(files, needed_cols, casts_float32, date_col)
    if date_col: df_all = df_all.with_columns(pl.col(date_col).cast(pl.Datetime))
    else: date_col="__fileDate"
    df_train = df_all.filter((pl.col(date_col)>=train_start)&(pl.col(date_col)<=train_end))
    df_valid = df_all.filter((pl.col(date_col)>=valid_start)&(pl.col(date_col)<=valid_end))
    dtrain = xgb.DMatrix(df_train.select(feat_cols).to_arrow(), label=df_train["winLabel"].to_numpy(np.float32))
    dvalid = xgb.DMatrix(df_valid.select(feat_cols).to_arrow(), label=df_valid["winLabel"].to_numpy(np.float32))
    booster = train_xgb(make_xgb_params(args.device), dtrain, dvalid, must_use_gpu=(args.device=="cuda"))
    preds = booster.predict(dvalid)
    y_valid = df_valid["winLabel"].to_numpy(np.float32)
    odds_valid = df_valid["ltp"].to_numpy(np.float32)
    p_model = preds.astype(np.float32)
    p_market = (1.0/np.clip(odds_valid,1e-12,None)).astype(np.float32)
    print(f"[Value] logloss={safe_logloss(y_valid,preds):.4f} auc={roc_auc_score(y_valid,preds):.3f}")
    EDGE_THRESHES=[0.010,0.012,0.015]; PM_CUTOFFS=[0.60,0.65]; TOPK=[1,2]; LTP_W=[(1.5,5.0)]
    STAKE_MODES=[("flat",None,None,args.bankroll_nom),("kelly",args.kelly_cap,args.kelly_floor,args.bankroll_nom)]
    recs=[]
    for e,pm,t,(lo,hi),(sm,cap,floor_,bank) in itertools.product(EDGE_THRESHES,PM_CUTOFFS,TOPK,LTP_W,STAKE_MODES):
        m=evaluate_selection(df_valid,odds_valid,y_valid,p_model,p_market,e,pm,t,lo,hi,sm,cap or 0.0,floor_ or 0.0,bank or 1000.0,args.commission)
        m.update(dict(edge_thresh=e,pm_cutoff=pm,topk=t,ltp_min=lo,ltp_max=hi,stake_mode=sm,kelly_cap=cap,kelly_floor=floor_,bankroll_nom=bank))
        recs.append(m)
    sweep=pl.DataFrame(recs).sort(["roi","n_trades"],descending=[True,True])
    out_csv=OUTPUT_DIR/f"edge_sweep_{args.asof}.csv"
    sweep.write_csv(str(out_csv))
    print(f"Sweep saved → {out_csv}")
    print(sweep.select(["roi","profit","n_trades","edge_thresh","pm_cutoff","stake_mode"]).head(10))

if __name__=="__main__":
    main()
