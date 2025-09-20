#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge Temporal — trainer (snapshots + results + market definitions)

- Reads orderbook_snapshots_5s, results, and market_definitions
- Joins on (sport, marketId, selectionId)
- GPU-first XGBoost training
- Sweep loop with ROI + P&L (flat + kelly)
"""

import argparse, os, sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import itertools

# ---------------------------------------------------------------------

def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def safe_logloss(y_true, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return float(-(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean())

def files_for_range(root: Path, sub: str, start: datetime, end: datetime):
    files=[]
    for d in (start + timedelta(days=i) for i in range((end-start).days+1)):
        dd = d.strftime("%Y-%m-%d")
        ddir = root / sub / f"date={dd}"
        if ddir.exists():
            files.extend(ddir.glob("*.parquet"))
    return files

# ---------------------------------------------------------------------

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str):
    files = files_for_range(curated, f"orderbook_snapshots_5s/sport={sport}", start, end)
    if not files: die("no snapshot files found")
    lfs=[pl.scan_parquet(f) for f in files]
    lf=pl.concat(lfs)
    keep=["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1"]
    lf=lf.select([c for c in keep if c in lf.collect_schema().names()])
    return lf

def load_results(curated: Path, start: datetime, end: datetime, sport: str):
    files = files_for_range(curated, f"results/sport={sport}", start, end)
    if not files: die("no result files found")
    lfs=[pl.scan_parquet(f) for f in files]
    lf=pl.concat(lfs)
    keep=["sport","marketId","selectionId","winLabel"]
    lf=lf.select([c for c in keep if c in lf.collect_schema().names()])
    return lf

def load_defs(curated: Path, start: datetime, end: datetime, sport: str):
    files = files_for_range(curated, f"market_definitions/sport={sport}", start, end)
    if not files: return None
    lfs=[pl.scan_parquet(f) for f in files]
    lf=pl.concat(lfs)
    lf=lf.explode("runners")
    lf=lf.select([
        "sport","marketId",
        pl.col("runners")["selectionId"].alias("selectionId"),
        pl.col("marketStartMs").alias("marketStartMs"),
        pl.col("marketType").alias("marketType"),
        pl.col("countryCode").alias("countryCode"),
    ])
    return lf

def join_all(snap_lf, res_lf, defs_lf):
    df = snap_lf.join(res_lf, on=["sport","marketId","selectionId"], how="inner")
    if defs_lf is not None:
        df = df.join(defs_lf, on=["sport","marketId","selectionId"], how="left")
    return df.collect()

# ---------------------------------------------------------------------

def make_params(device):
    if device.lower()=="cuda":
        return dict(objective="binary:logistic",eval_metric="logloss",
                    tree_method="gpu_hist",predictor="gpu_predictor",
                    max_depth=7,eta=0.08,subsample=0.9,colsample_bytree=0.9,max_bin=256)
    else:
        return dict(objective="binary:logistic",eval_metric="logloss",
                    tree_method="hist",predictor="cpu_predictor",
                    max_depth=7,eta=0.08,subsample=0.9,colsample_bytree=0.9,max_bin=512)

def train(params,dtrain,dvalid,must_gpu=False):
    try:
        return xgb.train(params,dtrain,500,[(dtrain,"train"),(dvalid,"valid")],
                         early_stopping_rounds=50,verbose_eval=False)
    except xgb.core.XGBoostError as e:
        if must_gpu: die(f"gpu training failed: {e}")
        raise

# ---------------------------------------------------------------------

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
    # passthrough args from .sh
    ap.add_argument("--sport",default="horse-racing")
    ap.add_argument("--preoff-mins",type=int,default=30)
    ap.add_argument("--downsample-secs",type=int,default=5)
    ap.add_argument("--commission",type=float,default=0.02)
    ap.add_argument("--pm-horizon-secs",type=int,default=300)
    ap.add_argument("--pm-tick-threshold",type=int,default=1)
    ap.add_argument("--pm-slack-secs",type=int,default=3)
    ap.add_argument("--edge-prob",default="cal")
    ap.add_argument("--market-prob",default="overround")
    ap.add_argument("--side",choices=["back","lay"],default="back")
    return ap.parse_args()

# ---------------------------------------------------------------------

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    for col in ["marketType","countryCode"]:
        if col in df.columns:
            df=df.with_columns(pl.col(col).fill_null("UNK"))
            df=df.to_dummies(columns=[col])
    return df

def evaluate(df,odds,y,p_model,p_market,edge_t,topk,lo,hi,stake_mode,cap,floor_,bank,commission):
    edge=p_model-p_market
    mask=(odds>=lo)&(odds<=hi)&np.isfinite(edge)&(edge>=edge_t)
    if not mask.any(): return dict(n_trades=0,roi=0.0,profit=0.0)
    outcomes=y[mask]; odds_sel=odds[mask]; p_model_sel=p_model[mask]
    if stake_mode=="kelly":
        def kf(p,o): b=o-1; q=1-p; return max(0,(b*p-q)/b) if b>0 else 0
        f=np.array([max(floor_,min(cap,kf(pi,oi))) for pi,oi in zip(p_model_sel,odds_sel)],dtype=np.float32)
        stakes=f*bank
    else:
        stakes=np.full_like(odds_sel,10.0,dtype=np.float32)
    gross=outcomes*(odds_sel-1.0)*stakes
    net=gross*(1.0-commission)
    losses=(1-outcomes)*stakes
    profit=net-losses
    pnl=float(profit.sum()); staked=float(stakes.sum())
    roi=pnl/staked if staked>0 else 0.0
    return dict(n_trades=int(outcomes.size),roi=roi,profit=pnl)

# ---------------------------------------------------------------------

def main():
    args=parse_args()
    OUTPUT_DIR=Path(os.environ.get("OUTPUT_DIR","/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    asof_dt=parse_date(args.asof)
    train_end   = asof_dt - timedelta(days=args.valid_days)
    train_start = train_end - timedelta(days=args.train_days - 1)
    valid_start = asof_dt - timedelta(days=args.valid_days - 1)
    valid_end   = asof_dt

    curated=Path(args.curated)
    snap_lf=load_snapshots(curated,train_start,valid_end,args.sport)
    res_lf=load_results(curated,train_start,valid_end,args.sport)
    defs_lf=load_defs(curated,train_start,valid_end,args.sport)
    df_all=join_all(snap_lf,res_lf,defs_lf)

    if "marketStartMs" in df_all.columns:
        df_all=df_all.with_columns((pl.col("marketStartMs")-pl.col("publishTimeMs")).alias("secs_to_start"))

    df_all=encode_categoricals(df_all)

    # split
    df_train=df_all.filter((pl.col("publishTimeMs")>=int(train_start.timestamp()*1000)) &
                           (pl.col("publishTimeMs")<=int(train_end.timestamp()*1000)))
    df_valid=df_all.filter((pl.col("publishTimeMs")>=int(valid_start.timestamp()*1000)) &
                           (pl.col("publishTimeMs")<=int(valid_end.timestamp()*1000)))

    feats=[c for c in df_all.columns if c not in ("winLabel","sport","marketId","selectionId","marketStartMs")]

    dtrain = xgb.DMatrix(df_train.select(feats).to_arrow(),
                         label=df_train["winLabel"].to_numpy(dtype=np.float32))
    dvalid = xgb.DMatrix(df_valid.select(feats).to_arrow(),
                         label=df_valid["winLabel"].to_numpy(dtype=np.float32))

    booster=train(make_params(args.device),dtrain,dvalid,must_gpu=(args.device=="cuda"))

    preds=booster.predict(dvalid)
    y=df_valid["winLabel"].to_numpy(dtype=np.float32)
    odds=df_valid["ltp"].to_numpy(dtype=np.float32)
    p_model=preds.astype(np.float32)
    p_market=(1.0/np.clip(odds,1e-12,None)).astype(np.float32)

    print(f"[Value] logloss={safe_logloss(y,preds):.4f} auc={roc_auc_score(y,preds):.3f}")

    EDGE_T=[0.010,0.012,0.015]; TOPK=[1,2]; LTP_W=[(1.5,5.0)]
    STAKE=[("flat",None,None,args.bankroll_nom),("kelly",args.kelly_cap,args.kelly_floor,args.bankroll_nom)]
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
