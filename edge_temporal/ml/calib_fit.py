#!/usr/bin/env python3
# Fits an out-of-fold Isotonic Regression calibrator on the TRAIN window only,
# mirrors your pipeline; saves to pickle for later use in backtests/trading.

import os, sys, glob, pickle, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

# ---- copy of key helpers from training ----
def parse_date(s): return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
def to_ms(dt): return int(dt.timestamp() * 1000)
def compute_windows(start_date_str, asof_str, valid_days):
    start = parse_date(start_date_str); asof = parse_date(asof_str)
    valid_end = asof; valid_start = asof - timedelta(days=valid_days-1)
    train_start = start; train_end = valid_start - timedelta(days=1)
    return train_start, train_end, valid_start, valid_end

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files=[]; cur=start
    while cur<=end:
        d=cur.strftime("%Y-%m-%d"); p=root/sub/f"date={d}"
        files+=glob.glob(str(p/"*.parquet")); cur+=timedelta(days=1)
    return files

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str):
    files=list_parquet_between(curated, Path(f"orderbook_snapshots_5s/sport={sport}"), start, end)
    if not files: return pl.DataFrame()
    lf=pl.scan_parquet(files)
    keep=["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","ltpTick"]
    names=lf.collect_schema().names(); cols=[c for c in keep if c in names]
    return lf.select(cols).collect()

def load_results(curated: Path, start: datetime, end: datetime, sport: str):
    files=list_parquet_between(curated, Path(f"results/sport={sport}"), start, end)
    if not files: return pl.DataFrame()
    lf=pl.scan_parquet(files)
    keep=["sport","marketId","selectionId","winLabel"]; names=lf.collect_schema().names()
    cols=[c for c in keep if c in names]; return lf.select(cols).collect()

def load_defs(curated: Path, start: datetime, end: datetime, sport: str):
    files=list_parquet_between(curated, Path(f"market_definitions/sport={sport}"), start, end)
    if not files: return pl.DataFrame()
    lf=pl.scan_parquet(files); names=lf.collect_schema().names()
    if "runners" in names:
        lf=lf.explode("runners").select([
            "sport","marketId",
            pl.col("runners").struct.field("selectionId").alias("selectionId"),
            pl.col("marketStartMs"), pl.col("marketType").alias("marketType_def"),
            pl.col("countryCode"),
            pl.col("runners").struct.field("handicap"),
            pl.col("runners").struct.field("sortPriority"),
            pl.col("runners").struct.field("reductionFactor"),
        ])
    else:
        have=[c for c in ["sport","marketId","marketStartMs","marketType","countryCode"] if c in names]
        lf=lf.select(have)
    return lf.collect()

def join_all(snap_df, res_df, defs_df):
    if snap_df.is_empty() or res_df.is_empty(): return pl.DataFrame()
    df=snap_df.join(res_df.select(["sport","marketId","selectionId","winLabel"]),
                    on=["sport","marketId","selectionId"], how="inner")
    if not defs_df.is_empty():
        on_cols=[c for c in ["sport","marketId","selectionId"] if c in defs_df.columns and c in df.columns]
        df=df.join(defs_df, on=on_cols, how="left")
    return df

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    for col in ["marketType_def","countryCode"]:
        if col in df.columns:
            df=df.with_columns(pl.col(col).fill_null("UNK"))
            df=df.to_dummies(columns=[col])
    return df

def add_preoff_filter(df: pl.DataFrame, preoff_mins: int) -> pl.DataFrame:
    if "marketStartMs" not in df.columns:
        print("[ERROR] marketStartMs missing", file=sys.stderr); sys.exit(2)
    df=df.filter(pl.col("marketStartMs").is_not_null())
    df=df.with_columns([
        (pl.col("marketStartMs")-pl.col("publishTimeMs")).alias("secs_to_start"),
        ((pl.col("marketStartMs")-pl.col("publishTimeMs"))/60000).alias("mins_to_start"),
    ])
    return df.filter((pl.col("mins_to_start")>=0)&(pl.col("mins_to_start")<=preoff_mins))

def apply_pm_gate(df: pl.DataFrame, pm_cutoff: float) -> pl.DataFrame:
    if "pm_label" in df.columns:
        df=df.filter(pl.col("pm_label").is_not_null()).filter(pl.col("pm_label")>=pm_cutoff)
    return df

def numeric_only(df: pl.DataFrame, exclude:set) -> pl.DataFrame:
    df=df.drop([c for c in exclude if c in df.columns])
    drop_ls=[c for c,dt in zip(df.columns, df.dtypes) if isinstance(dt,(pl.List,pl.Struct))]
    if drop_ls: df=df.drop(drop_ls)
    keep=[c for c,dt in zip(df.columns, df.dtypes) if dt.is_numeric() or dt==pl.Boolean]
    return df.select(keep)

def make_params(device="cuda"):
    return {"objective":"binary:logistic","eval_metric":["logloss","auc"],"tree_method":"hist",
            "device":"cuda" if device!="cpu" else "cpu","max_depth":6,"eta":0.05,
            "subsample":0.8,"colsample_bytree":0.8}

# ---- args & env ----
import argparse
p=argparse.ArgumentParser(description="Fit OOF isotonic calibrator on TRAIN window.")
p.add_argument("--curated", required=True)
p.add_argument("--asof", required=True)
p.add_argument("--start-date", required=True)
p.add_argument("--valid-days", type=int, default=7)
p.add_argument("--sport", default="horse-racing")
p.add_argument("--device", default="cuda")
p.add_argument("--output", default="/opt/BetfairBotML/edge_temporal/output/isotonic.pkl")
args=p.parse_args()

PREOFF_MINS=int(os.environ.get("PREOFF_MINS","30"))
PM_CUTOFF=float(os.environ.get("PM_CUTOFF","0.65"))

# ---- windows ----
train_start, train_end, valid_start, valid_end = compute_windows(args.start_date, args.asof, args.valid_days)
print(f"[calfit] TRAIN: {train_start.date()}..{train_end.date()}  VALID(for ref only): {valid_start.date()}..{valid_end.date()}")

# ---- load train only ----
curated=Path(args.curated)
snap=load_snapshots(curated, train_start, train_end, args.sport)
res =load_results  (curated, train_start, train_end, args.sport)
defs=load_defs     (curated, train_start, train_end, args.sport)
if snap.is_empty() or res.is_empty():
    print("[ERROR] no train data", file=sys.stderr); sys.exit(2)
df=join_all(snap,res,defs)
df=df.filter(pl.col("winLabel").is_not_null()).with_columns(
    pl.when(pl.col("winLabel")>0).then(1).otherwise(0).alias("winLabel")
).filter(pl.col("ltp").is_not_null())
df=add_preoff_filter(df, PREOFF_MINS)
df=apply_pm_gate(df, PM_CUTOFF)
df=encode_categoricals(df)

exclude={"winLabel","sport","marketId","selectionId","marketStartMs","secs_to_start"}
X=numeric_only(df, exclude)
y=df["winLabel"].to_numpy().astype(np.float32)

print(f"[calfit] rows={df.height:,} feats={X.width}")
dmat=xgb.DMatrix(X.to_arrow(), label=y)

# ---- OOF predictions with KFold on TRAIN ----
kf=KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred=np.zeros_like(y, dtype=np.float32)
params=make_params(args.device)

X_np=X.to_numpy()
for fold,(tr,va) in enumerate(kf.split(X_np), start=1):
    dtr=xgb.DMatrix(X_np[tr], label=y[tr])
    dva=xgb.DMatrix(X_np[va], label=y[va])
    bst=xgb.train(params, dtr, num_boost_round=500,
                  evals=[(dtr,"train"),(dva,"valid")],
                  early_stopping_rounds=30, verbose_eval=False)
    oof_pred[va]=bst.predict(dva).astype(np.float32)
    print(f"[calfit] fold {fold}: done, valid={len(va):,}")

# ---- Fit isotonic on OOF ----
iso=IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
iso.fit(oof_pred, y)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "wb") as f:
    pickle.dump({"type":"isotonic","iso":iso}, f)
print(f"[calfit] saved calibrator → {args.output}")
