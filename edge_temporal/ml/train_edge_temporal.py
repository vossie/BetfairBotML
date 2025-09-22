#!/usr/bin/env python3
import os, sys, glob, itertools
from pathlib import Path
from datetime import datetime, timedelta, timezone
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)
def safe_logloss(y_true, y_pred):
    try:
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        return log_loss(y_true, y_pred)
    except Exception:
        return float("nan")
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

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files=[]; cur=start
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
        have = [c for c in ["sport","marketId","marketStartMs","marketType","countryCode"] if c in lf.collect_schema().names()]
        lf = lf.select(have)
    return lf.collect()

def join_all(snap_df: pl.DataFrame, res_df: pl.DataFrame, defs_df: pl.DataFrame):
    if snap_df.is_empty() or res_df.is_empty(): return pl.DataFrame()
    df = snap_df.join(
        res_df.select([c for c in ["sport","marketId","selectionId","winLabel","marketType","runnerStatus"] if c in res_df.columns]),
        on=["sport","marketId","selectionId"], how="inner"
    )
    if not defs_df.is_empty():
        on_cols = [c for c in ["sport","marketId","selectionId"] if c in defs_df.columns and c in df.columns]
        df = df.join(defs_df, on=on_cols, how="left")
    return df

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

def apply_pm_gate(df: pl.DataFrame, pm_cutoff: float) -> pl.DataFrame:
    if "pm_label" in df.columns:
        df = df.filter(pl.col("pm_label").is_not_null())
        df = df.filter(pl.col("pm_label") >= pm_cutoff)
    return df

def numeric_only(df: pl.DataFrame, exclude: set) -> pl.DataFrame:
    df = df.drop([c for c in exclude if c in df.columns])
    drop_ls = [c for c, dt in zip(df.columns, df.dtypes) if isinstance(dt, (pl.List, pl.Struct))]
    if drop_ls: df = df.drop(drop_ls)
    keep = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric() or isinstance(dt, pl.Boolean)]
    return df.select(keep)

def make_params(device="cuda"):
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc"],
        "tree_method": "hist",
        "device": "cuda" if device != "cpu" else "cpu",
        "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8
    }

def train(params, dtrain, dvalid):
    return xgb.train(params, dtrain, 500, [(dtrain,"train"),(dvalid,"valid")],
                     early_stopping_rounds=30, verbose_eval=False)

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
    df = df.with_columns(
        pl.col("edge").rank(method="dense", descending=True).over("marketId").alias("rk")
    ).filter(pl.col("rk") <= topk).drop("rk")
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

def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    asof_dt = parse_date(args.asof)
    start_dt = parse_date(args.start_date)

    # Training / validation split
    train_end   = asof_dt - timedelta(days=args.valid_days)
    train_start = start_dt
    valid_start = train_end + timedelta(days=1)
    valid_end   = asof_dt

    print(f"Training window:   {train_start.date()} .. {train_end.date()}")
    print(f"Validation window: {valid_start.date()} .. {valid_end.date()}")

    curated = Path(args.curated)

    # Load datasets
    snap_lf = load_snapshots(curated, train_start, valid_end, args.sport)
    defs_lf = load_defs(curated, train_start, valid_end, args.sport)
    res_lf  = load_results(curated, train_start, valid_end, args.sport)

    # Join — only keep winLabel from results
    res_lf = res_lf.select(["marketId", "selectionId", "winLabel"])
    df_all = join_snapshots_results(snap_lf, res_lf).join(defs_lf, on=["marketId", "selectionId"], how="left")

    # Drop leakage-prone cols
    drop_cols = {"runnerStatus", "settledTimeMs", "eventId", "marketType"}
    df_all = df_all.drop([c for c in drop_cols if c in df_all.columns])

    # Split train/valid
    df_train = df_all.filter(
        (pl.col("publishTimeMs") >= int(train_start.replace(tzinfo=timezone.utc).timestamp() * 1000)) &
        (pl.col("publishTimeMs") <  int((train_end + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp() * 1000))
    )
    df_valid = df_all.filter(
        (pl.col("publishTimeMs") >= int(valid_start.replace(tzinfo=timezone.utc).timestamp() * 1000)) &
        (pl.col("publishTimeMs") <  int((valid_end + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp() * 1000))
    )

    feats = [c for c in df_all.columns if c not in ("winLabel", "sport", "marketId", "selectionId")]
    print(f"[rows] train={df_train.height:,}  valid={df_valid.height:,}  features={len(feats)}")
    print("Features:", feats)

    # Build matrices
    dtrain = xgb.DMatrix(df_train.select(feats).to_arrow(), label=df_train["winLabel"].to_numpy().astype(np.float32))
    dvalid = xgb.DMatrix(df_valid.select(feats).to_arrow(), label=df_valid["winLabel"].to_numpy().astype(np.float32))

    # Train
    booster = train(make_params(args.device), dtrain, dvalid, must_gpu=(args.device == "cuda"))
    preds = booster.predict(dvalid)

    # Validation metrics
    y_valid = df_valid["winLabel"].to_numpy().astype(np.float32)
    odds    = df_valid["ltp"].to_numpy().astype(np.float32)
    p_model = preds.astype(np.float32)
    p_market = (1.0 / np.clip(odds, 1e-12, None)).astype(np.float32)

    print(f"[Value] logloss={safe_logloss(y_valid, preds):.4f}  auc={roc_auc_score(y_valid, preds):.3f}")

    # Sweep
    EDGE_T = [float(os.environ.get("EDGE_THRESH", 0.015))]
    TOPK   = [int(os.environ.get("PER_MARKET_TOPK", 1))]
    LTP_W  = [(float(os.environ.get("LTP_MIN", 1.5)), float(os.environ.get("LTP_MAX", 5.0)))]
    STAKE  = [
        ("flat", None, None, args.bankroll_nom),
        ("kelly", args.kelly_cap, args.kelly_floor, args.bankroll_nom),
    ]

    recs = []
    for e, t, (lo, hi), (sm, cap, floor_, bank) in itertools.product(EDGE_T, TOPK, LTP_W, STAKE):
        m = evaluate(df_valid, odds, y_valid, p_model, p_market,
                     e, t, lo, hi, sm, cap or 0.0, floor_ or 0.0, bank or 1000.0, args.commission)
        m.update(dict(edge_thresh=e, topk=t, ltp_min=lo, ltp_max=hi, stake_mode=sm))
        recs.append(m)

    sweep = pl.DataFrame(recs).sort(["roi", "n_trades"], descending=[True, True])
    out = OUTPUT_DIR / f"edge_sweep_{args.asof}.csv"
    sweep.write_csv(str(out))
    print(f"sweep saved → {out}")
    print(sweep.select(["roi", "profit", "n_trades", "edge_thresh", "stake_mode"]).head(10))


if __name__ == "__main__":
    main()
