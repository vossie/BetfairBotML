#!/usr/bin/env python3
"""
Forward tester with realism controls:
- --dedupe-mode {none,first_cross,max_edge}
- --downsample-secs N
- --haircut-ticks N  (worse price for back bets by N Betfair ticks)
Outputs per-day ROI.

Assumes back bets (P&L = y*(odds-1)*stake - (1-y)*stake).
"""
import os, sys, glob, json, pickle, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import numpy as np
import xgboost as xgb

# ---------- utils ----------
def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files=[]; cur=start
    while cur<=end:
        d=cur.strftime("%Y-%m-%d"); p=root/sub/f"date={d}"
        files+=glob.glob(str(p/"*.parquet")); cur+=timedelta(days=1)
    return files

def _collect_gpu(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="gpu")
    except Exception:
        return lf.collect()

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files=list_parquet_between(curated, f"orderbook_snapshots_5s/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf=pl.scan_parquet(files)
    keep=["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","ltpTick"]
    names=lf.collect_schema().names(); cols=[c for c in keep if c in names]
    return _collect_gpu(lf.select(cols))

def load_results(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files=list_parquet_between(curated, f"results/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf=pl.scan_parquet(files)
    keep=["sport","marketId","selectionId","winLabel"]; names=lf.collect_schema().names()
    cols=[c for c in keep if c in names]; return _collect_gpu(lf.select(cols))

def load_defs(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files=list_parquet_between(curated, f"market_definitions/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf=pl.scan_parquet(files); names=lf.collect_schema().names()
    if "runners" in names:
        lf=lf.explode("runners").select([
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
        have=[c for c in ["sport","marketId","marketStartMs","marketType","countryCode"] if c in names]
        lf=lf.select(have)
    return _collect_gpu(lf)

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

def add_preoff_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "marketStartMs" not in df.columns:
        print("[ERROR] marketStartMs missing after join", file=sys.stderr); sys.exit(2)
    df=df.filter(pl.col("marketStartMs").is_not_null())
    return df.with_columns([
        (pl.col("marketStartMs")-pl.col("publishTimeMs")).alias("secs_to_start"),
        ((pl.col("marketStartMs")-pl.col("publishTimeMs"))/60000).alias("mins_to_start"),
    ])

def numeric_only(df: pl.DataFrame, exclude:set) -> pl.DataFrame:
    df=df.drop([c for c in exclude if c in df.columns])
    drop_ls=[c for c,dt in zip(df.columns, df.dtypes) if isinstance(dt,(pl.List,pl.Struct))]
    if drop_ls: df=df.drop(drop_ls)
    keep=[c for c,dt in zip(df.columns, df.dtypes) if dt.is_numeric() or dt==pl.Boolean]
    return df.select(keep)

# Betfair tick utilities (odds ladder)
def _tick_size(o: float) -> float:
    if o < 2.0:      return 0.01
    if o < 3.0:      return 0.02
    if o < 4.0:      return 0.05
    if o < 6.0:      return 0.10
    if o < 10.0:     return 0.20
    if o < 20.0:     return 0.50
    if o < 30.0:     return 1.00
    if o < 50.0:     return 2.00
    if o < 100.0:    return 5.00
    return 10.0  # 100-1000

def shift_ticks_down(odds: float, n_ticks: int) -> float:
    """Worsen a BACK price by moving down n Betfair ticks."""
    if n_ticks <= 0: return max(1.01, odds)
    o = odds
    for _ in range(n_ticks):
        step = _tick_size(o)
        o = max(1.01, round(o - step + 1e-12, 2))
    return o

def evaluate(df_valid, odds, y, p_model, p_market_norm, edge_thresh, topk, lo, hi,
             stake_mode, cap, floor_, bank, commission):
    edge = p_model - p_market_norm
    mask = (odds >= lo) & (odds <= hi) & np.isfinite(edge)
    if not mask.any(): return dict(roi=0.0, profit=0.0, n_trades=0)
    df = pl.DataFrame({
        "marketId": df_valid["marketId"].to_numpy()[mask],
        "publishTimeMs": df_valid["publishTimeMs"].to_numpy()[mask],
        "selectionId": df_valid["selectionId"].to_numpy()[mask],
        "ltp": odds[mask],
        "edge": edge[mask],
        "y": y[mask],
        "p_model": p_model[mask],
    }).filter(pl.col("edge") >= edge_thresh)
    if df.height == 0: return dict(roi=0.0, profit=0.0, n_trades=0)
    df = df.with_columns(
        pl.col("edge").rank(method="dense", descending=True).over(["marketId","publishTimeMs"]).alias("rk")
    ).filter(pl.col("rk") <= topk).drop("rk")

    outcomes = df["y"].to_numpy().astype(np.float32)
    odds_sel = df["ltp"].to_numpy().astype(np.float32)

    if stake_mode == "kelly":
        def kf(p,o):
            b=o-1.0
            if b<=0: return 0.0
            q=1.0-p
            return max(0.0,(b*p - q)/b)
        p_model_sel = df["p_model"].to_numpy().astype(np.float32)
        f=np.array([max(floor_, min(cap, kf(pi,oi))) for pi,oi in zip(p_model_sel,odds_sel)],dtype=np.float32)
        stakes=f*bank
    else:
        stakes=np.full_like(odds_sel,10.0,dtype=np.float32)

    gross=outcomes*(odds_sel-1.0)*stakes
    net=gross*(1.0-commission)
    loss=(1.0-outcomes)*stakes
    profit=net-loss
    pnl=float(profit.sum())
    staked=float(stakes.sum())
    roi=pnl/staked if staked>0 else 0.0
    return dict(roi=roi, profit=pnl, n_trades=int(outcomes.size))

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--model-dir", required=True, help="Dir containing model.json, isotonic.pkl, best_config.json")
    ap.add_argument("--preoff-max", type=int, default=180)
    ap.add_argument("--commission", type=float, default=None, help="Override commission; default from best_config")
    ap.add_argument("--stake-mode", choices=["flat","kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.002)
    ap.add_argument("--bankroll-nom", type=float, default=None, help="Override bankroll; default from best_config")
    ap.add_argument("--downsample-secs", type=int, default=0, help="0 = disabled")
    ap.add_argument("--dedupe-mode", choices=["none","first_cross","max_edge"], default="none")
    ap.add_argument("--haircut-ticks", type=int, default=0, help="Worsen back odds by N ticks (0 = none)")
    ap.add_argument("--output-csv", required=True)
    args=ap.parse_args()

    model_dir=Path(args.model_dir)
    with open(model_dir/"best_config.json","r") as f:
        best=json.load(f)
    booster=xgb.Booster(); booster.load_model(str(model_dir/"model.json"))
    with open(model_dir/"isotonic.pkl","rb") as f:
        iso=pickle.load(f)["iso"]

    pm_cutoff   = float(best["best_params"]["pm_cutoff"])
    edge_thresh = float(best["best_params"]["edge_thresh"])
    ltp_min     = float(best["best_params"]["ltp_min"])
    ltp_max     = float(best["best_params"]["ltp_max"])
    commission  = float(args.commission if args.commission is not None else best.get("commission", 0.02))
    bankroll_nom= float(args.bankroll_nom if args.bankroll_nom is not None else best.get("bankroll_nom", 5000.0))

    start=parse_date(args.start_date); end=parse_date(args.end_date)
    curated=Path(args.curated)

    # Load
    snap=load_snapshots(curated, start, end, args.sport)
    res =load_results  (curated, start, end, args.sport)
    defs=load_defs     (curated, start, end, args.sport)
    if snap.is_empty():
        print(f"[ERROR] no snapshots under {curated}/orderbook_snapshots_5s/sport={args.sport} for {args.start_date}..{args.end_date}", file=sys.stderr); sys.exit(2)
    if res.is_empty():
        print(f"[ERROR] no results under {curated}/results/sport={args.sport} for {args.start_date}..{args.end_date}", file=sys.stderr); sys.exit(2)

    df=join_all(snap,res,defs)
    if df.is_empty(): print("[ERROR] empty after join", file=sys.stderr); sys.exit(2)

    # Clean + constraints
    df=(df.filter(pl.col("winLabel").is_not_null())
          .with_columns(pl.when(pl.col("winLabel")>0).then(1).otherwise(0).alias("winLabel"))
          .filter(pl.col("ltp").is_not_null()))
    df=add_preoff_columns(df).filter((pl.col("mins_to_start")>=0) & (pl.col("mins_to_start")<=args.preoff_max))
    df=encode_categoricals(df)

    # Full time mask
    ts_np=df["publishTimeMs"].to_numpy()
    mask_time=(ts_np >= to_ms(start)) & (ts_np < to_ms(end + timedelta(days=1)))

    # PM gate
    if "pm_label" in df.columns:
        pm_mask=df["pm_label"].to_numpy() >= pm_cutoff
    else:
        pm_mask=np.ones(df.height, dtype=bool)

    use_mask=mask_time & pm_mask
    if not use_mask.any():
        print("[WARN] no rows after masks")
        pl.DataFrame({"day":[], "roi":[], "profit":[], "n_trades":[]}).write_csv(args.output_csv)
        return

    # Features + labels for masked slice
    exclude={"winLabel","sport","marketId","selectionId","marketStartMs","secs_to_start"}
    X_all=numeric_only(df, exclude).to_numpy()
    y_all=df["winLabel"].to_numpy().astype(np.float32)

    X=X_all[use_mask].astype(np.float32, copy=False)
    y=y_all[use_mask]

    base = (
        df.select(["marketId","publishTimeMs","selectionId","ltp","winLabel"])
          .with_columns(pl.Series("__mask", use_mask))
          .filter(pl.col("__mask")).drop("__mask")
          .with_columns(
              pl.from_epoch(pl.col("publishTimeMs"), time_unit="ms").dt.replace_time_zone("UTC").dt.date().alias("__day")
          )
    )

    # Optional downsample per runner into time buckets
    if args.downsample_secs and args.downsample_secs > 0:
        bucket = (pl.col("publishTimeMs") // (args.downsample_secs*1000)).alias("__bucket")
        base = (base
                .with_columns(bucket)
                .sort(["marketId","selectionId","publishTimeMs"])
                .unique(subset=["marketId","selectionId","__bucket"], keep="first")
                .drop("__bucket"))

        # Rebuild X,y to match base rows:
        keys = ["marketId","selectionId","publishTimeMs"]
        masked_df = (df.select(keys)
                       .with_columns(pl.Series("__mask", use_mask))
                       .filter(pl.col("__mask")).drop("__mask"))
        masked_df = masked_df.with_row_count("__idx")
        base = base.join(masked_df, on=keys, how="left")
        idx = base["__idx"].to_numpy()
        good = np.isfinite(idx)
        base = base.filter(pl.Series(good))
        idx = idx[good].astype(np.int64)
        X = X[idx]
        y = y[idx]

    # Predictions (raw -> isotonic-calibrated)
    p_raw = booster.inplace_predict(X).astype(np.float32)
    p_cal = iso.predict(p_raw).astype(np.float32)

    # Normalized market probabilities (overround fix) on the *current base*
    dv = base.select(["marketId","publishTimeMs","ltp","__day"]).with_columns(
        (1.0/pl.col("ltp").clip(lower_bound=1e-12)).alias("__inv")
    )
    sums = dv.group_by(["marketId","publishTimeMs"]).agg(pl.col("__inv").sum().alias("__inv_sum"))
    dv = dv.join(sums, on=["marketId","publishTimeMs"], how="left").with_columns(
        (pl.col("__inv")/pl.col("__inv_sum").clip(lower_bound=1e-12)).alias("__p_mkt_norm")
    )

    p_mkt = dv["__p_mkt_norm"].to_numpy().astype(np.float32)
    odds  = base["ltp"].to_numpy().astype(np.float32)
    days  = base["__day"].to_list()

    # Apply haircuts (worsen back odds by N ticks)
    if args.haircut_ticks and args.haircut_ticks > 0:
        odds = np.array([shift_ticks_down(float(o), int(args.haircut_ticks)) for o in odds], dtype=np.float32)

    # Build working frame with model + market probs and edge
    work = pl.DataFrame({
        "marketId": base["marketId"],
        "selectionId": base["selectionId"],
        "publishTimeMs": base["publishTimeMs"],
        "__day": base["__day"],
        "ltp": odds,
        "y": y,
        "p_model": p_cal,
        "p_mkt": p_mkt
    }).with_columns((pl.col("p_model") - pl.col("p_mkt")).alias("__edge"))

    # Filter by odds band + edge threshold
    work = work.filter(
        (pl.col("ltp") >= ltp_min) & (pl.col("ltp") <= ltp_max) & (pl.col("__edge") >= edge_thresh)
    )

    # Dedupe options
    if args.dedupe_mode == "first_cross":
        work = (work.sort("publishTimeMs")
                    .unique(subset=["marketId","selectionId"], keep="first"))
    elif args.dedupe_mode == "max_edge":
        work = (work.sort(["__edge","publishTimeMs"], descending=[True, False])
                    .unique(subset=["marketId","selectionId"], keep="first"))

    if work.is_empty():
        print("[WARN] no rows after filters/dedupe")
        pl.DataFrame({"day":[], "roi":[], "profit":[], "n_trades":[]}).write_csv(args.output_csv)
        return

    # Evaluate per day
    uniq_days, seen = [], set()
    for d in work["__day"].to_list():
        if d not in seen: uniq_days.append(d); seen.add(d)

    rows=[]
    for d in uniq_days:
        df_day = work.filter(pl.col("__day") == d)
        df_valid = df_day.select(["marketId","publishTimeMs","selectionId"])
        odds_day = df_day["ltp"].to_numpy().astype(np.float32)
        y_day = df_day["y"].to_numpy().astype(np.float32)
        p_model_day = df_day["p_model"].to_numpy().astype(np.float32)
        p_mkt_day = df_day["p_mkt"].to_numpy().astype(np.float32)

        met = evaluate(
            df_valid,
            odds_day, y_day, p_model_day, p_mkt_day,
            edge_thresh, 1, ltp_min, ltp_max,
            args.stake_mode, float(args.kelly_cap), float(args.kelly_floor),
            float(bankroll_nom), float(commission)
        )
        rows.append({"day":str(d), "roi":met["roi"], "profit":met["profit"], "n_trades":met["n_trades"]})

    out=pl.DataFrame(rows).sort("day")
    out.write_csv(args.output_csv)
    print(out)
    print(f"Wrote per-day ROI → {args.output_csv}")

if __name__=="__main__":
    main()