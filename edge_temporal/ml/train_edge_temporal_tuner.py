#!/usr/bin/env python3
# Fast parameter sweeper for edge temporal:
# - Load/prepare once
# - Train once per (PREOFF_MINS, PM_CUTOFF)
# - Evaluate many (EDGE_THRESH, LTP bands, stake modes) without retraining
#
# Usage example:
#   python3 train_edge_temporal_fastsweep.py \
#     --curated /mnt/nvme/betfair-curated --asof 2025-09-25 --start-date 2025-09-05 \
#     --preoff 5 8 12 --pm-cutoff 0.90 0.92 0.94 \
#     --edge 0.06 0.08 0.10 \
#     --ltp-bands 2.2:2.8 2.4:3.1 2.6:3.2 \
#     --device cuda --output-dir /opt/BetfairBotML/edge_temporal/output/fastsweep
#
# Produces: summary.csv (ranked by ROI, flat staking by default) + per-config logs.

import os, sys, glob, pickle, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression

# ----------------- helpers copied/adapted from your trainer -----------------

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)
def compute_windows(start_date_str: str, asof_str: str, valid_days: int):
    start_date = parse_date(start_date_str); asof = parse_date(asof_str)
    valid_end = asof; valid_start = asof - timedelta(days=valid_days - 1)
    train_start = start_date; train_end = valid_start - timedelta(days=1)
    return train_start, train_end, valid_start, valid_end

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files = []; cur = start
    while cur <= end:
        d = cur.strftime("%Y-%m-%d"); p = root / sub / f"date={d}"
        files.extend(glob.glob(str(p / "*.parquet"))); cur += timedelta(days=1)
    return files

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"orderbook_snapshots_5s/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","ltpTick"]
    names = lf.collect_schema().names(); cols = [c for c in keep if c in names]
    return lf.select(cols).collect()

def load_results(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"results/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport","marketId","selectionId","winLabel"]; names = lf.collect_schema().names()
    cols = [c for c in keep if c in names]; return lf.select(cols).collect()

def load_defs(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"market_definitions/sport={sport}", start, end)
    if not files: return pl.DataFrame()
    lf = pl.scan_parquet(files); names = lf.collect_schema().names()
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

def join_all(snap_df: pl.DataFrame, res_df: pl.DataFrame, defs_df: pl.DataFrame) -> pl.DataFrame:
    if snap_df.is_empty() or res_df.is_empty(): return pl.DataFrame()
    df = snap_df.join(res_df.select(["sport","marketId","selectionId","winLabel"]),
                      on=["sport","marketId","selectionId"], how="inner")
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

def add_preoff_columns(df: pl.DataFrame) -> pl.DataFrame:
    # compute mins_to_start once; later we filter with PREOFF_MINS values cheaply
    if "marketStartMs" not in df.columns:
        raise SystemExit("[ERROR] marketStartMs missing after join")
    df = df.filter(pl.col("marketStartMs").is_not_null())
    return df.with_columns([
        (pl.col("marketStartMs") - pl.col("publishTimeMs")).alias("secs_to_start"),
        ((pl.col("marketStartMs") - pl.col("publishTimeMs"))/60000).alias("mins_to_start"),
    ])

def numeric_only(df: pl.DataFrame, exclude: set) -> pl.DataFrame:
    df = df.drop([c for c in exclude if c in df.columns])
    drop_ls = [c for c, dt in zip(df.columns, df.dtypes) if isinstance(dt, (pl.List, pl.Struct))]
    if drop_ls: df = df.drop(drop_ls)
    keep = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric() or dt == pl.Boolean]
    return df.select(keep)

def make_params(device="cuda"):
    return {"objective":"binary:logistic","eval_metric":["logloss","auc"],"tree_method":"hist",
            "device":"cuda" if device!="cpu" else "cpu","max_depth":6,"eta":0.05,
            "subsample":0.8,"colsample_bytree":0.8}

def train_xgb(params, Xtr, ytr, Xva, yva):
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    return xgb.train(params, dtr, num_boost_round=500,
                     evals=[(dtr,"train"),(dva,"valid")],
                     early_stopping_rounds=30, verbose_eval=False)

def safe_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    try: return log_loss(y_true, y_pred)
    except Exception: return float("nan")

def evaluate(df_valid, odds, y, p_model, p_market_norm, edge_thresh, topk, lo, hi,
             stake_mode, cap, floor_, bank, commission):
    edge = p_model - p_market_norm
    mask = (odds >= lo) & (odds <= hi) & np.isfinite(edge)
    if not mask.any():
        return dict(roi=0.0, profit=0.0, n_trades=0)
    import polars as pl
    df = pl.DataFrame({
        "marketId": df_valid["marketId"].to_numpy()[mask],
        "publishTimeMs": df_valid["publishTimeMs"].to_numpy()[mask],
        "selectionId": df_valid["selectionId"].to_numpy()[mask],
        "ltp": odds[mask],
        "edge": edge[mask],
        "y": y[mask],
        "p_model": p_model[mask],
    }).filter(pl.col("edge") >= edge_thresh)
    if df.height == 0:
        return dict(roi=0.0, profit=0.0, n_trades=0)
    df = df.with_columns(
        pl.col("edge").rank(method="dense", descending=True).over(["marketId","publishTimeMs"]).alias("rk")
    ).filter(pl.col("rk") <= topk).drop("rk")

    outcomes = df["y"].to_numpy().astype(np.float32)
    odds_sel = df["ltp"].to_numpy().astype(np.float32)
    p_model_sel = df["p_model"].to_numpy().astype(np.float32)

    if stake_mode == "kelly":
        def kf(p, o):
            b = o - 1.0
            if b <= 0: return 0.0
            q = 1.0 - p
            return max(0.0, (b*p - q)/b)
        f = np.array([max(floor_, min(cap, kf(pi, oi))) for pi, oi in zip(p_model_sel, odds_sel)], dtype=np.float32)
        stakes = f * bank
    else:
        stakes = np.full_like(odds_sel, 10.0, dtype=np.float32)

    gross = outcomes * (odds_sel - 1.0) * stakes
    net = gross * (1.0 - commission)
    loss = (1.0 - outcomes) * stakes
    profit = net - loss
    pnl = float(profit.sum())
    staked = float(stakes.sum())
    roi = pnl / staked if staked > 0 else 0.0
    return dict(roi=roi, profit=pnl, n_trades=int(outcomes.size))

# ----------------- fast sweep -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--valid-days", type=int, default=7)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--bankroll-nom", type=float, default=5000.0)
    ap.add_argument("--kelly-cap", type=float, default=0.01)  # cooler default
    ap.add_argument("--kelly-floor", type=float, default=0.0)
    ap.add_argument("--preoff", nargs="+", type=int, required=True)
    ap.add_argument("--pm-cutoff", nargs="+", type=float, required=True)
    ap.add_argument("--edge", nargs="+", type=float, required=True)
    ap.add_argument("--ltp-bands", nargs="+", type=str, required=True, help="Format: a:b a:b ...")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank-stake", default="flat", choices=["flat","kelly"])
    args = ap.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) LOAD ONCE
    train_start, train_end, valid_start, valid_end = compute_windows(args.start_date, args.asof, args.valid_days)
    curated = Path(args.curated)

    snap_df = load_snapshots(curated, train_start, valid_end, args.sport)
    res_df  = load_results (curated, train_start, valid_end, args.sport)
    defs_df = load_defs    (curated, train_start, valid_end, args.sport)
    if snap_df.is_empty() or res_df.is_empty():
        print("[ERROR] no snapshots or results loaded"); sys.exit(2)

    df = join_all(snap_df, res_df, defs_df)
    if df.is_empty(): print("[ERROR] empty after join"); sys.exit(2)
    df = (
        df.filter(pl.col("winLabel").is_not_null())
          .with_columns(pl.when(pl.col("winLabel")>0).then(1).otherwise(0).alias("winLabel"))
          .filter(pl.col("ltp").is_not_null())
    )
    # compute mins_to_start once
    df = add_preoff_columns(df)
    # one-hot once
    df = encode_categoricals(df)

    # split indices once
    train_end_excl = to_ms(train_end + timedelta(days=1))
    valid_end_excl = to_ms(valid_end + timedelta(days=1))
    mask_train_time = (df["publishTimeMs"] >= to_ms(train_start)) & (df["publishTimeMs"] < train_end_excl)
    mask_valid_time = (df["publishTimeMs"] >= to_ms(valid_start)) & (df["publishTimeMs"] < valid_end_excl)

    # Prepare feature view (numeric only), but we still filter rows per config later
    exclude = {"winLabel","sport","marketId","selectionId","marketStartMs","secs_to_start"}
    X_all = numeric_only(df, exclude)
    y_all = df["winLabel"].to_numpy().astype(np.float32)

    # For p_market_norm we’ll compute per VALID subset
    base_valid_for_mkt = df.select(["marketId","publishTimeMs","ltp"])

    # parse bands
    bands = []
    for s in args.ltp_bands:
        a,b = s.split(":"); bands.append((float(a), float(b)))

    # 2) for each (preoff, pm_cut), TRAIN ONCE, predict VALID ONCE
    results = []
    params = make_params(args.device)

    for pre in args.preoff:
        # cheap row filter by mins_to_start
        pre_mask = (df["mins_to_start"].to_numpy() >= 0) & (df["mins_to_start"].to_numpy() <= pre)

        for pmc in args.pm_cutoff:
            # optional PM gate if column exists
            if "pm_label" in df.columns:
                pm_mask = df["pm_label"].to_numpy() >= pmc
            else:
                pm_mask = np.ones(df.height, dtype=bool)

            mask_train = mask_train_time & pre_mask & pm_mask
            mask_valid = mask_valid_time & pre_mask & pm_mask

            if not mask_train.any() or not mask_valid.any():
                print(f"[WARN] empty train/valid for PREOFF={pre}, PM_CUTOFF={pmc} → skipping")
                continue

            Xtr = X_all.to_numpy()[mask_train].astype(np.float32, copy=False)
            ytr = y_all[mask_train]
            Xva = X_all.to_numpy()[mask_valid].astype(np.float32, copy=False)
            yva = y_all[mask_valid]

            booster = train_xgb(params, Xtr, ytr, Xva, yva)
            p_raw = booster.predict(xgb.DMatrix(Xva)).astype(np.float32)

            # OOF isotonic on TRAIN (fit ONCE per (pre, pmc))
            Xtr_np = Xtr  # already np
            oof = np.zeros_like(ytr, dtype=np.float32)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr, va in kf.split(Xtr_np):
                b = xgb.train(params, xgb.DMatrix(Xtr_np[tr], label=ytr[tr]),
                              num_boost_round=500,
                              evals=[(xgb.DMatrix(Xtr_np[tr], label=ytr[tr]),"train"),
                                     (xgb.DMatrix(Xtr_np[va], label=ytr[va]),"valid")],
                              early_stopping_rounds=30, verbose_eval=False)
                oof[va] = b.predict(xgb.DMatrix(Xtr_np[va])).astype(np.float32)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6).fit(oof, ytr)
            p_val = iso.predict(p_raw).astype(np.float32)

            odds = df["ltp"].to_numpy().astype(np.float32)[mask_valid]

            # market prob normalization on this VALID subset
            dv = base_valid_for_mkt.filter(mask_valid.tolist()).with_columns(
                (1.0 / pl.col("ltp").clip(lower_bound=1e-12)).alias("__inv")
            )
            sums = dv.group_by(["marketId","publishTimeMs"]).agg(pl.col("__inv").sum().alias("__inv_sum"))
            dv = dv.join(sums, on=["marketId","publishTimeMs"], how="left").with_columns(
                (pl.col("__inv")/pl.col("__inv_sum").clip(lower_bound=1e-12)).alias("__p_mkt_norm")
            )
            p_mkt = dv["__p_mkt_norm"].to_numpy().astype(np.float32)

            # quick quality print
            print(f"[fit] PREOFF={pre:>2}  PM_CUTOFF={pmc:.2f}  "
                  f"logloss={safe_logloss(yva,p_val):.4f}  auc={roc_auc_score(yva,p_val):.3f}  n_valid={yva.size}")

            # 3) evaluate many cheap knobs without retraining
            for (lo, hi) in bands:
                for edge in args.edge:
                    for stake_mode in ("flat","kelly"):
                        cap = args.kelly_cap if stake_mode=="kelly" else 0.0
                        floor_ = args.kelly_floor if stake_mode=="kelly" else 0.0
                        bank = args.bankroll_nom
                        met = evaluate(
                            df.filter(mask_valid.tolist()),
                            odds, yva, p_val, p_mkt,
                            edge, 1, lo, hi,
                            stake_mode, cap, floor_, bank, float(args.commission)
                        )
                        results.append({
                            "roi": met["roi"], "profit": met["profit"], "n_trades": met["n_trades"],
                            "stake_mode": (stake_mode if stake_mode=="flat" else f"kelly_cap{cap}_floor{floor_}"),
                            "pm_cutoff": pmc, "preoff_mins": pre, "edge_thresh": edge,
                            "ltp_min": lo, "ltp_max": hi
                        })

    if not results:
        print("No results produced."); return

    # rank
    rank_mode = args.rank_stake
    ranked = [r for r in results if (rank_mode=="flat" and r["stake_mode"]=="flat") or rank_mode=="kelly"]
    ranked.sort(key=lambda r: (r["roi"], r["profit"], -r["n_trades"]), reverse=True)

    df_out = pl.DataFrame(ranked)
    out_csv = Path(args.output_dir) / f"summary_{args.asof}.csv"
    df_out.write_csv(str(out_csv))
    print(f"\nWrote summary → {out_csv}")
    print(df_out.head(10))

if __name__ == "__main__":
    main()
