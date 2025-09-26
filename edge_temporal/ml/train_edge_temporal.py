#!/usr/bin/env python3
# Drop-in: trains XGBoost on real data and evaluates with per-market overround-normalized market probabilities.
# - Polars 1.33 compatible
# - CUDA by default (XGBoost: device="cuda", tree_method="hist")
# - Env-driven filters: PREOFF_MINS, PM_CUTOFF, EDGE_THRESH, PER_MARKET_TOPK, LTP_MIN, LTP_MAX
# - Optional isotonic calibration: --fit-calib / CALIB_PATH
# - Outputs sweep (flat & kelly) and per-day ROI tables

import os, sys, glob, pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression


# ----------------- utilities -----------------

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def compute_windows(start_date_str: str, asof_str: str, valid_days: int):
    start_date = parse_date(start_date_str)
    asof = parse_date(asof_str)
    valid_end = asof
    valid_start = asof - timedelta(days=valid_days - 1)
    train_start = start_date
    train_end = valid_start - timedelta(days=1)
    return train_start, train_end, valid_start, valid_end

def epoch_date_expr(colname: str) -> pl.Expr:
    # Works for ms or s
    return (
        pl.when(pl.col(colname) >= 100_000_000_000)
        .then(pl.from_epoch(pl.col(colname), time_unit="ms"))
        .otherwise(pl.from_epoch(pl.col(colname), time_unit="s"))
    ).dt.replace_time_zone("UTC").dt.date()

def die(msg, code=2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def safe_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    try:
        return log_loss(y_true, y_pred)
    except Exception:
        return float("nan")


# ----------------- data loading -----------------

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files = []
    cur = start
    while cur <= end:
        d = cur.strftime("%Y-%m-%d")
        p = root / sub / f"date={d}"
        files.extend(glob.glob(str(p / "*.parquet")))
        cur += timedelta(days=1)
    return files

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"orderbook_snapshots_5s/sport={sport}", start, end)
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = [
        "sport", "marketId", "selectionId", "publishTimeMs",
        "ltp", "tradedVolume", "spreadTicks", "imbalanceBest1", "ltpTick"
    ]
    names = lf.collect_schema().names()
    cols = [c for c in keep if c in names]
    return lf.select(cols).collect()

def load_results(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"results/sport={sport}", start, end)
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport", "marketId", "selectionId", "winLabel"]
    names = lf.collect_schema().names()
    cols = [c for c in keep if c in names]
    return lf.select(cols).collect()

def load_defs(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"market_definitions/sport={sport}", start, end)
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files)
    names = lf.collect_schema().names()
    if "runners" in names:
        lf = lf.explode("runners").select([
            "sport", "marketId",
            pl.col("runners").struct.field("selectionId").alias("selectionId"),
            pl.col("marketStartMs"),
            pl.col("marketType").alias("marketType_def"),
            pl.col("countryCode"),
            pl.col("runners").struct.field("handicap"),
            pl.col("runners").struct.field("sortPriority"),
            pl.col("runners").struct.field("reductionFactor"),
        ])
    else:
        have = [c for c in ["sport", "marketId", "marketStartMs", "marketType", "countryCode"] if c in names]
        lf = lf.select(have)
    return lf.collect()

def join_all(snap_df: pl.DataFrame, res_df: pl.DataFrame, defs_df: pl.DataFrame) -> pl.DataFrame:
    if snap_df.is_empty() or res_df.is_empty():
        return pl.DataFrame()
    df = snap_df.join(
        res_df.select(["sport", "marketId", "selectionId", "winLabel"]),
        on=["sport", "marketId", "selectionId"],
        how="inner"
    )
    if not defs_df.is_empty():
        on_cols = [c for c in ["sport", "marketId", "selectionId"] if c in defs_df.columns and c in df.columns]
        df = df.join(defs_df, on=on_cols, how="left")
    return df


# ----------------- feature prep -----------------

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    for col in ["marketType_def", "countryCode"]:
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
        ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000).alias("mins_to_start"),
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
    if drop_ls:
        df = df.drop(drop_ls)
    keep = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric() or dt == pl.Boolean]
    return df.select(keep)


# ----------------- modeling -----------------

def make_params(device="cuda"):
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
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
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

def evaluate(df_valid, odds, y, p_model, p_market_norm, edge_thresh, topk, lo, hi,
             stake_mode, cap, floor_, bank, commission):
    edge = p_model - p_market_norm
    mask = (odds >= lo) & (odds <= hi) & np.isfinite(edge)
    if not mask.any():
        return dict(roi=0.0, profit=0.0, n_trades=0)
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
        pl.col("edge").rank(method="dense", descending=True).over(["marketId", "publishTimeMs"]).alias("rk")
    ).filter(pl.col("rk") <= topk).drop("rk")

    outcomes = df["y"].to_numpy().astype(np.float32)
    odds_sel = df["ltp"].to_numpy().astype(np.float32)
    p_model_sel = df["p_model"].to_numpy().astype(np.float32)

    if stake_mode == "kelly":
        def kf(p, o):
            b = o - 1.0
            if b <= 0:
                return 0.0
            q = 1.0 - p
            return max(0.0, (b * p - q) / b)
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


# ----------------- CLI -----------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--device", default="cuda")
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.05)
    p.add_argument("--kelly-floor", type=float, default=0.002)
    # calibration options
    p.add_argument("--fit-calib", action="store_true", help="Fit OOF isotonic on TRAIN and apply to VALID; optionally save with --calib-out.")
    p.add_argument("--calib-out", default="", help="Path to save fitted calibrator pickle.")
    return p.parse_args()


# ----------------- main -----------------

def main():
    args = parse_args()
    outdir = Path(os.environ.get("OUTPUT_DIR", "/opt/BetfairBotML/edge_temporal/output"))
    outdir.mkdir(parents=True, exist_ok=True)

    # env filters
    PREOFF_MINS      = int(os.environ.get("PREOFF_MINS", "30"))
    PM_CUTOFF        = float(os.environ.get("PM_CUTOFF", "0.65"))
    EDGE_THRESH      = float(os.environ.get("EDGE_THRESH", "0.015"))
    PER_MARKET_TOPK  = int(os.environ.get("PER_MARKET_TOPK", "1"))
    LTP_MIN          = float(os.environ.get("LTP_MIN", "1.5"))
    LTP_MAX          = float(os.environ.get("LTP_MAX", "5.0"))

    train_start, train_end, valid_start, valid_end = compute_windows(args.start_date, args.asof, args.valid_days)
    print(f"Training window:   {train_start.date()} .. {train_end.date()}")
    print(f"Validation window: {valid_start.date()} .. {valid_end.date()}")

    curated = Path(args.curated)
    # load
    snap_df = load_snapshots(curated, train_start, valid_end, args.sport)
    res_df  = load_results(curated,  train_start, valid_end, args.sport)
    defs_df = load_defs(curated,     train_start, valid_end, args.sport)
    if snap_df.is_empty() or res_df.is_empty():
        die("no snapshots or results loaded")

    df_all = join_all(snap_df, res_df, defs_df)
    if df_all.is_empty(): die("empty after join")
    if "winLabel" not in df_all.columns: die("winLabel missing")
    if "ltp" not in df_all.columns: die("ltp missing")

    # label clean + filters
    df_all = (
        df_all
        .filter(pl.col("winLabel").is_not_null())
        .with_columns(pl.when(pl.col("winLabel") > 0).then(1).otherwise(0).alias("winLabel"))
        .filter(pl.col("ltp").is_not_null())
    )
    df_all = add_preoff_filter(df_all, PREOFF_MINS)
    df_all = apply_pm_gate(df_all, PM_CUTOFF)
    df_all = encode_categoricals(df_all)

    # split by publishTimeMs
    train_end_excl = to_ms(train_end + timedelta(days=1))
    valid_end_excl = to_ms(valid_end + timedelta(days=1))
    df_train = df_all.filter((pl.col("publishTimeMs") >= to_ms(train_start)) & (pl.col("publishTimeMs") < train_end_excl))
    df_valid = df_all.filter((pl.col("publishTimeMs") >= to_ms(valid_start)) & (pl.col("publishTimeMs") < valid_end_excl))
    if df_train.is_empty() or df_valid.is_empty():
        die("empty train/valid")

    exclude = {"winLabel", "sport", "marketId", "selectionId", "marketStartMs", "secs_to_start"}
    X_train = numeric_only(df_train, exclude)
    X_valid = numeric_only(df_valid, exclude)
    y_train = df_train["winLabel"].to_numpy().astype(np.float32)
    y_valid = df_valid["winLabel"].to_numpy().astype(np.float32)

    print(f"[rows] train={df_train.height:,}  valid={df_valid.height:,}  features={X_train.width}")

    # xgboost
    params = make_params(args.device)
    dtrain = xgb.DMatrix(X_train.to_arrow(), label=y_train)
    dvalid = xgb.DMatrix(X_valid.to_arrow(),  label=y_valid)
    booster = train(params, dtrain, dvalid)

    # raw predictions on VALID
    p = booster.predict(dvalid).astype(np.float32)

    # optional: fit calibrator on TRAIN (OOF) and apply to VALID
    iso = None
    if args.fit_calib:
        X_np = X_train.to_numpy()
        oof = np.zeros_like(y_train, dtype=np.float32)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in kf.split(X_np):
            bst = xgb.train(
                params,
                xgb.DMatrix(X_np[tr], label=y_train[tr]),
                num_boost_round=500,
                evals=[(xgb.DMatrix(X_np[tr], label=y_train[tr]), "train"),
                       (xgb.DMatrix(X_np[va], label=y_train[va]), "valid")],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            oof[va] = bst.predict(xgb.DMatrix(X_np[va])).astype(np.float32)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
        iso.fit(oof, y_train)
        if args.calib_out:
            Path(args.calib_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.calib_out, "wb") as f:
                pickle.dump({"type": "isotonic", "iso": iso}, f)
            print(f"[calib] saved → {args.calib_out}")

    # or load from CALIB_PATH
    if iso is None:
        calib_path = os.environ.get("CALIB_PATH", "").strip()
        if calib_path and Path(calib_path).exists():
            with open(calib_path, "rb") as f:
                obj = pickle.load(f)
            if obj.get("type") == "isotonic":
                iso = obj["iso"]

    if iso is not None:
        p = iso.predict(p).astype(np.float32)

    odds = df_valid["ltp"].to_numpy().astype(np.float32)

    # ----------- MARKET PROBABILITY: OVERROUND-NORMALIZED -----------
    # Compute p_market_norm per (marketId, publishTimeMs) so probs sum to 1 within each book snapshot.
    dv_mkt = (
        df_valid
        .select(["marketId", "publishTimeMs", "ltp"])
        .with_columns((1.0 / pl.col("ltp").clip_min(1e-12)).alias("__inv"))
    )
    sums = dv_mkt.group_by(["marketId", "publishTimeMs"]).agg(pl.col("__inv").sum().alias("__inv_sum"))
    dv_mkt = dv_mkt.join(sums, on=["marketId", "publishTimeMs"], how="left").with_columns(
        (pl.col("__inv") / pl.col("__inv_sum").clip_min(1e-12)).alias("__p_mkt_norm")
    )
    p_market_norm = dv_mkt["__p_mkt_norm"].to_numpy().astype(np.float32)

    print(f"[Value] logloss={safe_logloss(y_valid, p):.4f}  auc={roc_auc_score(y_valid, p):.3f}")

    # evaluate flat & kelly with env filters and normalized market probs
    configs = [
        ("flat",  None, None, args.bankroll_nom),
        ("kelly", args.kelly_cap, args.kelly_floor, args.bankroll_nom),
    ]
    recs = []
    for (sm, cap, floor_, bank) in configs:
        m = evaluate(
            df_valid, odds, y_valid, p, p_market_norm,
            EDGE_THRESH, PER_MARKET_TOPK, LTP_MIN, LTP_MAX,
            sm, cap or 0.0, floor_ or 0.0, bank or 1000.0, float(args.commission)
        )
        m.update(dict(
            edge_thresh=EDGE_THRESH, topk=PER_MARKET_TOPK,
            ltp_min=LTP_MIN, ltp_max=LTP_MAX,
            stake_mode=("flat" if sm == "flat" else f"kelly_cap{cap}_floor{floor_}")
        ))
        recs.append(m)

    sweep = pl.DataFrame(recs).sort(["roi", "n_trades"], descending=[True, True])
    out = outdir / f"edge_sweep_{args.asof}.csv"
    sweep.write_csv(str(out))
    print(f"sweep saved → {out}")
    print(sweep)

    # per-day ROI with same normalized p_market
    df_valid = df_valid.with_columns(epoch_date_expr("publishTimeMs").alias("__vday"))
    days = df_valid.select("__vday").unique().to_series().to_list()
    daily_rows = []
    for d in sorted(days):
        dv = df_valid.filter(pl.col("__vday") == d)
        if dv.is_empty():
            continue
        Xv = numeric_only(dv, exclude)
        if Xv.is_empty():
            continue
        yv = dv["winLabel"].to_numpy().astype(np.float32)
        pv = booster.predict(xgb.DMatrix(Xv.to_arrow())).astype(np.float32)
        if iso is not None:
            pv = iso.predict(pv).astype(np.float32)
        ov = dv["ltp"].to_numpy().astype(np.float32)

        # daily normalized market probs
        dm = (
            dv.select(["marketId", "publishTimeMs", "ltp"])
              .with_columns((1.0 / pl.col("ltp").clip_min(1e-12)).alias("__inv"))
        )
        ds = dm.group_by(["marketId", "publishTimeMs"]).agg(pl.col("__inv").sum().alias("__inv_sum"))
        dm = dm.join(ds, on=["marketId", "publishTimeMs"], how="left").with_columns(
            (pl.col("__inv") / pl.col("__inv_sum").clip_min(1e-12)).alias("__p_mkt_norm")
        )
        pm = dm["__p_mkt_norm"].to_numpy().astype(np.float32)

        for (sm, cap, floor_, bank) in configs:
            met = evaluate(
                dv, ov, yv, pv, pm, EDGE_THRESH, PER_MARKET_TOPK, LTP_MIN, LTP_MAX,
                sm, cap or 0.0, floor_ or 0.0, bank or 1000.0, float(args.commission)
            )
            daily_rows.append({
                "day": str(d),
                "stake_mode": ("flat" if sm == "flat" else f"kelly_cap{cap}_floor{floor_}"),
                "roi": met["roi"],
                "profit": met["profit"],
                "n_trades": met["n_trades"],
            })

    if daily_rows:
        daily_df = pl.DataFrame(daily_rows).sort(["day", "stake_mode"])
        daily_out = outdir / f"edge_daily_{args.asof}.csv"
        daily_df.write_csv(str(daily_out))
        print(f"daily saved → {daily_out}")
        print(daily_df)


if __name__ == "__main__":
    main()
