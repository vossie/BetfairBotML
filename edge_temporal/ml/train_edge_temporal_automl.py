#!/usr/bin/env python3
"""
AutoML tuner for Edge Temporal with realism controls (GPU train + GPU predict).

Features:
- Pre-off constraint (<= --preoff-max minutes; default 180)
- Validation on the LAST N DAYS (default 7)
- Objective = MEDIAN of DAILY ROI (robust)
- GPU training via XGBoost QuantileDMatrix + GPU predictions via CuPy inplace_predict
- Searches policy (pm_cutoff, edge_thresh, ltp band) + model params
- Realism knobs during validation:
    * --downsample-secs N
    * --dedupe-mode {none,first_cross,max_edge}
    * --haircut-ticks N
    * --stake-mode {flat,kelly} (+ caps)
Saves: trials CSV, best_config.json, model.json, isotonic.pkl
"""
import os, sys, glob, json, pickle, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import numpy as np
import cupy as cp
import xgboost as xgb
import optuna
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression

# ---------------- utils ----------------
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

def list_parquet_between(root: Path, sub: str, start: datetime, end: datetime):
    files = []
    cur = start
    while cur <= end:
        d = cur.strftime("%Y-%m-%d")
        p = root / sub / f"date={d}"
        files.extend(glob.glob(str(p / "*.parquet")))
        cur += timedelta(days=1)
    return files

def _collect_gpu(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="gpu")
    except Exception:
        return lf.collect()

def load_snapshots(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"orderbook_snapshots_5s/sport={sport}", start, end)
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport", "marketId", "selectionId", "publishTimeMs", "ltp",
            "tradedVolume", "spreadTicks", "imbalanceBest1", "ltpTick"]
    names = lf.collect_schema().names()
    cols = [c for c in keep if c in names]
    return _collect_gpu(lf.select(cols))

def load_results(curated: Path, start: datetime, end: datetime, sport: str) -> pl.DataFrame:
    files = list_parquet_between(curated, f"results/sport={sport}", start, end)
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files)
    keep = ["sport", "marketId", "selectionId", "winLabel"]
    names = lf.collect_schema().names()
    cols = [c for c in keep if c in names]
    return _collect_gpu(lf.select(cols))

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
    return _collect_gpu(lf)

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

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    for col in ["marketType_def", "countryCode"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null("UNK"))
            df = df.to_dummies(columns=[col])
    return df

def add_preoff_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "marketStartMs" not in df.columns:
        print("[ERROR] marketStartMs missing after join", file=sys.stderr)
        sys.exit(2)
    df = df.filter(pl.col("marketStartMs").is_not_null())
    return df.with_columns([
        (pl.col("marketStartMs") - pl.col("publishTimeMs")).alias("secs_to_start"),
        ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000).alias("mins_to_start"),
    ])

def numeric_only(df: pl.DataFrame, exclude: set) -> pl.DataFrame:
    df = df.drop([c for c in exclude if c in df.columns])
    drop_ls = [c for c, dt in zip(df.columns, df.dtypes) if isinstance(dt, (pl.List, pl.Struct))]
    if drop_ls:
        df = df.drop(drop_ls)
    keep = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric() or dt == pl.Boolean]
    return df.select(keep)

# Betfair tick utilities
def _tick_size(o: float) -> float:
    if o < 2.0: return 0.01
    if o < 3.0: return 0.02
    if o < 4.0: return 0.05
    if o < 6.0: return 0.10
    if o < 10.0: return 0.20
    if o < 20.0: return 0.50
    if o < 30.0: return 1.00
    if o < 50.0: return 2.00
    if o < 100.0: return 5.00
    return 10.0

def shift_ticks_down(odds: float, n_ticks: int) -> float:
    if n_ticks <= 0: return max(1.01, odds)
    o = odds
    for _ in range(n_ticks):
        step = _tick_size(o)
        o = max(1.01, round(o - step + 1e-12, 2))
    return o

# ---------------- modeling ----------------
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
        "max_bin": 256,
    }

def train_xgb(params, Xtr, ytr, Xva, yva):
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr)
    dva = xgb.QuantileDMatrix(Xva, label=yva)
    return xgb.train(
        params, dtr, num_boost_round=500,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=30, verbose_eval=False
    )

def safe_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    try:
        return log_loss(y_true, y_pred)
    except Exception:
        return float("nan")

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

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--valid-days", type=int, default=7)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preoff-max", type=int, default=180)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--bankroll-nom", type=float, default=5000.0)
    ap.add_argument("--stake-mode", choices=["flat","kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.002)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--dedupe-mode", choices=["none","first_cross","max_edge"], default="none")
    ap.add_argument("--haircut-ticks", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Windows
    train_start, train_end, valid_start, valid_end = compute_windows(args.start_date, args.asof, args.valid_days)
    curated = Path(args.curated)

    # Load once
    snap = load_snapshots(curated, train_start, valid_end, args.sport)
    res  = load_results(curated, train_start, valid_end, args.sport)
    defs = load_defs(curated, train_start, valid_end, args.sport)
    if snap.is_empty() or res.is_empty():
        print("[ERROR] no snapshots or results", file=sys.stderr); sys.exit(2)

    df = join_all(snap, res, defs)
    if df.is_empty():
        print("[ERROR] empty after join", file=sys.stderr); sys.exit(2)
    if "ltp" not in df.columns or "winLabel" not in df.columns:
        print("[ERROR] missing ltp/winLabel", file=sys.stderr); sys.exit(2)

    # Clean + enforce pre-off
    df = (
        df.filter(pl.col("winLabel").is_not_null())
          .with_columns(pl.when(pl.col("winLabel") > 0).then(1).otherwise(0).alias("winLabel"))
          .filter(pl.col("ltp").is_not_null())
    )
    df = add_preoff_columns(df).filter(
        (pl.col("mins_to_start") >= 0) & (pl.col("mins_to_start") <= args.preoff_max)
    )
    df = encode_categoricals(df)

    # Time masks
    train_end_excl = to_ms(train_end + timedelta(days=1))
    valid_end_excl = to_ms(valid_end + timedelta(days=1))
    ts_np = df["publishTimeMs"].to_numpy()
    mask_train_time = (ts_np >= to_ms(train_start)) & (ts_np < train_end_excl)
    mask_valid_time = (ts_np >= to_ms(valid_start)) & (ts_np < valid_end_excl)

    # Features once
    exclude = {"winLabel", "sport", "marketId", "selectionId", "marketStartMs", "secs_to_start"}
    X_all = numeric_only(df, exclude)
    X_np_all = X_all.to_numpy()
    y_all = df["winLabel"].to_numpy().astype(np.float32)

    # Base frame for masking/realism
    base_cols = ["marketId", "publishTimeMs", "selectionId", "ltp", "winLabel"]
    full_base = df.select(base_cols)

    params_base = make_params(args.device)

    # Objective: median of daily ROI on VALID with realism applied
    def objective(trial: optuna.trial.Trial):
        # Policy
        pm_cutoff = trial.suggest_float("pm_cutoff", 0.85, 0.97)
        edge_thr  = trial.suggest_float("edge_thresh", 0.03, 0.15)
        ltp_min   = trial.suggest_float("ltp_min", 1.8, 3.0)
        ltp_max   = trial.suggest_float("ltp_max", max(ltp_min + 0.2, 2.2), 4.0)

        # Model tweaks
        params = params_base.copy()
        params["max_depth"] = trial.suggest_int("max_depth", 4, 8)
        params["eta"] = trial.suggest_float("eta", 0.02, 0.15, log=True)
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 10.0)

        # PM gate
        if "pm_label" in df.columns:
            pm_mask = df["pm_label"].to_numpy() >= pm_cutoff
        else:
            pm_mask = np.ones(df.height, dtype=bool)

        mask_train = mask_train_time & pm_mask
        mask_valid = mask_valid_time & pm_mask
        if not mask_train.any() or not mask_valid.any():
            raise optuna.TrialPruned()

        # Train on TRAIN slice
        Xtr = X_np_all[mask_train].astype(np.float32, copy=False)
        ytr = y_all[mask_train]
        Xva = X_np_all[mask_valid].astype(np.float32, copy=False)
        yva = y_all[mask_valid]

        booster = train_xgb(params, Xtr, ytr, Xva, yva)

        # Build VALID base and apply realism before predictions
        base = (
            full_base
            .with_columns(pl.Series("__mask", mask_valid))
            .filter(pl.col("__mask")).drop("__mask")
            .with_columns(
                pl.from_epoch(pl.col("publishTimeMs"), time_unit="ms")
                  .dt.replace_time_zone("UTC")
                  .dt.date().alias("__vday")
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
            # map to original masked indices to slice Xva/yva
            keys = ["marketId","selectionId","publishTimeMs"]
            masked_df = (full_base.select(keys)
                           .with_columns(pl.Series("__mask", mask_valid))
                           .filter(pl.col("__mask")).drop("__mask"))
            masked_df = masked_df.with_row_index("__idx")
            base = base.join(masked_df, on=keys, how="left")
            idx = base["__idx"].to_numpy()
            good = np.isfinite(idx)
            base = base.filter(pl.Series(good))
            idx = idx[good].astype(np.int64)
            Xva_ds = Xva[idx]
            yva_ds = yva[idx]
        else:
            Xva_ds = Xva
            yva_ds = yva
            base = base.with_row_index("__idx")

        # Predict on GPU, then bring back for isotonic
        p_raw_gpu = booster.inplace_predict(cp.asarray(Xva_ds))
        p_raw = cp.asnumpy(p_raw_gpu).astype(np.float32)

        # OOF isotonic on TRAIN (CPU arrays)
        oof = np.zeros_like(ytr, dtype=np.float32)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, va_idx in kf.split(Xtr):
            b = xgb.train(
                params,
                xgb.QuantileDMatrix(Xtr[tr_idx], label=ytr[tr_idx]),
                num_boost_round=500,
                evals=[
                    (xgb.QuantileDMatrix(Xtr[tr_idx], label=ytr[tr_idx]), "train"),
                    (xgb.QuantileDMatrix(Xtr[va_idx], label=ytr[va_idx]), "valid"),
                ],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            oof_pred_gpu = b.inplace_predict(cp.asarray(Xtr[va_idx]))
            oof[va_idx] = cp.asnumpy(oof_pred_gpu).astype(np.float32)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6).fit(oof, ytr)
        p_val = iso.predict(p_raw).astype(np.float32)

        # Overround-normalized market probs on the CURRENT base
        dv = base.select(["marketId","publishTimeMs","ltp","__vday"]).with_columns(
            (1.0 / pl.col("ltp").clip(lower_bound=1e-12)).alias("__inv")
        )
        sums = dv.group_by(["marketId","publishTimeMs"]).agg(pl.col("__inv").sum().alias("__inv_sum"))
        dv = dv.join(sums, on=["marketId","publishTimeMs"], how="left").with_columns(
            (pl.col("__inv") / pl.col("__inv_sum").clip(lower_bound=1e-12)).alias("__p_mkt_norm")
        )

        p_mkt = dv["__p_mkt_norm"].to_numpy().astype(np.float32)
        odds  = base["ltp"].to_numpy().astype(np.float32)
        if args.haircut_ticks and args.haircut_ticks > 0:
            odds = np.array([shift_ticks_down(float(o), int(args.haircut_ticks)) for o in odds], dtype=np.float32)

        # Build work frame for dedupe + filters
        work = pl.DataFrame({
            "marketId": base["marketId"],
            "selectionId": base["selectionId"],
            "publishTimeMs": base["publishTimeMs"],
            "__vday": base["__vday"],
            "ltp": odds,
            "y": yva_ds,
            "p_model": p_val,
            "p_mkt": p_mkt
        }).with_columns((pl.col("p_model") - pl.col("p_mkt")).alias("__edge"))

        # Filter by odds band + edge threshold
        work = work.filter(
            (pl.col("ltp") >= ltp_min) & (pl.col("ltp") <= ltp_max) & (pl.col("__edge") >= edge_thr)
        )

        # Dedupe options (one bet per runner)
        if args.dedupe_mode == "first_cross":
            work = (work.sort("publishTimeMs")
                        .unique(subset=["marketId","selectionId"], keep="first"))
        elif args.dedupe_mode == "max_edge":
            work = (work.sort(["__edge","publishTimeMs"], descending=[True, False])
                        .unique(subset=["marketId","selectionId"], keep="first"))

        if work.is_empty():
            raise optuna.TrialPruned()

        # Per-day evaluation
        uniq_days, seen = [], set()
        for d in work["__vday"].to_list():
            if d not in seen: uniq_days.append(d); seen.add(d)

        daily_rois, total_profit, total_trades = [], 0.0, 0
        for d in uniq_days:
            df_day = work.filter(pl.col("__vday") == d)
            df_valid = df_day.select(["marketId","publishTimeMs","selectionId"])
            odds_day = df_day["ltp"].to_numpy().astype(np.float32)
            y_day = df_day["y"].to_numpy().astype(np.float32)
            p_model_day = df_day["p_model"].to_numpy().astype(np.float32)
            p_mkt_day = df_day["p_mkt"].to_numpy().astype(np.float32)
            met = evaluate(
                df_valid,
                odds_day, y_day, p_model_day, p_mkt_day,
                edge_thr, 1, ltp_min, ltp_max,
                args.stake_mode, float(args.kelly_cap), float(args.kelly_floor),
                float(args.bankroll_nom), float(args.commission)
            )
            daily_rois.append(met["roi"])
            total_profit += float(met["profit"])
            total_trades += int(met["n_trades"])

        if not daily_rois:
            raise optuna.TrialPruned()

        median_roi = float(np.median(daily_rois))
        avg_roi    = float(np.mean(daily_rois))
        trial.set_user_attr("median_daily_roi", median_roi)
        trial.set_user_attr("avg_daily_roi", avg_roi)
        trial.set_user_attr("min_daily_roi", float(np.min(daily_rois)))
        trial.set_user_attr("max_daily_roi", float(np.max(daily_rois)))
        trial.set_user_attr("days_evaluated", len(daily_rois))
        trial.set_user_attr("total_profit", total_profit)
        trial.set_user_attr("total_trades", total_trades)
        try:
            trial.set_user_attr("logloss", float(safe_logloss(np.asarray(work["y"]), np.asarray(work["p_model"]))))
            trial.set_user_attr("auc", float(roc_auc_score(np.asarray(work["y"]), np.asarray(work["p_model"]))))
        except Exception:
            pass

        return median_roi

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # Summarize trials
    rows = []
    for t in study.trials:
        r = {
            **t.params,
            "score_median_daily_roi": t.user_attrs.get("median_daily_roi", float("-inf")),
            "score_avg_daily_roi":    t.user_attrs.get("avg_daily_roi", float("nan")),
            "min_daily_roi":          t.user_attrs.get("min_daily_roi", float("nan")),
            "max_daily_roi":          t.user_attrs.get("max_daily_roi", float("nan")),
            "days":                   t.user_attrs.get("days_evaluated", 0),
            "total_profit":           t.user_attrs.get("total_profit", 0.0),
            "total_trades":           t.user_attrs.get("total_trades", 0),
            "logloss":                t.user_attrs.get("logloss", float("nan")),
            "auc":                    t.user_attrs.get("auc", float("nan")),
        }
        rows.append(r)
    trials_df = pl.DataFrame(rows).sort(["score_median_daily_roi", "total_profit"], descending=[True, True])
    trials_csv = outdir / f"trials_{args.asof}.csv"
    trials_df.write_csv(str(trials_csv))
    print(trials_df.head(10))
    print(f"Wrote trials → {trials_csv}")

    # Save best config (+ realism knobs)
    best = study.best_trial
    best_cfg = {
        "asof": args.asof,
        "start_date": args.start_date,
        "valid_days": args.valid_days,
        "preoff_max_minutes": args.preoff_max,
        "commission": args.commission,
        "bankroll_nom": args.bankroll_nom,
        "stake_mode": args.stake_mode,
        "kelly_cap": args.kelly_cap,
        "kelly_floor": args.kelly_floor,
        "downsample_secs": args.downsample_secs,
        "dedupe_mode": args.dedupe_mode,
        "haircut_ticks": args.haircut_ticks,
        "best_median_daily_roi": best.user_attrs.get("median_daily_roi"),
        "best_avg_daily_roi": best.user_attrs.get("avg_daily_roi"),
        "best_params": best.params,
    }
    with open(outdir / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    print("Best config →", outdir / "best_config.json")

    # Refit best model on TRAIN slice & save calibrator
    p = best.params
    pm_cutoff = p["pm_cutoff"]
    params = make_params(args.device)
    params.update({
        "max_depth": int(p["max_depth"]),
        "eta": float(p["eta"]),
        "subsample": float(p["subsample"]),
        "colsample_bytree": float(p["colsample_bytree"]),
        "min_child_weight": float(p["min_child_weight"]),
    })

    if "pm_label" in df.columns:
        pm_mask = df["pm_label"].to_numpy() >= pm_cutoff
    else:
        pm_mask = np.ones(df.height, dtype=bool)

    mask_train = mask_train_time & pm_mask
    mask_valid = mask_valid_time & pm_mask

    Xtr = X_np_all[mask_train].astype(np.float32, copy=False)
    ytr = y_all[mask_train]
    Xva = X_np_all[mask_valid].astype(np.float32, copy=False)
    yva = y_all[mask_valid]

    booster = train_xgb(params, Xtr, ytr, Xva, yva)
    booster.save_model(str(outdir / "model.json"))

    # OOF isotonic for persistence
    oof = np.zeros_like(ytr, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(Xtr):
        b = xgb.train(
            params,
            xgb.QuantileDMatrix(Xtr[tr_idx], label=ytr[tr_idx]),
            num_boost_round=500,
            evals=[
                (xgb.QuantileDMatrix(Xtr[tr_idx], label=ytr[tr_idx]), "train"),
                (xgb.QuantileDMatrix(Xtr[va_idx], label=ytr[va_idx]), "valid")
            ],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        oof_pred_gpu = b.inplace_predict(cp.asarray(Xtr[va_idx]))
        oof[va_idx] = cp.asnumpy(oof_pred_gpu).astype(np.float32)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6).fit(oof, ytr)
    with open(outdir / "isotonic.pkl", "wb") as f:
        pickle.dump({"type": "isotonic", "iso": iso}, f)
    print("Saved model.json and isotonic.pkl")

if __name__ == "__main__":
    main()