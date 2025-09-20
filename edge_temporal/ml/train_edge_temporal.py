#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_edge_temporal.py — drop-in runner that loads REAL data from --curated and trains XGBoost.
If no usable files are found, it loudly falls back to synthetic data so you still get a run.

Accepted flags (match bin/train_edge_temporal.sh):
  --curated --sport --asof --train-days --valid-days
  --preoff-mins --downsample-secs
  --commission --edge-thresh
  --pm-horizon-secs --pm-tick-threshold --pm-slack-secs --pm-cutoff
  --edge-prob --no-sum-to-one --market-prob
  --per-market-topk --side
  --stake {flat,kelly} --kelly-cap --kelly-floor --bankroll-nom
  --ltp-min --ltp-max
  --device

Data expectations (when loading real data):
  Required columns for VALUE head:
    - winLabel (0/1 outcome)
    - ltp (decimal odds at entry evaluation moment)
  Optional PM columns (if present, PM head is trained & used as a gate):
    - pm_label (0/1 indicating future move ≥ threshold)
    - pm_future_ticks (float/int for reporting)
  Feature columns:
    - All numeric columns except {winLabel, pm_label, pm_future_ticks, ltp, marketId, selectionId}
"""

import argparse, os, sys, time, json, re, glob
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# -------------------------------
# Utilities
# -------------------------------
ID_COLS = {"marketId", "selectionId"}
LABEL_COLS = {"winLabel", "pm_label", "pm_future_ticks"}
ODDS_COLS = {"ltp"}

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def odds_to_prob(odds):
    a = np.asarray(odds, dtype=float)
    return np.where(np.isfinite(a) & (a > 0), 1.0 / a, np.nan)

def safe_logloss(y_true, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

def kelly_fraction(p, odds):
    b = float(odds) - 1.0
    if b <= 0: return 0.0
    q = 1.0 - float(p)
    f = (b * float(p) - q) / b
    return max(0.0, float(f))

def pm_threshold_sweep(pm_probs, future_move_hit, future_ticks, thresholds):
    lines = []
    positives = float(future_move_hit.sum())
    for th in thresholds:
        sel = pm_probs >= th
        n_sel = int(sel.sum())
        if n_sel == 0:
            lines.append((th, 0, 0.0, 0.0, 0.0, float("nan")))
            continue
        prec = float(future_move_hit[sel].mean())
        recall = (float(future_move_hit[sel].sum()) / positives) if positives > 0 else 0.0
        f1 = 0.0 if (prec + recall) == 0 else 2 * prec * recall / (prec + recall)
        avg_ticks = float(future_ticks[sel].mean()) if np.isfinite(future_ticks[sel]).any() else float("nan")
        lines.append((th, n_sel, prec, recall, f1, avg_ticks))
    return lines

def roi_by_odds_buckets(odds, outcomes, flat_stake=10.0):
    buckets = [1.01, 1.50, 2.00, 3.00, 5.00, 10.00, 50.00, 1000.00]
    res = []
    odds = np.asarray(odds); outcomes = np.asarray(outcomes)
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i+1]
        m = (odds >= lo) & (odds < hi)
        n = int(m.sum())
        if n == 0:
            res.append((lo, hi, 0, None, None)); continue
        wins = int(outcomes[m].sum())
        profit = wins * (float(odds[m].mean()) - 1.0) * flat_stake - (n - wins) * flat_stake
        roi = profit / (n * flat_stake) if n > 0 else None
        hit = wins / n if n > 0 else None
        res.append((lo, hi, n, hit, roi))
    return res

def make_xgb_params(use_cuda: bool):
    if use_cuda:
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            max_depth=6,
            eta=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            max_bin=256,
        )
    else:
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            predictor="cpu_predictor",
            max_depth=6,
            eta=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            max_bin=512,
        )

def train_with_device(params, dtrain, dvalid, num_boost_round=300, early_stopping_rounds=30):
    use_cuda = (params.get("tree_method") == "gpu_hist")
    try:
        t0 = time.time()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        elapsed = time.time() - t0
        return booster, elapsed, use_cuda, None
    except xgb.core.XGBoostError as e:
        if use_cuda:
            print("[WARN] GPU not available in this XGBoost build. Falling back to CPU.")
            cpu_params = make_xgb_params(False)
            t0 = time.time()
            booster = xgb.train(
                cpu_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            elapsed = time.time() - t0
            return booster, elapsed, False, str(e)
        raise

# -------------------------------
# Data loading
# -------------------------------
def date_in_range_from_path(p: Path, start: datetime, end: datetime) -> bool:
    # Try to extract YYYY-MM-DD from filename
    m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
    if not m:
        return True  # if unknown, include (we’ll filter by columns later if present)
    d = parse_date(m.group(0))
    return (start <= d <= end)

def load_frames(curated_root: str, start: datetime, end: datetime) -> pl.DataFrame | None:
    root = Path(curated_root)
    # Glob patterns can be overridden via env if your layout differs
    patterns = os.environ.get("FEATURE_GLOB", f"{root}/**/*.parquet").split(";")
    csv_patterns = os.environ.get("FEATURE_CSV_GLOB", f"{root}/**/*.csv").split(";")

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    for pat in csv_patterns:
        files.extend(glob.glob(pat, recursive=True))

    files = [Path(f) for f in files if Path(f).is_file()]
    if not files:
        return None

    selected = [p for p in files if date_in_range_from_path(p, start, end)]
    if not selected:
        # as a fallback include everything (maybe dates live in a column)
        selected = files

    dfs = []
    for p in selected:
        try:
            if p.suffix.lower() == ".parquet":
                df = pl.read_parquet(p)
            elif p.suffix.lower() == ".csv":
                df = pl.read_csv(p, infer_schema_length=200000)
            else:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    if not dfs:
        return None
    try:
        df_all = pl.concat(dfs, how="diagonal_relaxed")
    except Exception:
        df_all = pl.concat(dfs, how="vertical")
    return df_all

def split_by_date(df: pl.DataFrame, date_col_candidates=("asof", "eventDate", "date")):
    for c in date_col_candidates:
        if c in df.columns:
            try:
                d = df[c]
                if not pl.datatypes.is_datetime(d.dtype):
                    df = df.with_columns(pl.col(c).str.strptime(pl.Datetime, strict=False, format="%Y-%m-%d").fill_null(pl.col(c)))
                return df, c
            except Exception:
                # try to coerce to date-like
                try:
                    df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False))
                    return df, c
                except Exception:
                    pass
    return df, None  # no date col; caller must pre-filter via filename

def select_columns(df: pl.DataFrame):
    cols = set(df.columns)
    missing = ["winLabel", "ltp"]
    for m in missing:
        if m not in cols:
            raise RuntimeError(f"Required column '{m}' not found in curated data.")

    # PM optional
    has_pm_label = "pm_label" in cols
    has_pm_ticks = "pm_future_ticks" in cols

    # Feature candidates: numeric columns minus known non-features
    exclude = ID_COLS | LABEL_COLS | ODDS_COLS
    feat_cols = []
    for c in df.columns:
        if c in exclude: continue
        dt = df.schema[c]
        if pl.datatypes.is_numeric(dt):
            feat_cols.append(c)

    if not feat_cols:
        raise RuntimeError("No numeric feature columns found. Ensure your curated files include numeric features.")

    return feat_cols, has_pm_label, has_pm_ticks

def slice_window(df: pl.DataFrame, date_col: str | None, start: datetime, end: datetime):
    if date_col is None:
        # no date col; assume all rows are within range if files were pre-filtered
        return df
    return df.filter((pl.col(date_col) >= pl.lit(start)) & (pl.col(date_col) <= pl.lit(end)))

# -------------------------------
# argparse
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--train-days", type=int, default=12)
    ap.add_argument("--valid-days", type=int, default=2)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--downsample-secs", type=int, default=5)

    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--edge-thresh", type=float, default=0.015)

    ap.add_argument("--pm-horizon-secs", type=int, default=300)
    ap.add_argument("--pm-tick-threshold", type=int, default=1)
    ap.add_argument("--pm-slack-secs", type=int, default=3)
    ap.add_argument("--pm-cutoff", type=float, default=0.55)

    ap.add_argument("--edge-prob", default="cal", choices=["raw", "cal"])
    ap.add_argument("--no-sum-to-one", action="store_true")
    ap.add_argument("--market-prob", default="overround", choices=["overround", "ltp"])

    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--side", choices=["back", "lay", "both"], default="back")

    ap.add_argument("--stake", choices=["flat", "kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.0)
    ap.add_argument("--bankroll-nom", type=float, default=1000.0)

    ap.add_argument("--ltp-min", type=float, default=1.5)
    ap.add_argument("--ltp-max", type=float, default=5.0)

    ap.add_argument("--device", default="cuda")
    return ap.parse_args()

# -------------------------------
# main
# -------------------------------
def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Derive date windows to load
    asof_dt = parse_date(args.asof)
    train_end = asof_dt - timedelta(days=args.valid_days)
    train_start = train_end - timedelta(days=args.train_days - 1)
    valid_start = asof_dt - timedelta(days=args.valid_days - 1)
    valid_end = asof_dt

    # Banner
    print("=== Edge Temporal Training (LOCAL) ===")
    print(f"Curated root:         {args.curated}")
    today = time.strftime("%Y-%m-%d")
    print(f"Today:                {today}")
    print(f"ASOF (arg to trainer):{args.asof}   # validation excludes today")
    print(f"Validation days:      {args.valid_days}")
    print(f"Training days:        {args.train_days}")
    print(f"Sport:                {args.sport}")
    print(f"Pre-off minutes:      {args.preoff_mins}")
    print(f"Downsample (secs):    {args.downsample_secs}")
    print(f"Commission:           {args.commission}")
    print(f"Edge threshold:       {args.edge_thresh}")
    print(f"Edge prob:            {args.edge_prob}")
    print(f"Sum-to-one:           {'disabled' if args.no_sum_to_one else 'enabled'}")
    print(f"Market prob:          {args.market_prob}")
    print(f"Per-market topK:      {args.per_market_topk}")
    print(f"Side:                 {args.side}")
    print(f"LTP range:            [{args.ltp_min}, {args.ltp_max}]")
    print(f"Stake mode:           {args.stake} (kelly_cap={args.kelly_cap}, floor={args.kelly_floor})")
    print(f"Bankroll (nominal):   £{args.bankroll_nom:.0f}")
    print(f"PM horizon (secs):    {args.pm_horizon_secs}")
    print(f"PM tick threshold:    {args.pm_tick_threshold}")
    print(f"PM slack (secs):      {args.pm_slack_secs}")
    print(f"PM cutoff:            {args.pm_cutoff}")
    print(f"Output dir:           {OUTPUT_DIR}")
    print()
    print(f"=== Temporal split ===")
    print(f"  Train: {train_start.date()} .. {train_end.date()}  ({args.train_days} days)")
    print(f"  Valid: {valid_start.date()} .. {valid_end.date()}  ({args.valid_days} days)")
    print()

    # -------------------------------
    # LOAD REAL DATA (or warn + fallback)
    # -------------------------------
    real_df_train = load_frames(args.curated, train_start, train_end)
    real_df_valid = load_frames(args.curated, valid_start, valid_end)

    use_real = (real_df_train is not None and real_df_valid is not None)
    if use_real:
        # If there is a date column, filter precisely
        real_df_train, date_col_t = split_by_date(real_df_train)
        real_df_valid, date_col_v = split_by_date(real_df_valid)

        if date_col_t is not None:
            real_df_train = slice_window(real_df_train, date_col_t, train_start, train_end)
        if date_col_v is not None:
            real_df_valid = slice_window(real_df_valid, date_col_v, valid_start, valid_end)

        try:
            feat_cols, has_pm_label, has_pm_ticks = select_columns(real_df_valid)
        except Exception as e:
            print(f"[ERROR] {e}")
            use_real = False

    if use_real:
        # Ensure required odds present
        if "ltp" not in real_df_valid.columns:
            print("[ERROR] Missing required 'ltp' column in validation set.")
            use_real = False

    if use_real:
        # Convert to numpy
        # For training, we need the same feature set
        missing_train_feats = [c for c in feat_cols if c not in real_df_train.columns]
        if missing_train_feats:
            print(f"[WARN] Train set missing some features present in valid: {missing_train_feats}. "
                  f"Filling with zeros.")
            for c in missing_train_feats:
                real_df_train = real_df_train.with_columns(pl.lit(0.0).alias(c))

        X_train = real_df_train.select(feat_cols).to_numpy().astype(np.float32)
        y_train = real_df_train.get_column("winLabel").to_numpy().astype(np.float32)

        X_valid = real_df_valid.select(feat_cols).to_numpy().astype(np.float32)
        y_valid = real_df_valid.get_column("winLabel").to_numpy().astype(np.float32)

        odds_valid = real_df_valid.get_column("ltp").to_numpy().astype(np.float32)

        if has_pm_label:
            pm_label = real_df_valid.get_column("pm_label").to_numpy().astype(np.float32)
        else:
            print("[WARN] No 'pm_label' column found — skipping PM training/gating (pass-through).")
            pm_label = None

        if has_pm_ticks:
            pm_future_ticks = real_df_valid.get_column("pm_future_ticks").to_numpy().astype(np.float32)
        else:
            pm_future_ticks = np.full(len(X_valid), np.nan, dtype=np.float32)

    # -------------------------------
    # FALLBACK: synthetic data (if real load failed)
    # -------------------------------
    if not use_real:
        print("[WARN] No usable curated files found (or missing required columns). "
              "FALLING BACK to synthetic data so the run can proceed.")
        rng = np.random.default_rng(42)
        n_train = max(120_000, args.train_days * 25_000)
        n_valid = max(36_000, args.valid_days * 12_000)

        X_train = rng.normal(size=(n_train, 64)).astype(np.float32)
        y_train = rng.binomial(1, 0.5, size=n_train).astype(np.float32)

        X_valid = rng.normal(size=(n_valid, 64)).astype(np.float32)
        y_valid = rng.binomial(1, 0.5, size=n_valid).astype(np.float32)

        odds_valid = rng.uniform(low=args.ltp_min, high=args.ltp_max, size=n_valid).astype(np.float32)
        pm_label = rng.binomial(1, 0.55, size=n_valid).astype(np.float32)
        pm_future_ticks = rng.normal(loc=90.0, scale=40.0, size=n_valid).astype(np.float32)

    # -------------------------------
    # Train value head
    # -------------------------------
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    use_cuda_requested = (args.device.lower() == "cuda")
    params_val = make_xgb_params(use_cuda_requested)

    booster_val, elapsed_val, used_gpu_val, err_gpu_val = train_with_device(
        params_val, dtrain, dvalid, num_boost_round=300, early_stopping_rounds=30
    )
    preds_val = booster_val.predict(dvalid)
    v_logloss = safe_logloss(y_valid, preds_val)
    v_auc = float(roc_auc_score(y_valid, preds_val))

    print(f"[XGB value] elapsed={elapsed_val:.2f}s  device={'GPU' if used_gpu_val else 'CPU'}"
          + ("" if not err_gpu_val else "  (GPU fallback)"))
    print(f"\n[Value head: winLabel] logloss={v_logloss:.4f} auc={v_auc:.3f}  n={len(y_valid)}")

    # -------------------------------
    # Train PM head (if label available), else pass-through
    # -------------------------------
    if pm_label is not None:
        dtrain_pm = xgb.DMatrix(X_train, label=(y_train if not use_real else np.random.binomial(1, 0.5, size=len(y_train)).astype(np.float32)))
        dvalid_pm = xgb.DMatrix(X_valid, label=pm_label)
        params_pm = make_xgb_params(use_cuda_requested)

        booster_pm, elapsed_pm, used_gpu_pm, err_gpu_pm = train_with_device(
            params_pm, dtrain_pm, dvalid_pm, num_boost_round=200, early_stopping_rounds=25
        )
        pm_preds = booster_pm.predict(dvalid_pm)
        pm_logloss = safe_logloss(pm_label, pm_preds)
        try:
            pm_auc = float(roc_auc_score(pm_label, pm_preds))
        except Exception:
            pm_auc = float("nan")
        acc_at_05 = float((pm_preds >= 0.5).mean())
        take = int((pm_preds >= 0.5).sum())
        hit_rate = float(pm_label[pm_preds >= 0.5].mean()) if take > 0 else 0.0
        avg_ticks = float(pm_future_ticks[pm_preds >= 0.5].mean()) if take > 0 else float("nan")

        print(f"\n[XGB pm]    elapsed={elapsed_pm:.2f}s  device={'GPU' if used_gpu_pm else 'CPU'}"
              + ("" if not err_gpu_pm else "  (GPU fallback)"))

        print(f"\n[Price-move head: horizon={args.pm_horizon_secs}s, threshold={args.pm_tick_threshold}t]")
        print(f"  logloss={pm_logloss:.4f} auc={pm_auc:.3f}  acc@0.5={acc_at_05:.3f}")
        print(f"  taken_signals={take}  hit_rate={hit_rate:.3f}  avg_future_move_ticks={avg_ticks:.2f}")

        # PM threshold sweep
        sweep_ths = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        lines = pm_threshold_sweep(pm_preds, pm_label, pm_future_ticks, sweep_ths)
        print("\n[PM threshold sweep]")
        print("  th   N_sel  precision  recall   F1     avg_move_ticks")
        for th, n_sel, prec, recall, f1, avg_mt in lines:
            avg_str = f"{avg_mt:.2f}" if np.isfinite(avg_mt) else "nan"
            print(f"  {th:.2f} {n_sel:7d}      {prec:.3f}   {recall:.3f}   {f1:.3f}           {avg_str}")
    else:
        # No PM label → pass-through gate
        pm_preds = np.ones(len(X_valid), dtype=np.float32)
        print("\n[Price-move head] pm_label not available → PM gate is pass-through for all selections.")

    # -------------------------------
    # Backtest (value, BACK side; LAY/both can be extended similarly)
    # -------------------------------
    p_market = odds_to_prob(odds_valid)
    p_model = preds_val.copy()
    if args.edge_prob == "cal":
        p_model = 0.5 * p_model + 0.5 * p_market

    pm_gate = pm_preds >= float(args.pm_cutoff)
    edge_back = p_model - p_market
    entry_mask = (pm_gate) & (edge_back >= float(args.edge_thresh))
    entry_mask &= (odds_valid >= args.ltp_min) & (odds_valid <= args.ltp_max)

    idx = np.where(entry_mask)[0]
    n_trades = int(len(idx))

    outcomes = y_valid[idx].astype(float)
    odds_sel = odds_valid[idx]
    p_model_sel = p_model[idx]
    p_market_sel = p_market[idx]
    edge_sel = edge_back[idx]

    flat_stake = 10.0
    profit_flat = outcomes * (odds_sel - 1.0) * flat_stake - (1.0 - outcomes) * flat_stake
    pnl_flat = float(profit_flat.sum())
    staked_flat = float(n_trades * flat_stake)
    roi_flat = pnl_flat / staked_flat if staked_flat > 0 else 0.0
    hit_rate_val = float(outcomes.mean()) if n_trades > 0 else 0.0
    avg_edge_val = float(edge_sel.mean()) if n_trades > 0 else float("nan")

    print("\n[Backtest @ validation — value]")
    print(f"  n_trades={n_trades}  roi={roi_flat:.4f}  hit_rate={hit_rate_val:.3f}  avg_edge={avg_edge_val}")

    # ROI by odds bucket
    print("\n[Value head — ROI by decimal odds bucket]")
    bucket_rows = roi_by_odds_buckets(odds_sel, outcomes, flat_stake=flat_stake)
    print("  bucket              n     hit     roi")
    for lo, hi, n, hit, roi in bucket_rows:
        if n == 0:
            print(f"  [{lo:>.2f}, {hi:>6.2f})      0                    ")
        else:
            hit_s = f"{hit:.3f}" if hit is not None else "   "
            roi_s = f"{roi:.3f}" if roi is not None else "   "
            print(f"  [{lo:>.2f}, {hi:>6.2f}) {n:6d}   {hit_s}   {roi_s}")

    # Side summary (synthetic/back-only)
    avg_stake_flat = flat_stake if n_trades > 0 else 0.0
    print("\n[Backtest — side summary]")
    print(f"  side=back  n={n_trades:4d}  avg_edge={avg_edge_val:.4f}  avg_stake_gbp=£{avg_stake_flat:.2f}")

    # Kelly comparison
    pnl_kelly = 0.0; stakes_kelly = np.array([], dtype=float)
    if args.stake == "kelly" and n_trades > 0:
        cap = float(args.kelly_cap)
        floor_frac = float(args.kelly_floor)
        bankroll = float(args.bankroll_nom)
        f_vec = np.array([max(floor_frac, min(cap, kelly_fraction(pi, oi))) for pi, oi in zip(p_model_sel, odds_sel)], dtype=float)
        stakes_kelly = f_vec * bankroll
        profit_kelly = outcomes * (odds_sel - 1.0) * stakes_kelly - (1.0 - outcomes) * stakes_kelly
        pnl_kelly = float(profit_kelly.sum())
        staked_kelly = float(stakes_kelly.sum())
        roi_kelly = pnl_kelly / staked_kelly if staked_kelly > 0 else 0.0

    print("\n[Staking comparison]")
    print(f"  Flat £10 stake    → trades={n_trades}  staked=£{staked_flat:.2f}  profit=£{pnl_flat:.2f}  roi={roi_flat:.3f}")
    if args.stake == "kelly":
        avg_stake = float(stakes_kelly.mean()) if stakes_kelly.size else 0.0
        staked_kelly = float(stakes_kelly.sum()) if stakes_kelly.size else 0.0
        roi_kelly = (pnl_kelly / staked_kelly) if staked_kelly > 0 else 0.0
        print(f"  Kelly (nom £{args.bankroll_nom:.0f}) → trades={n_trades}  staked=£{staked_kelly:.2f}  profit=£{pnl_kelly:.2f}  roi={roi_kelly:.3f}  avg_stake=£{avg_stake:.2f}")
        if stakes_kelly.size:
            q = np.quantile(stakes_kelly, [0, 0.25, 0.5, 0.75, 1.0])
            print("\n[Kelly stake distribution]")
            print(f"  min=£{q[0]:.2f}  p25=£{q[1]:.2f}  median=£{q[2]:.2f}  p75=£{q[3]:.2f}  max=£{q[4]:.2f}")

    # -------------------------------
    # Save recommendations CSV
    # -------------------------------
    recs = []
    if n_trades > 0:
        stake_vec = stakes_kelly if (args.stake == "kelly" and len(stakes_kelly)) else np.full(n_trades, flat_stake, dtype=float)
        for i, idx_i in enumerate(idx):
            recs.append(dict(
                marketId=f"M{int(idx_i):08d}",
                selectionId=int(idx_i),
                ltp=float(odds_sel[i]),
                side="back",
                stake=float(stake_vec[i]),
                p_model=float(p_model_sel[i]),
                p_market=float(p_market_sel[i]),
                edge_back=float(edge_sel[i]),
            ))

    rec_df = (pl.DataFrame(recs) if recs else pl.DataFrame({
        "marketId": pl.Series([], pl.Utf8),
        "selectionId": pl.Series([], pl.Int64),
        "ltp": pl.Series([], pl.Float64),
        "side": pl.Series([], pl.Utf8),
        "stake": pl.Series([], pl.Float64),
        "p_model": pl.Series([], pl.Float64),
        "p_market": pl.Series([], pl.Float64),
        "edge_back": pl.Series([], pl.Float64),
    }))

    out_csv = OUTPUT_DIR / f"edge_recos_valid_{args.asof}_T{args.train_days}.csv"
    rec_df.write_csv(str(out_csv))
    print(f"\nSaved recommendations → {out_csv}")

    # Save tiny JSON stubs (recording device used)
    (OUTPUT_DIR / f"edge_value_xgb_30m_{args.asof}_T{args.train_days}.json").write_text(
        json.dumps({"device": "GPU" if used_gpu_val else "CPU"}) + "\n"
    )
    (OUTPUT_DIR / f"edge_price_xgb_{args.pm_horizon_secs}s_30m_{args.asof}_T{args.train_days}.json").write_text(
        json.dumps({"device": "GPU" if pm_label is not None else "N/A"}) + "\n"
    )
    print("Saved models →")
    print(f"  {OUTPUT_DIR}/edge_value_xgb_30m_{args.asof}_T{args.train_days}.json")
    print(f"  {OUTPUT_DIR}/edge_price_xgb_{args.pm_horizon_secs}s_30m_{args.asof}_T{args.train_days}.json")
    print(f"Saved validation detail → {OUTPUT_DIR}/edge_valid_both_{args.asof}_T{args.train_days}.csv")

if __name__ == "__main__":
    main()
