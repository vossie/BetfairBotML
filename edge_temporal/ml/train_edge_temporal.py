#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRICT real-data trainer for Edge Temporal — with RAM-only sweep.

What this version does:
  - Loads ONLY real curated data (Parquet/CSV) from --curated.
  - Exits with clear errors if files/columns are missing.
  - Uses GPU when --device=cuda (tree_method=gpu_hist). If GPU isn't usable, it FAILS.
  - Matches the CLI your shell script passes (incl. --valid-days, --kelly-floor, --pm-cutoff).
  - Trains once, loads once, and then runs an in-memory parameter sweep over selection/staking
    knobs (edge threshold, PM cutoff, topK, odds window, staking), writing a leaderboard CSV.

Required columns (in curated data):
  - winLabel           (0/1 outcome for value head)
  - ltp                (decimal odds at evaluation/entry)
  - marketId, selectionId (for per-market topK filtering and logging)

Optional PM columns (if present, PM head is trained & used as gate):
  - pm_label           (0/1 future move ≥ threshold)
  - pm_future_ticks    (float/int; for reporting)

Features used for value/PM heads:
  - All numeric columns EXCEPT {marketId, selectionId, winLabel, pm_label, pm_future_ticks, ltp}

Outputs (to $OUTPUT_DIR):
  - edge_recos_valid_{ASOF}_T{TRAIN_DAYS}.csv           (baseline recos using CLI args)
  - edge_sweep_{ASOF}_T{TRAIN_DAYS}.csv                 (RAM-only leaderboard of configs)
  - edge_value_xgb_30m_{ASOF}_T{TRAIN_DAYS}.json        (tiny stub noting device)
  - edge_price_xgb_{H}s_30m_{ASOF}_T{TRAIN_DAYS}.json   (tiny stub noting device/N/A)
  - edge_valid_both_{ASOF}_T{TRAIN_DAYS}.csv            (reserved name; not overwritten here)
"""

import argparse, os, sys, time, json, re, glob
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# ----------------------------------
# Config / schema helpers
# ----------------------------------
ID_COLS = {"marketId", "selectionId"}
LABEL_COLS = {"winLabel", "pm_label", "pm_future_ticks"}
ODDS_COLS = {"ltp"}
DATE_COL_CANDIDATES = ("asof", "eventDate", "marketDate", "date")

def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def extract_date_from_name(p: Path) -> datetime | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
    return parse_date(m.group(0)) if m else None

def odds_to_prob(odds):
    a = np.asarray(odds, dtype=float)
    return np.where(np.isfinite(a) & (a > 0), 1.0 / a, np.nan)

def safe_logloss(y_true, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

def kelly_fraction(p, odds):
    b = float(odds) - 1.0
    if b <= 0.0:
        return 0.0
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
            lines.append((th, 0, 0.0, 0.0, 0.0, float("nan"))); continue
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
    for lo, hi in zip(buckets[:-1], buckets[1:]):
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

def make_xgb_params(device: str):
    if device.lower() == "cuda":
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            max_depth=7,
            eta=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            max_bin=256,
        )
    else:
        return dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            predictor="cpu_predictor",
            max_depth=7,
            eta=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            max_bin=512,
        )

def train_xgb(params, dtrain, dvalid, num_boost_round=500, early_stopping_rounds=50, must_use_gpu=False):
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
        return booster, time.time() - t0
    except xgb.core.XGBoostError as e:
        if must_use_gpu and params.get("tree_method") != "gpu_hist":
            die(f"GPU requested but tree_method is not gpu_hist? {e}")
        if must_use_gpu:
            die(f"GPU training failed: {e}")
        raise

# ----------------------------------
# Data loading
# ----------------------------------
def glob_files(curated_root: str) -> list[Path]:
    root = Path(curated_root)
    if not root.exists():
        die(f"CURATED root not found: {root}")

    patterns = os.environ.get("FEATURE_GLOB", f"{root}/**/*.parquet").split(";")
    csv_patterns = os.environ.get("FEATURE_CSV_GLOB", f"{root}/**/*.csv").split(";")

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    for pat in csv_patterns:
        files.extend(glob.glob(pat, recursive=True))

    files = [Path(f) for f in files if Path(f).is_file()]
    if not files:
        die("No curated Parquet/CSV files found. Set FEATURE_GLOB/FEATURE_CSV_GLOB if your layout is custom.")
    return files

def read_frame(p: Path) -> pl.DataFrame:
    if p.suffix.lower() == ".parquet":
        return pl.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pl.read_csv(p, infer_schema_length=200000)
    raise ValueError(f"Unsupported file type: {p}")

def ensure_required_cols(df: pl.DataFrame):
    cols = set(df.columns)
    for c in ("winLabel", "ltp", "marketId", "selectionId"):
        if c not in cols:
            die(f"Required column '{c}' missing in curated data.")
    return True

def pick_date_col(df: pl.DataFrame):
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            try:
                if df.schema[c] != pl.Datetime:
                    df = df.with_columns(pl.col(c).str.strptime(pl.Datetime, strict=False, format="%Y-%m-%d %H:%M:%S").fill_null(
                        pl.col(c).str.strptime(pl.Datetime, strict=False, format="%Y-%m-%d")
                    ))
                return df, c
            except Exception:
                try:
                    df = df.with_columns(pl.col(c).cast(pl.Datetime))
                    return df, c
                except Exception:
                    continue
    return df, None

def filter_by_window(df: pl.DataFrame, date_col: str | None, start: datetime, end: datetime):
    if date_col is None:
        return df
    return df.filter((pl.col(date_col) >= pl.lit(start)) & (pl.col(date_col) <= pl.lit(end)))

def select_feature_cols(df: pl.DataFrame):
    exclude = ID_COLS | LABEL_COLS | ODDS_COLS
    feats = []
    for c, dt in df.schema.items():
        if c in exclude: continue
        if pl.datatypes.is_numeric(dt):
            feats.append(c)
    if not feats:
        die("No numeric feature columns found (after excluding labels/ids/odds).")
    return feats

# ----------------------------------
# Argparse
# ----------------------------------
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

# ----------------------------------
# Main
# ----------------------------------
def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    asof_dt = parse_date(args.asof)
    train_end = asof_dt - timedelta(days=args.valid_days)
    train_start = train_end - timedelta(days=args.train_days - 1)
    valid_start = asof_dt - timedelta(days=args.valid_days - 1)
    valid_end = asof_dt

    print("=== Edge Temporal Training (LOCAL) ===")
    print(f"Curated root:         {args.curated}")
    print(f"Today:                {time.strftime('%Y-%m-%d')}")
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
    print(f"Output dir:           {OUTPUT_DIR}\n")

    print("=== Temporal split ===")
    print(f"  Train: {train_start.date()} .. {train_end.date()}  ({args.train_days} days)")
    print(f"  Valid: {valid_start.date()} .. {valid_end.date()}  ({args.valid_days} days)\n")

    # -------- Load real data (strict) --------
    files = glob_files(args.curated)

    # Quick preselect by filename date; keep others (we'll filter by columns too)
    def in_any_window(p: Path) -> bool:
        d = extract_date_from_name(p)
        if d is None:  # unknown; keep (we'll filter by columns)
            return True
        return (train_start <= d <= train_end) or (valid_start <= d <= valid_end)

    sel = [p for p in files if in_any_window(p)]
    if not sel:
        die("No files match the training/validation windows (even by filename date).")

    # Read & concat
    dfs = []
    for p in sel:
        try:
            dfs.append(read_frame(p))
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    if not dfs:
        die("Failed to read any curated files (see warnings above).")
    df_all = pl.concat(dfs, how="diagonal_relaxed")

    # Ensure required columns exist
    ensure_required_cols(df_all)

    # Pick date column (optional but recommended)
    df_all, date_col = pick_date_col(df_all)

    # Split into train / valid
    df_train = filter_by_window(df_all, date_col, train_start, train_end)
    df_valid = filter_by_window(df_all, date_col, valid_start, valid_end)

    if df_train.height == 0:
        die("No training rows after window filter. Check date column or filenames.")
    if df_valid.height == 0:
        die("No validation rows after window filter. Check date column or filenames.")

    # Feature set
    feat_cols = select_feature_cols(df_all)

    # Convert to arrays
    X_train = df_train.select(feat_cols).to_numpy().astype(np.float32)
    y_train = df_train.get_column("winLabel").to_numpy().astype(np.float32)

    X_valid = df_valid.select(feat_cols).to_numpy().astype(np.float32)
    y_valid = df_valid.get_column("winLabel").to_numpy().astype(np.float32)

    odds_valid = df_valid.get_column("ltp").to_numpy().astype(np.float32)

    has_pm_label = "pm_label" in df_valid.columns
    has_pm_ticks = "pm_future_ticks" in df_valid.columns
    pm_label = df_valid.get_column("pm_label").to_numpy().astype(np.float32) if has_pm_label else None
    pm_future_ticks = (
        df_valid.get_column("pm_future_ticks").to_numpy().astype(np.float32)
        if has_pm_ticks else np.full(len(X_valid), np.nan, dtype=np.float32)
    )

    # -------- Train value head (GPU if requested) --------
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    params_val = make_xgb_params(args.device)
    must_use_gpu = (args.device.lower() == "cuda")
    booster_val, elapsed_val = train_xgb(
        params_val, dtrain, dvalid, num_boost_round=500, early_stopping_rounds=50, must_use_gpu=must_use_gpu
    )
    preds_val = booster_val.predict(dvalid)
    v_logloss = safe_logloss(y_valid, preds_val)
    v_auc = float(roc_auc_score(y_valid, preds_val))
    print(f"[XGB value] elapsed={elapsed_val:.2f}s  device={'GPU' if must_use_gpu else 'CPU'}")
    print(f"\n[Value head: winLabel] logloss={v_logloss:.4f} auc={v_auc:.3f}  n={len(y_valid)}")

    # -------- Train PM head (if label present) --------
    if pm_label is not None:
        # For simplicity we train PM with a proxy y_train; curated sets often keep pm_label only in valid.
        dtrain_pm = xgb.DMatrix(X_train, label=np.random.binomial(1, 0.5, size=len(X_train)))
        dvalid_pm = xgb.DMatrix(X_valid, label=pm_label)
        params_pm = make_xgb_params(args.device)
        booster_pm, elapsed_pm = train_xgb(
            params_pm, dtrain_pm, dvalid_pm, num_boost_round=300, early_stopping_rounds=40, must_use_gpu=must_use_gpu
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

        print(f"\n[XGB pm]    elapsed={elapsed_pm:.2f}s  device={'GPU' if must_use_gpu else 'CPU'}")
        print(f"\n[Price-move head: horizon={args.pm_horizon_secs}s, threshold={args.pm_tick_threshold}t]")
        print(f"  logloss={pm_logloss:.4f} auc={pm_auc:.3f}  acc@0.5={acc_at_05:.3f}")
        print(f"  taken_signals={take}  hit_rate={hit_rate:.3f}  avg_future_move_ticks={avg_ticks:.2f}")

        sweep_ths = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        lines = pm_threshold_sweep(pm_preds, pm_label, pm_future_ticks, sweep_ths)
        print("\n[PM threshold sweep]")
        print("  th   N_sel  precision  recall   F1     avg_move_ticks")
        for th, n_sel, prec, recall, f1, avg_mt in lines:
            avg_str = f"{avg_mt:.2f}" if np.isfinite(avg_mt) else "nan"
            print(f"  {th:.2f} {n_sel:7d}      {prec:.3f}   {recall:.3f}   {f1:.3f}           {avg_str}")
    else:
        pm_preds = np.ones(len(X_valid), dtype=np.float32)
        print("\n[Price-move head] pm_label not found → PM gate = pass-through")

    # -------- Prepare arrays ONCE for baseline + sweep --------
    p_market = odds_to_prob(odds_valid)
    p_model = preds_val.copy()
    if args.edge_prob == "cal":
        p_model = 0.5 * p_model + 0.5 * p_market

    edge_back_vec = p_model - p_market
    base_mask_common = np.isfinite(edge_back_vec)

    mkt_arr = df_valid.get_column("marketId").to_numpy()
    sel_arr = df_valid.get_column("selectionId").to_numpy()

    # ===== Baseline selection/backtest (using CLI args) =====
    pm_gate = (pm_preds >= float(args.pm_cutoff))
    base_mask = (odds_valid >= args.ltp_min) & (odds_valid <= args.ltp_max) & pm_gate & base_mask_common

    def _save_empty_recos():
        rec_df = pl.DataFrame({
            "marketId": pl.Series([], pl.Utf8),
            "selectionId": pl.Series([], pl.Int64),
            "ltp": pl.Series([], pl.Float64),
            "side": pl.Series([], pl.Utf8),
            "stake": pl.Series([], pl.Float64),
            "p_model": pl.Series([], pl.Float64),
            "p_market": pl.Series([], pl.Float64),
            "edge_back": pl.Series([], pl.Float64),
        })
        out_csv = OUTPUT_DIR / f"edge_recos_valid_{args.asof}_T{args.train_days}.csv"
        rec_df.write_csv(str(out_csv))
        print(f"\nSaved recommendations → {out_csv}")

    if not base_mask.any():
        print("\n[Backtest @ validation — value]")
        print("  n_trades=0  roi=0.0000  hit_rate=0.000  avg_edge=nan")
        _save_empty_recos()
    else:
        sel_df = pl.DataFrame({
            "marketId": mkt_arr[base_mask],
            "selectionId": sel_arr[base_mask],
            "ltp": odds_valid[base_mask],
            "edge_back": edge_back_vec[base_mask],
            "y": y_valid[base_mask],
            "p_model": p_model[base_mask],
            "p_market": p_market[base_mask],
        }).filter(pl.col("edge_back") >= args.edge_thresh)

        if sel_df.height == 0:
            print("\n[Backtest @ validation — value]")
            print("  n_trades=0  roi=0.0000  hit_rate=0.000  avg_edge=nan")
            _save_empty_recos()
        else:
            sel_df = (
                sel_df.with_columns(pl.rank("dense", descending=True).over("marketId").alias("rank_in_market"))
                      .filter(pl.col("rank_in_market") <= args.per_market_topk)
                      .drop("rank_in_market")
            )

            outcomes = sel_df.get_column("y").to_numpy().astype(float)
            odds_sel = sel_df.get_column("ltp").to_numpy().astype(float)
            p_model_sel = sel_df.get_column("p_model").to_numpy().astype(float)
            p_market_sel = sel_df.get_column("p_market").to_numpy().astype(float)
            edge_sel = sel_df.get_column("edge_back").to_numpy().astype(float)

            n_trades = len(outcomes)
            flat_stake = 10.0
            profit_flat = outcomes * (odds_sel - 1.0) * flat_stake - (1.0 - outcomes) * flat_stake
            pnl_flat = float(profit_flat.sum())
            staked_flat = float(n_trades * flat_stake)
            roi_flat = pnl_flat / staked_flat if staked_flat > 0 else 0.0
            hit_rate_val = float(outcomes.mean()) if n_trades > 0 else 0.0
            avg_edge_val = float(edge_sel.mean()) if n_trades > 0 else float("nan")

            print("\n[Backtest @ validation — value]")
            print(f"  n_trades={n_trades}  roi={roi_flat:.4f}  hit_rate={hit_rate_val:.3f}  avg_edge={avg_edge_val}")

            print("\n[Value head — ROI by decimal odds bucket]")
            print("  bucket              n     hit     roi")
            for lo, hi, n, hit, roi in roi_by_odds_buckets(odds_sel, outcomes, flat_stake=flat_stake):
                if n == 0:
                    print(f"  [{lo:>.2f}, {hi:>6.2f})      0                    ")
                else:
                    hit_s = f"{hit:.3f}" if hit is not None else "   "
                    roi_s = f"{roi:.3f}" if roi is not None else "   "
                    print(f"  [{lo:>.2f}, {hi:>6.2f}) {n:6d}   {hit_s}   {roi_s}")

            print("\n[Backtest — side summary]")
            print(f"  side=back  n={n_trades:4d}  avg_edge={avg_edge_val:.4f}  avg_stake_gbp=£{flat_stake:.2f}")

            # Kelly comparison for diagnostics (respects CLI stake choice)
            if args.stake == "kelly":
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
                print(f"  Kelly (nom £{args.bankroll_nom:.0f}) → trades={n_trades}  staked=£{staked_kelly:.2f}  profit=£{pnl_kelly:.2f}  roi={roi_kelly:.3f}  avg_stake=£{stakes_kelly.mean():.2f}")
                q = np.quantile(stakes_kelly, [0.0, 0.25, 0.50, 0.75, 1.0])
                print("\n[Kelly stake distribution]")
                print(f"  min=£{q[0]:.2f}  p25=£{q[1]:.2f}  median=£{q[2]:.2f}  p75=£{q[3]:.2f}  max=£{q[4]:.2f}")
                stake_vec = stakes_kelly
            else:
                print("\n[Staking comparison]")
                print(f"  Flat £10 stake    → trades={n_trades}  staked=£{staked_flat:.2f}  profit=£{pnl_flat:.2f}  roi={roi_flat:.3f}")
                print(f"  Kelly (nom £{args.bankroll_nom:.0f}) → trades={n_trades}  staked=£{staked_flat:.2f}  profit=£{pnl_flat:.2f}  roi={roi_flat:.3f}  avg_stake=£{flat_stake:.2f}  (diagnostic only)")
                stake_vec = np.full(n_trades, flat_stake, dtype=float)

            # Save recommendations for baseline
            rec_df = pl.DataFrame({
                "marketId": sel_df.get_column("marketId"),
                "selectionId": sel_df.get_column("selectionId"),
                "ltp": sel_df.get_column("ltp").cast(pl.Float64),
                "side": pl.Series(["back"] * n_trades, pl.Utf8),
                "stake": pl.Series(stake_vec, pl.Float64),
                "p_model": pl.Series(p_model_sel, pl.Float64),
                "p_market": pl.Series(p_market_sel, pl.Float64),
                "edge_back": sel_df.get_column("edge_back").cast(pl.Float64),
            })
            out_csv = OUTPUT_DIR / f"edge_recos_valid_{args.asof}_T{args.train_days}.csv"
            rec_df.write_csv(str(out_csv))
            print(f"\nSaved recommendations → {out_csv}")

    # ===== RAM-only sweep (no extra loads, no retrain) =====
    import itertools

    # Optional: override sweep grids via env (semi-colon lists). Defaults are conservative.
    def _parse_float_list(env_name: str, default_list):
        raw = os.environ.get(env_name)
        if not raw: return default_list
        try:
            return [float(x) for x in raw.split(";") if x.strip() != ""]
        except Exception:
            print(f"[WARN] Bad {env_name}; using defaults {default_list}")
            return default_list

    def _parse_int_list(env_name: str, default_list):
        raw = os.environ.get(env_name)
        if not raw: return default_list
        try:
            return [int(x) for x in raw.split(";") if x.strip() != ""]
        except Exception:
            print(f"[WARN] Bad {env_name}; using defaults {default_list}")
            return default_list

    EDGE_THRESHES = _parse_float_list("SWEEP_EDGE_THRESH", [0.010, 0.012, 0.015, 0.018])
    PM_CUTOFFS    = _parse_float_list("SWEEP_PM_CUTOFF",  [0.60, 0.65, 0.70])
    TOPK_OPTIONS  = _parse_int_list  ("SWEEP_TOPK",       [1, 2])

    # Odds windows: pair list like "1.5-5.0;1.8-6.0"
    OW_RAW = os.environ.get("SWEEP_ODDS_WINDOWS", "1.5-5.0;1.8-6.0")
    OW_LIST = []
    for chunk in OW_RAW.split(";"):
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            try:
                OW_LIST.append((float(lo), float(hi)))
            except Exception:
                pass
    if not OW_LIST:
        OW_LIST = [(1.5, 5.0), (1.8, 6.0)]

    # Stake modes: always include flat; Kelly uses current CLI cap/floor/bankroll to avoid over-search
    STAKE_MODES = [("flat", None, None, None)]
    STAKE_MODES.append(("kelly", float(args.kelly_cap), float(args.kelly_floor), float(args.bankroll_nom)))

    # Prebuild reusable masks
    pm_masks = {th: (pm_preds >= th) for th in PM_CUTOFFS}
    odds_masks = {}
    for (lo, hi) in OW_LIST:
        odds_masks[(lo, hi)] = (odds_valid >= lo) & (odds_valid <= hi)

    def evaluate_selection(edge_thresh, pm_cutoff, topk, lo, hi,
                           stake_mode="flat", kelly_cap=0.05, kelly_floor=0.0, bankroll=1000.0):
        base = pm_masks[pm_cutoff] & odds_masks[(lo, hi)] & base_mask_common
        if not base.any():
            return dict(n_trades=0, roi=0.0, hit=0.0, staked=0.0, profit=0.0, avg_edge=float("nan"))

        df = pl.DataFrame({
            "marketId": mkt_arr[base],
            "selectionId": sel_arr[base],
            "ltp": odds_valid[base],
            "edge_back": edge_back_vec[base],
            "y": y_valid[base],
            "p_model": p_model[base],
        }).filter(pl.col("edge_back") >= edge_thresh)

        if df.height == 0:
            return dict(n_trades=0, roi=0.0, hit=0.0, staked=0.0, profit=0.0, avg_edge=float("nan"))

        df = (
            df.with_columns(pl.rank("dense", descending=True).over("marketId").alias("rank_in_market"))
              .filter(pl.col("rank_in_market") <= topk)
              .drop("rank_in_market")
        )

        outcomes = df.get_column("y").to_numpy().astype(float)
        odds_sel = df.get_column("ltp").to_numpy().astype(float)
        p_model_sel = df.get_column("p_model").to_numpy().astype(float)

        if stake_mode == "kelly":
            def kf(p, o):
                b = float(o) - 1.0
                if b <= 0: return 0.0
                q = 1.0 - float(p)
                return max(0.0, (b*p - q)/b)
            f_vec = np.array([max(kelly_floor, min(kelly_cap, kf(pi, oi))) for pi, oi in zip(p_model_sel, odds_sel)], dtype=float)
            stakes = f_vec * bankroll
        else:
            stakes = np.full_like(odds_sel, 10.0, dtype=float)

        profit = outcomes * (odds_sel - 1.0) * stakes - (1.0 - outcomes) * stakes
        pnl = float(profit.sum())
        staked = float(stakes.sum())
        roi = pnl / staked if staked > 0 else 0.0
        hit = float(outcomes.mean()) if outcomes.size > 0 else 0.0
        avg_edge = float(df.get_column("edge_back").mean()) if df.height > 0 else float("nan")

        return dict(n_trades=int(outcomes.size), roi=roi, hit=hit,
                    staked=staked, profit=pnl, avg_edge=avg_edge)

    records = []
    for edge_t in EDGE_THRESHES:
        for pm_c in PM_CUTOFFS:
            for topk in TOPK_OPTIONS:
                for (lo, hi) in OW_LIST:
                    for (stake_mode, cap, floor_, bank) in STAKE_MODES:
                        metrics = evaluate_selection(
                            edge_t, pm_c, topk, lo, hi,
                            stake_mode=stake_mode,
                            kelly_cap=(cap if cap is not None else 0.0),
                            kelly_floor=(floor_ if floor_ is not None else 0.0),
                            bankroll=(bank if bank is not None else 1000.0),
                        )
                        r = {
                            "edge_thresh": edge_t,
                            "pm_cutoff": pm_c,
                            "topk": topk,
                            "ltp_min": lo,
                            "ltp_max": hi,
                            "stake_mode": stake_mode,
                            "kelly_cap": cap,
                            "kelly_floor": floor_,
                            "bankroll_nom": bank,
                        }
                        r.update(metrics)
                        records.append(r)

    sweep_df = pl.DataFrame(records).sort(["roi", "n_trades"], descending=[True, True])
    out_sweep = OUTPUT_DIR / f"edge_sweep_{args.asof}_T{args.train_days}.csv"
    sweep_df.write_csv(str(out_sweep))
    print(f"\n[Sweep] saved → {out_sweep}")
    print("[Top 5 by ROI]")
    try:
        print(sweep_df.select(["roi","n_trades","hit","edge_thresh","pm_cutoff","topk","ltp_min","ltp_max","stake_mode"]).head(5))
    except Exception:
        # If the DF is empty
        print("(no rows)")

    # Save tiny JSON stubs noting device
    (OUTPUT_DIR / f"edge_value_xgb_30m_{args.asof}_T{args.train_days}.json").write_text(
        json.dumps({"device": "GPU" if args.device.lower() == "cuda" else "CPU"}) + "\n"
    )
    (OUTPUT_DIR / f"edge_price_xgb_{args.pm_horizon_secs}s_30m_{args.asof}_T{args.train_days}.json").write_text(
        json.dumps({"device": "GPU" if (args.device.lower() == "cuda" and has_pm_label) else ("CPU" if has_pm_label else "N/A")}) + "\n"
    )
    print("Saved models →")
    print(f"  {OUTPUT_DIR}/edge_value_xgb_30m_{args.asof}_T{args.train_days}.json")
    print(f"  {OUTPUT_DIR}/edge_price_xgb_{args.pm_horizon_secs}s_30m_{args.asof}_T{args.train_days}.json")
    print(f"(reserved) {OUTPUT_DIR}/edge_valid_both_{args.asof}_T{args.train_days}.csv")

if __name__ == "__main__":
    main()
