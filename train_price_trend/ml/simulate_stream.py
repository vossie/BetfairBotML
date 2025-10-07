#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, os, glob, math, json, sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import polars as pl

# Optional XGBoost (prediction); script still runs without it if a precomputed ev is provided
try:
    import numpy as np
    import xgboost as xgb
except Exception:
    xgb = None
    np = None

# --------------------------- Logging ---------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

# --------------------------- Dates -----------------------------

def _parse_date(s: str) -> datetime.date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _date_iter(start: str, end: str):
    d = _parse_date(start)
    e = _parse_date(end)
    while d <= e:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)

# --------------------------- IO helpers ------------------------

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

def _uniform_sample(seq: list[str], k: int) -> list[str]:
    if k <= 0 or k >= len(seq):
        return seq
    step = len(seq) / float(k)
    idxs, seen = [], set()
    for i in range(k):
        idx = int(round(i * step))
        idx = max(0, min(len(seq) - 1, idx))
        if idx not in seen:
            idxs.append(idx); seen.add(idx)
    j = 0
    while len(idxs) < k:
        cand = j % len(seq)
        if cand not in seen:
            idxs.append(cand); seen.add(cand)
        j += 1
    idxs.sort()
    return [seq[i] for i in idxs]

def scan_obs_glob(glob_path: str, want_cols: list[str], max_files: int | None, sample_mode: str) -> tuple[pl.LazyFrame | None, int]:
    files = sorted(glob.glob(glob_path))
    total = len(files)
    if not files:
        return None, 0
    if max_files is not None and max_files > 0 and total > max_files:
        if sample_mode == "uniform":
            files = _uniform_sample(files, max_files)
        elif sample_mode == "tail":
            files = files[-max_files:]
        else:
            files = files[:max_files]
    lf = pl.scan_parquet(files, low_memory=True)
    have = lf.collect_schema().names()
    take = [c for c in want_cols if c in have]
    if not take:
        return None, total
    return lf.select([pl.col(c) for c in take]), total

def scan_defs_range(curated: str, sport: str, start_date: str, end_date: str) -> tuple[pl.LazyFrame | None, int]:
    base = Path(curated) / "market_definitions" / f"sport={sport}"
    files: list[str] = []
    for d in _date_iter(start_date, end_date):
        files.extend(glob.glob(str(base / f"date={d}" / "*.parquet")))
    if not files:
        return None, 0
    lf = pl.scan_parquet(files, low_memory=True).select([
        pl.col("marketId").cast(pl.Utf8),
        pl.col("marketStartMs").cast(pl.Int64),
        pl.col("countryCode").cast(pl.Utf8),
        pl.col("publishTimeMs").alias("__def_pub").cast(pl.Int64),
    ]).filter(pl.col("marketId").is_not_null())
    lf = (
        lf.sort(["marketId","__def_pub"])
          .group_by("marketId")
          .agg([
              pl.col("marketStartMs").last().alias("marketStartMs"),
              pl.col("countryCode").last().alias("countryCode"),
          ])
    )
    return lf, len(files)

# --------------------------- Features & EV ----------------------

def add_features_with_future(
    lf_join: pl.LazyFrame,
    preoff_max_min: int,
    horizon_secs: int,
    row_sample_secs: int | None,
    apply_preoff_filter: bool
) -> pl.LazyFrame:
    lf = lf_join
    if row_sample_secs and row_sample_secs > 0:
        step = max(1, int(round(row_sample_secs / 5)))
        lf = lf.with_columns(((pl.col("publishTimeMs") // 5000) % step).alias("__mod")).filter(pl.col("__mod") == 0).drop("__mod")

    lf = lf.with_columns([
        pl.col("ltp").cast(pl.Float32).alias("ltp_f"),
        pl.col("tradedVolume").cast(pl.Float32).alias("vol_f"),
        ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off"),
    ])

    grp = ["marketId", "selectionId"]
    steps = max(1, int(round(horizon_secs / 5)))
    lf = lf.sort(["marketId", "selectionId", "publishTimeMs"]).with_columns([
        pl.col("ltp_f").diff().over(grp).alias("ltp_diff_5s"),
        pl.col("vol_f").diff().over(grp).alias("vol_diff_5s"),
        pl.col("ltp_f").shift(6).over(grp).alias("ltp_lag30s"),
        pl.col("ltp_f").shift(12).over(grp).alias("ltp_lag60s"),
        pl.col("ltp_f").shift(24).over(grp).alias("ltp_lag120s"),
        pl.col("vol_f").shift(6).over(grp).alias("tradedVolume_lag30s"),
        pl.col("vol_f").shift(12).over(grp).alias("tradedVolume_lag60s"),
        pl.col("vol_f").shift(24).over(grp).alias("tradedVolume_lag120s"),
    ]).with_columns([
        (pl.col("ltp_f") - pl.col("ltp_lag30s")).alias("ltp_mom_30s"),
        (pl.col("ltp_f") - pl.col("ltp_lag60s")).alias("ltp_mom_60s"),
        (pl.col("ltp_f") - pl.col("ltp_lag120s")).alias("ltp_mom_120s"),
        pl.when(pl.col("ltp_lag30s") > 0).then((pl.col("ltp_f")/pl.col("ltp_lag30s") - 1.0)).otherwise(0.0).alias("ltp_ret_30s"),
        pl.when(pl.col("ltp_lag60s") > 0).then((pl.col("ltp_f")/pl.col("ltp_lag60s") - 1.0)).otherwise(0.0).alias("ltp_ret_60s"),
        pl.when(pl.col("ltp_lag120s") > 0).then((pl.col("ltp_f")/pl.col("ltp_lag120s") - 1.0)).otherwise(0.0).alias("ltp_ret_120s"),
        pl.col("ltp_f").shift(-steps).over(grp).alias("ltp_future"),
        (pl.col("ltp_f").shift(-steps).over(grp) - pl.col("ltp_f")).alias("__dp"),
    ])

    if apply_preoff_filter:
        lf = lf.filter(
            (pl.col("marketStartMs").is_not_null()) &
            (pl.col("__dp").is_not_null()) &
            (pl.col("mins_to_off") >= 0) &
            (pl.col("mins_to_off") <= float(preoff_max_min))
        )
    else:
        lf = lf.filter(pl.col("__dp").is_not_null())

    # IMPORTANT: keep diff columns so model prediction doesn't fail
    keep = [
        "marketId","selectionId","publishTimeMs","marketStartMs","mins_to_off","countryCode",
        "ltp_f","vol_f",
        "ltp_diff_5s","vol_diff_5s",                 # <-- added: required by model
        "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
        "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
        "ltp_lag30s","ltp_lag60s","ltp_lag120s",
        "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
        "ltp_future","__dp"
    ]
    have = lf.collect_schema().names()
    take = [c for c in keep if c in have]
    return lf.select([pl.col(c) for c in take])

def clamp_to_ev(dp: pl.Expr, ev_scale: float, ev_cap: float) -> pl.Expr:
    x = pl.when(dp > ev_cap).then(pl.lit(ev_cap)).otherwise(dp)
    x = pl.when(x < -ev_cap).then(pl.lit(-ev_cap)).otherwise(x)
    return (x * ev_scale).cast(pl.Float64)

# --------------------------- Liquidity checks ------------------

def level1_fill_ok(df: pl.DataFrame, min_fill_frac: float) -> pl.Series:
    have = set(df.columns)
    need = {"backSizes","laySizes"}
    if not need.issubset(have):
        return pl.Series([True] * df.height, dtype=pl.Boolean)
    bs = df["backSizes"].list.get(0).fill_null(0.0)
    return (bs >= float(min_fill_frac))

# --------------------------- Staking ---------------------------

def net_back_odds(odds: float, commission: float) -> float:
    return max(0.0, (odds - 1.0) * (1.0 - commission))

def kelly_fraction_back_from_ev(ev_per_1: float, odds: float, commission: float) -> float:
    o = max(1.01, float(odds))
    p = max(0.0, min(1.0, (float(ev_per_1) + 1.0) / o))
    b = net_back_odds(o, commission)
    if b <= 0.0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)

def size_stake(
    stake_mode: str,
    ev: float,
    odds: float,
    commission: float,
    bankroll: float,
    kelly_cap: float,
    kelly_floor: float,
    per_market_budget: float,
    slots_in_market: int
) -> float:
    if stake_mode == "flat":
        return max(0.0, float(per_market_budget / max(1, slots_in_market)))
    f = kelly_fraction_back_from_ev(ev, odds, commission)
    f = min(kelly_cap, max(kelly_floor, f))
    stake = bankroll * f
    cap_unit = per_market_budget / max(1, slots_in_market)
    return float(min(cap_unit, stake))

# --------------------------- Prediction ------------------------

def load_model(model_path: str):
    if xgb is None:
        raise RuntimeError("xgboost is not available in this environment.")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

def predict_dp(booster, feats_df: pl.DataFrame, feature_names: list[str], batch_size: int = 200_000) -> pl.Series:
    take = [c for c in feature_names if c in feats_df.columns]
    if not take:
        raise RuntimeError("No overlapping feature columns between data and model feature names.")
    pdf = feats_df.select(take).to_pandas()
    for c in pdf.columns:
        if pdf[c].dtype != "float32":
            pdf[c] = pdf[c].astype("float32")
    preds_all = []
    n = len(pdf)
    for i in range(0, n, batch_size):
        pdf_batch = pdf.iloc[i:i+batch_size]
        try:
            dm = xgb.QuantileDMatrix(pdf_batch, missing=np.nan)
        except Exception:
            dm = xgb.DMatrix(pdf_batch, missing=np.nan)
        preds = booster.predict(dm)
        preds_all.append(preds)
    preds = np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=np.float32)
    return pl.Series(preds, dtype=pl.Float64)

# --------------------------- Selection -------------------------

def select_trades(
    df: pl.DataFrame,
    edge_thresh: float,
    odds_min: Optional[float],
    odds_max: Optional[float],
    enforce_liquidity: bool,
    min_fill_frac: float,
    per_market_topk: int
) -> pl.DataFrame:
    out = df.filter(pl.col("ev_per_1").is_not_null() & (pl.col("ev_per_1") >= float(edge_thresh)))
    if odds_min is not None:
        out = out.filter(pl.col("ltp_f") >= float(odds_min))
    if odds_max is not None:
        out = out.filter(pl.col("ltp_f") <= float(odds_max))
    if out.is_empty():
        return out
    if enforce_liquidity:
        ok = level1_fill_ok(out, min_fill_frac)
        out = out.filter(ok)
        if out.is_empty():
            return out
    # Polars-version-safe per-market topK by EV (no pl.row_number)
    out = (
        out.sort(["marketId", "ev_per_1"], descending=[False, True])
           .with_columns(
               pl.col("ev_per_1")
                 .rank(method="dense", descending=True)
                 .over("marketId")
                 .alias("__rk_ev")
           )
           .filter(pl.col("__rk_ev") <= int(per_market_topk))
           .drop("__rk_ev")
    )
    return out

# --------------------------- P&L computation -------------------

def compute_pnl(
    trades: pl.DataFrame,
    stake_mode: str,
    bankroll_nom: float,
    commission: float,
    kelly_cap: float,
    kelly_floor: float,
    per_market_budget: float,
    exit_on_move_ticks: int
) -> pl.DataFrame:
    if trades.is_empty():
        return trades
    slots = trades.group_by("marketId").len().rename({"len":"__slots"})
    trades = trades.join(slots, on="marketId", how="left")

    def stake_udf(ev, odds, slots_in_market):
        return size_stake(
            stake_mode=stake_mode,
            ev=float(ev),
            odds=float(odds),
            commission=commission,
            bankroll=bankroll_nom,
            kelly_cap=kelly_cap,
            kelly_floor=kelly_floor,
            per_market_budget=per_market_budget,
            slots_in_market=int(max(1, slots_in_market))
        )

    trades = trades.with_columns([
        pl.struct(["ev_per_1","ltp_f","__slots"]).map_elements(
            lambda s: stake_udf(s["ev_per_1"], s["ltp_f"], s["__slots"])
        ).alias("stake")
    ])

    trades = trades.with_columns((pl.col("ev_per_1") * pl.col("stake")).alias("exp_pnl"))

    if exit_on_move_ticks and "ltp_lag30s" in trades.columns:
        tick = 0.01
        cap = float(exit_on_move_ticks) * tick
        dp_real = pl.when(pl.col("__dp") > cap).then(pl.lit(cap)) \
                   .when(pl.col("__dp") < -cap).then(pl.lit(-cap)) \
                   .otherwise(pl.col("__dp"))
        trades = trades.with_columns((dp_real * pl.col("stake")).alias("real_pnl_mtm"))
    else:
        trades = trades.with_columns((pl.col("__dp") * pl.col("stake")).alias("real_pnl_mtm"))

    trades = trades.with_columns(pl.col("real_pnl_mtm").alias("real_pnl_settle"))
    return trades.drop("__slots")

# --------------------------- CLI -------------------------------

def parse_args():
    ap = argparse.ArgumentParser("simulate_stream (trading-aware)")
    ap.add_argument("--curated", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--valid-days", type=int, default=7)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--preoff-max", type=int, default=30)
    ap.add_argument("--horizon-secs", type=int, default=120)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--ev-scale", type=float, default=1.0)
    ap.add_argument("--ev-cap", type=float, default=1.0)
    ap.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=200_000)
    ap.add_argument("--edge-thresh", type=float, default=0.001)
    ap.add_argument("--stake-mode", choices=["flat","kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.02)
    ap.add_argument("--kelly-floor", type=float, default=0.001)
    ap.add_argument("--bankroll-nom", type=float, default=5000.0)
    ap.add_argument("--odds-min", type=float, default=None)
    ap.add_argument("--odds-max", type=float, default=None)
    ap.add_argument("--enforce-liquidity", action="store_true")
    ap.add_argument("--liquidity-levels", type=int, default=1)
    ap.add_argument("--min-fill-frac", type=float, default=0.25)
    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--per-market-budget", type=float, default=10.0)
    ap.add_argument("--exit-on-move-ticks", type=int, default=0)
    ap.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-files-per-day", type=int, default=0)
    ap.add_argument("--file-sample-mode", choices=["uniform","head","tail"], default="uniform")
    ap.add_argument("--row-sample-secs", type=int, default=0)
    ap.add_argument("--polars-max-threads", type=int, default=0)
    ap.add_argument("--defs-days-back", type=int, default=30)
    ap.add_argument("--defs-days-forward", type=int, default=7)
    ap.add_argument("--fallback-coverage-thresh", type=float, default=0.01)
    ap.add_argument("--force-skip-preoff", action="store_true")
    return ap.parse_args()

# --------------------------- Main ------------------------------

def main():
    args = parse_args()
    if args.polars_max_threads > 0:
        os.environ["POLARS_MAX_THREADS"] = str(args.polars_max_threads)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_days = list(_date_iter(args.start_date, args.asof))
    days = all_days[-args.valid_days:]

    defs_start = (_parse_date(args.start_date) - timedelta(days=args.defs_days_back)).strftime("%Y-%m-%d")
    defs_end   = (_parse_date(args.asof) + timedelta(days=args.defs_days_forward)).strftime("%Y-%m-%d")
    log(f"[simulate] scanning definitions {defs_start} .. {defs_end}")
    lf_defs, _ = scan_defs_range(args.curated, args.sport, defs_start, defs_end)

    # Model
    booster = None
    model_feature_names: list[str] = []
    if args.model_path and Path(args.model_path).exists():
        try:
            booster = load_model(args.model_path)
            # XGBoost may not persist feature names; accept empty and fall back later.
            model_feature_names = booster.feature_names or []
            log(f"[simulate] loaded model with {len(model_feature_names)} features")
        except Exception as e:
            log(f"[simulate] WARNING: failed to load model: {e}")

    base_cols = [
        "sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume",
        "backTicks","backSizes","layTicks","laySizes"
    ]
    total_days = 0
    mean_evs = []
    trades_rows: List[dict] = []

    for day in days:
        ob_dir = Path(args.curated) / "orderbook_snapshots_5s" / f"sport={args.sport}" / f"date={day}"
        lf_obs, total_files = scan_obs_glob(str(ob_dir / "*.parquet"), base_cols,
                                            args.max_files_per_day if args.max_files_per_day>0 else None,
                                            args.file_sample_mode)
        cap = args.max_files_per_day or "∞"
        log(f"[simulate] {day} … files={total_files} (cap={cap}, mode={args.file_sample_mode})")
        if lf_obs is None:
            log(f"[simulate] {day} … no snapshots; skipping")
            continue

        lf_join = lf_obs.join(lf_defs, on="marketId", how="left") if lf_defs is not None else lf_obs.with_columns([
            pl.lit(None).alias("marketStartMs"),
            pl.lit("UNK").alias("countryCode"),
        ])

        cov_lf = lf_join.select([pl.len().alias("__n"), pl.col("marketStartMs").is_not_null().sum().alias("__nn")])
        cov = collect_streaming(cov_lf).to_dict(as_series=False)
        n_total = int(cov["__n"][0]) if cov["__n"] else 0
        n_nonnull = int(cov["__nn"][0]) if cov["__nn"] else 0
        pct = (n_nonnull / n_total) if n_total else 0.0
        log(f"[simulate] {day} join coverage: rows={n_total:,} marketStartMs non-null={n_nonnull:,} ({pct:.2%})")

        apply_preoff = (not args.force_skip_preoff) and (pct >= args.fallback_coverage_thresh)
        if not apply_preoff:
            why = "force" if args.force_skip_preoff else f"coverage {pct:.2%} < {args.fallback_coverage_thresh:.2%}"
            log(f"[simulate] {day} pre-off filter: SKIPPED ({why})")

        lf_feats = add_features_with_future(lf_join, args.preoff_max, args.horizon_secs, args.row_sample_secs, apply_preoff)

        pre_stats = collect_streaming(
            lf_feats.select([
                pl.len().alias("__all"),
                pl.col("mins_to_off").is_between(0, float(args.preoff_max)).sum().alias("__inwin")
            ])
        ).to_dict(as_series=False)
        n_all = int(pre_stats["__all"][0]) if pre_stats["__all"] else 0
        n_in = int(pre_stats["__inwin"][0]) if pre_stats["__inwin"] else 0
        log(f"[simulate] {day} pre-off window rows: {n_in:,} / {n_all:,}")

        df = collect_streaming(lf_feats)
        if df.height == 0:
            log(f"[simulate] {day} … 0 rows post-feature; skipping")
            continue

        # Predict with model if possible; else use observed __dp (oracle)
        if booster is not None:
            # If booster has feature names, use them; else infer (drop identifiers/labels)
            if not model_feature_names:
                drop = {"marketId","selectionId","publishTimeMs","marketStartMs","countryCode","ltp_future","__dp"}
                model_feature_names = [c for c in df.columns if c not in drop]
            try:
                pred = predict_dp(booster, df, model_feature_names, batch_size=args.batch_size)
                df = df.with_columns(pred.alias("__pred_dp"))
                dp_for_ev = pl.col("__pred_dp")
            except Exception as e:
                log(f"[simulate] WARNING: prediction failed ({e}); using observed __dp for EV")
                dp_for_ev = pl.col("__dp")
        else:
            dp_for_ev = pl.col("__dp")

        df = df.with_columns(clamp_to_ev(dp_for_ev, args.ev_scale, args.ev_cap).alias("ev_per_1"))
        df = df.filter(pl.col("ev_per_1").is_not_null())
        if df.height == 0:
            log(f"[simulate] {day} … 0 rows after EV computation; skipping")
            continue

        avg_ev = float(df["ev_per_1"].mean())
        mean_evs.append(avg_ev)
        total_days += 1
        log(f"[simulate] {day} … rows={df.height:,}  avg_ev/£1={avg_ev:.6f} → sim_{day}.parquet")

        # Select & size trades
        sel = select_trades(
            df,
            edge_thresh=args.edge_thresh,
            odds_min=args.odds_min,
            odds_max=args.odds_max,
            enforce_liquidity=bool(args.enforce_liquidity),
            min_fill_frac=float(args.min_fill_frac),
            per_market_topk=int(args.per_market_topk),
        )
        if sel.is_empty():
            continue

        trades = compute_pnl(
            sel,
            stake_mode=args.stake_mode,
            bankroll_nom=float(args.bankroll_nom),
            commission=float(args.commission),
            kelly_cap=float(args.kelly_cap),
            kelly_floor=float(args.kelly_floor),
            per_market_budget=float(args.per_market_budget),
            exit_on_move_ticks=int(args.exit_on_move_ticks),
        )

        if trades.height:
            trades_rows.extend(trades.to_dicts())

        # Persist per-day EV rows (debugging)
        try:
            df.write_parquet(out_dir / f"sim_{day}.parquet")
        except Exception:
            pass

    # Aggregate & summarize
    if not trades_rows:
        log("[simulate] No trades selected across days.")
        summary = {
            "edge_thresh": args.edge_thresh,
            "stake_mode": args.stake_mode,
            "odds_min": args.odds_min,
            "odds_max": args.odds_max,
            "enforce_liquidity_effective": bool(args.enforce_liquidity),
            "liquidity_levels": int(args.liquidity_levels or 0),
            "min_fill_frac": float(args.min_fill_frac),
            "per_market_topk": int(args.per_market_topk),
            "per_market_budget": float(args.per_market_budget),
            "exit_on_move_ticks": int(args.exit_on_move_ticks),
            "ev_scale_used": float(args.ev_scale),
            "n_trades": 0,
            "overall_roi_exp": 0.0,
            "overall_roi_real_mtm": 0.0,
            "overall_roi_real_settle": 0.0,
            "avg_ev_per_1": float(np.nan) if np is not None else None,
        }
        (out_dir / f"summary_{args.asof}.json").write_text(json.dumps(summary, indent=2))
        return

    trades_df = pl.DataFrame(trades_rows)
    n_trades = int(trades_df.height)
    total_exp_profit = float(trades_df["exp_pnl"].sum())
    total_real_mtm = float(trades_df["real_pnl_mtm"].sum())
    total_real_settle = float(trades_df["real_pnl_settle"].sum())
    avg_ev_all = float(trades_df["ev_per_1"].mean())

    roi_exp = total_exp_profit / float(args.bankroll_nom) if args.bankroll_nom else None
    roi_mtm = total_real_mtm / float(args.bankroll_nom) if args.bankroll_nom else None
    roi_settle = total_real_settle / float(args.bankroll_nom) if args.bankroll_nom else None

    trades_csv = out_dir / f"trades_{args.asof}.csv"
    trades_df.write_csv(trades_csv)

    summary = {
        "edge_thresh": args.edge_thresh,
        "stake_mode": args.stake_mode,
        "odds_min": args.odds_min,
        "odds_max": args.odds_max,
        "enforce_liquidity_effective": bool(args.enforce_liquidity),
        "liquidity_levels": int(args.liquidity_levels or 0),
        "min_fill_frac": float(args.min_fill_frac),
        "per_market_topk": int(args.per_market_topk),
        "per_market_budget": float(args.per_market_budget),
        "exit_on_move_ticks": int(args.exit_on_move_ticks),
        "ev_scale_used": float(args.ev_scale),

        "n_trades": n_trades,
        "total_exp_profit": total_exp_profit,
        "overall_roi_exp": roi_exp,
        "overall_roi_real_mtm": roi_mtm,
        "overall_roi_real_settle": roi_settle,

        "avg_ev_per_1": avg_ev_all,
    }
    (out_dir / f"summary_{args.asof}.json").write_text(json.dumps(summary, indent=2))

    log(f"[simulate] Completed {total_days} day(s).")
    if mean_evs:
        log(f"[simulate] Mean EV/£1 over days = {sum(mean_evs)/len(mean_evs):.6f}")
    log(f"[simulate] Trades: n={n_trades:,}  ROI_exp={roi_exp:.6f}  ROI_mtm={roi_mtm:.6f}")

if __name__ == "__main__":
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(200)
    main()
