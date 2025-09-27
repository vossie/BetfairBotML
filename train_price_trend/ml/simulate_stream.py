#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import timedelta, timezone
from collections import defaultdict, deque

import numpy as np
import polars as pl
import xgboost as xgb
import pickle
import json

# --- local imports
from utils import (
    parse_date, to_ms, read_snapshots,
    implied_prob_from_ltp_expr, time_to_off_minutes_expr, filter_preoff,
    write_json, kelly_fraction_back, kelly_fraction_lay
)

UTC = timezone.utc

# match feature engineering in build_features.py (5s cadence)
LAG_30S = 6
LAG_60S = 12
LAG_120S = 24

NUMERIC_FEATS = [
    # base
    "ltp", "tradedVolume", "spreadTicks", "imbalanceBest1", "mins_to_off", "__p_now",
    # diffs
    "ltp_diff_5s", "vol_diff_5s",
    # lags
    "ltp_lag30s","ltp_lag60s","ltp_lag120s",
    "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
    # momentum / returns
    "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
    "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
]

def parse_args():
    p = argparse.ArgumentParser(description="Streaming-style simulator for price trend model (liquidity-aware + settlement P&L)")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)                 # inclusive valid end
    p.add_argument("--start-date", required=True)           # simulation window start
    p.add_argument("--valid-days", type=int, default=7)     # simulate over last N days up to asof
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120) # for documentation; model fixed
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--edge-thresh", type=float, default=0.002)  # realistic default (≈0.2% per £1)
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--odds-min", type=float, default=None, help="optional min odds (ltp) filter")
    p.add_argument("--odds-max", type=float, default=None, help="optional max odds (ltp) filter")
    # Liquidity controls
    p.add_argument("--liquidity-levels", type=int, default=1, help="sum available size across top-L book levels")
    p.add_argument("--enforce-liquidity", action="store_true", help="cap stake by available book size")
    # I/O + runtime
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--calib-path", default="", help="optional isotonic.pkl for calibrating __p_pred (rare for delta model)")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])  # used by XGB for pred
    p.add_argument("--batch-size", type=int, default=200_000, help="batch predict to avoid RAM spikes")
    return p.parse_args()

def _make_features_row(bufs, last_row):
    """Return engineered features for current tick (or None if not enough history)."""
    if len(bufs["ltp"]) < LAG_120S or len(bufs["vol"]) < LAG_120S:
        return None

    ltp_now = last_row["ltp"]
    vol_now = last_row["tradedVolume"]

    ltp_l30 = bufs["ltp"][-LAG_30S]
    ltp_l60 = bufs["ltp"][-LAG_60S]
    ltp_l120= bufs["ltp"][-LAG_120S]

    vol_l30 = bufs["vol"][-LAG_30S]
    vol_l60 = bufs["vol"][-LAG_60S]
    vol_l120= bufs["vol"][-LAG_120S]

    ltp_prev = bufs["ltp"][-1] if len(bufs["ltp"])>=1 else None
    vol_prev = bufs["vol"][-1] if len(bufs["vol"])>=1 else None
    ltp_diff_5s = (ltp_now - ltp_prev) if ltp_prev is not None else 0.0
    vol_diff_5s = (vol_now - vol_prev) if vol_prev is not None else 0.0

    def safe_div(a, b):
        return (a / b) if (b is not None and abs(b) > 1e-12) else None

    feats = {
        "ltp": ltp_now,
        "tradedVolume": vol_now,
        "spreadTicks": last_row.get("spreadTicks", 0.0) or 0.0,
        "imbalanceBest1": last_row.get("imbalanceBest1", 0.0) or 0.0,
        "mins_to_off": last_row["mins_to_off"],
        "__p_now": 1.0 / max(ltp_now, 1e-12),

        "ltp_diff_5s": ltp_diff_5s,
        "vol_diff_5s": vol_diff_5s,

        "ltp_lag30s": ltp_l30,
        "ltp_lag60s": ltp_l60,
        "ltp_lag120s": ltp_l120,

        "tradedVolume_lag30s": vol_l30,
        "tradedVolume_lag60s": vol_l60,
        "tradedVolume_lag120s": vol_l120,

        "ltp_mom_30s": ltp_now - ltp_l30,
        "ltp_mom_60s": ltp_now - ltp_l60,
        "ltp_mom_120s": ltp_now - ltp_l120,

        "ltp_ret_30s": safe_div(ltp_now - ltp_l30, ltp_l30),
        "ltp_ret_60s": safe_div(ltp_now - ltp_l60, ltp_l60),
        "ltp_ret_120s": safe_div(ltp_now - ltp_l120, ltp_l120),
    }
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = np.nan
    return feats

# ---------- EV helpers ----------
def _ev_mtm(p_now: float, p_pred: float, commission: float, side: str) -> float:
    """Mark-to-market EV per £1 stake based on predicted future implied probability."""
    eps = 1e-6
    p_now_c  = max(eps, min(1.0 - eps, p_now))
    p_pred_c = max(eps, min(1.0 - eps, p_pred))
    if side == "back":
        # profit = (p_pred / p_now) - 1
        return ((p_pred_c / p_now_c) - 1.0) * (1.0 - commission)
    else:
        # profit = 1 - (p_now / p_pred)
        return (1.0 - (p_now_c / p_pred_c)) * (1.0 - commission)

def _ev_settlement(p_pred: float, ltp: float, commission: float, side: str) -> float:
    """Settlement-style EV per £1 stake (valid only if p_pred is a true win probability)."""
    if side == "back":
        # p * (odds-1)*(1-comm) - (1-p)
        return p_pred * (ltp - 1.0) * (1.0 - commission) - (1.0 - p_pred)
    else:
        # (1-p)*(1-comm) - p*(odds-1)
        return (1.0 - p_pred) * (1.0 - commission) - p_pred * (ltp - 1.0)

# ---------- Liquidity helpers ----------
def _sum_available_size(side: str, back_sizes, lay_sizes, levels: int) -> float:
    """
    Sum available size across top 'levels' for the appropriate side.
    Assumption: provided sizes arrays represent available size you can take now.
    """
    sizes = back_sizes if side == "back" else lay_sizes
    if not sizes:
        return 0.0
    if levels <= 0:
        levels = 1
    return float(sum(sizes[:levels]))

# ---------- Results loader ----------
def _load_results(curated_root: Path, sport: str, start_dt, end_dt) -> pl.DataFrame:
    """
    Load settled results for [start_dt .. end_dt] (inclusive by day).
    Expects schema with: marketId (str), selectionId (int), winLabel (0/1).
    """
    # daily partitions
    days = []
    day = start_dt
    while day <= end_dt:
        days.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)

    paths = [f"{curated_root}/results/sport={sport}/date={d}/*.parquet" for d in days]
    # lazily scan & union; tolerate missing days via lazy scan try/catch
    lfs = []
    for pat in paths:
        try:
            lfs.append(pl.scan_parquet(pat))
        except Exception:
            continue
    if not lfs:
        return pl.DataFrame({"marketId": [], "selectionId": [], "winLabel": []})
    lf = pl.concat(lfs, how="vertical_relaxed")

    df = (
        lf.select([
            pl.col("marketId").cast(pl.Utf8),
            pl.col("selectionId").cast(pl.Int64),
            pl.col("winLabel").cast(pl.Int32).fill_null(0)
        ])
        .unique(maintain_order=False)
        .collect()
    )
    return df

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    curated = Path(args.curated)

    asof_dt = parse_date(args.asof)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)
    start_dt = parse_date(args.start_date)

    # Load only from start_dt..valid_end (snapshots) and filter to pre-off window
    lf = read_snapshots(curated, start_dt, valid_end, args.sport).with_columns([
        time_to_off_minutes_expr(),
    ])
    df = lf.collect()
    df = filter_preoff(df, args.preoff_max)

    # --- optional odds band filter on current LTP ---
    df = df.filter(pl.col("ltp").is_not_null())
    if args.odds_min is not None:
        df = df.filter(pl.col("ltp") >= float(args.odds_min))
    if args.odds_max is not None:
        df = df.filter(pl.col("ltp") <= float(args.odds_max))

    # Stream order: by time
    df = df.sort(["publishTimeMs","marketId","selectionId"])

    # Load model (+ optional calibrator)
    bst = xgb.Booster()
    bst.load_model(args.model_path)
    try:
        bst.set_param({"device": args.device})
    except Exception:
        pass

    calibrator = None
    if args.calib_path:
        with open(args.calib_path, "rb") as f:
            calibrator = pickle.load(f)

    # rolling buffers per (marketId, selectionId)
    bufs_ltp = defaultdict(lambda: deque(maxlen=LAG_120S))
    bufs_vol = defaultdict(lambda: deque(maxlen=LAG_120S))

    rows = df.iter_rows(named=True)
    feature_rows = []
    meta_rows = []

    # Simulate tick-by-tick
    for r in rows:
        key = (r["marketId"], int(r["selectionId"]))
        bufs_ltp[key].append(float(r["ltp"]))
        bufs_vol[key].append(float(r["tradedVolume"]))

        feats = _make_features_row({"ltp": bufs_ltp[key], "vol": bufs_vol[key]}, {
            "ltp": float(r["ltp"]),
            "tradedVolume": float(r["tradedVolume"]),
            "spreadTicks": float(r.get("spreadTicks") or 0.0),
            "imbalanceBest1": float(r.get("imbalanceBest1") or 0.0),
            "mins_to_off": float(r["mins_to_off"]),
        })
        if feats is None:  # not enough history yet
            continue

        # Keep meta incl. book sizes for liquidity capping
        meta_rows.append((
            r["publishTimeMs"],                    # 0
            r["marketId"],                         # 1
            int(r["selectionId"]),                 # 2
            float(r["ltp"]),                       # 3 (exec odds approximation)
            float(feats["__p_now"]),               # 4 (p_now from ltp)
            r.get("backSizes") or [],              # 5 list[float]
            r.get("laySizes") or [],               # 6 list[float]
        ))

        feature_rows.append([feats.get(c, np.nan) for c in NUMERIC_FEATS])

        # batch predict to keep memory bounded
        if len(feature_rows) >= args.batch_size:
            process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir)
            feature_rows.clear()
            meta_rows.clear()

    # flush remainder
    if feature_rows:
        process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir)

    # combine all chunks → trades.csv
    trades_parts = sorted(outdir.glob("trades_part_*.parquet"))
    if not trades_parts:
        print("[simulate] no trades generated.")
        return

    all_trades = pl.concat([pl.read_parquet(p) for p in trades_parts], how="vertical_relaxed")
    for p in trades_parts: p.unlink(missing_ok=True)

    # ----- JOIN RESULTS for settlement P&L -----
    results_df = _load_results(curated, args.sport, start_dt, valid_end)
    if results_df.height > 0:
        all_trades = (
            all_trades.join(
                results_df,
                on=["marketId","selectionId"],
                how="left",
            )
            .with_columns(pl.col("winLabel").fill_null(0).cast(pl.Int32))
        )
        # settlement P&L using exec odds ~= ltp at time of trade
        # back: win -> (odds-1)*stake*(1-comm); lose -> -stake
        # lay:  lose(win) -> -(odds-1)*stake; win(lose) -> +stake*(1-comm)
        comm = float(args.commission)

        def _real_pnl_expr():
            return (
                pl.when(pl.col("side") == "back")
                .then(
                    pl.when(pl.col("winLabel") == 1)
                    .then((pl.col("ltp") - 1.0) * pl.col("stake_filled") * (1.0 - comm))
                    .otherwise(-pl.col("stake_filled"))
                )
                .otherwise(
                    pl.when(pl.col("winLabel") == 1)
                    .then(-(pl.col("ltp") - 1.0) * pl.col("stake_filled"))
                    .otherwise(pl.col("stake_filled") * (1.0 - comm))
                )
            ).alias("real_pnl")
        all_trades = all_trades.with_columns(_real_pnl_expr())
    else:
        all_trades = all_trades.with_columns(pl.lit(None, dtype=pl.Float64).alias("real_pnl"))

    # write trades CSV
    trades_csv = outdir / f"trades_{args.asof}.csv"
    all_trades.write_csv(trades_csv)

    # daily summary: expected vs realised
    daily = (
        all_trades
        .with_columns(
            pl.from_epoch(pl.col("publishTimeMs").cast(pl.Int64), time_unit="ms")
            .dt.replace_time_zone("UTC").dt.date().cast(pl.Utf8).alias("day")
        )
        .group_by("day","stake_mode")
        .agg([
            pl.len().alias("n_trades"),
            pl.sum("exp_pnl").alias("exp_profit"),
            pl.sum("real_pnl").alias("real_profit"),
            pl.mean("ev_per_1").alias("avg_ev"),
            pl.mean("stake_filled").alias("avg_stake_filled"),
            pl.mean("fill_frac").alias("avg_fill_frac"),
        ])
        .with_columns([
            (pl.col("exp_profit")  / pl.lit(float(args.bankroll_nom))).alias("roi_exp"),
            (pl.col("real_profit") / pl.lit(float(args.bankroll_nom))).alias("roi_real"),
        ])
        .sort("day")
    )
    daily_csv = outdir / f"daily_{args.asof}.csv"
    daily.write_csv(daily_csv)

    # summary json (overall exp vs realised)
    total_exp_profit  = float(all_trades["exp_pnl"].sum())
    total_real_profit = float(all_trades["real_pnl"].sum()) if all_trades["real_pnl"].dtype != pl.Null else None
    summ = {
        "asof": args.asof,
        "start_date": args.start_date,
        "valid_days": args.valid_days,
        "preoff_max_minutes": args.preoff_max,
        "commission": args.commission,
        "edge_thresh": args.edge_thresh,
        "ev_mode": args.ev_mode,
        "odds_min": args.odds_min,
        "odds_max": args.odds_max,
        "stake_mode": args.stake_mode,
        "kelly_cap": args.kelly_cap,
        "kelly_floor": args.kelly_floor,
        "bankroll_nom": args.bankroll_nom,
        "liquidity_levels": args.liquidity_levels,
        "enforce_liquidity": bool(args.enforce_liquidity),
        "rows": int(all_trades.height),
        "n_trades": int(all_trades.height),
        "total_exp_profit": total_exp_profit,
        "total_real_profit": total_real_profit,
        "avg_ev_per_1": float(all_trades["ev_per_1"].mean()),
        "overall_roi_exp": (total_exp_profit / float(args.bankroll_nom)) if args.bankroll_nom else None,
        "overall_roi_real": (total_real_profit / float(args.bankroll_nom)) if (args.bankroll_nom and total_real_profit is not None) else None,
        "avg_fill_frac": float(all_trades["fill_frac"].mean()),
    }
    write_json(outdir / f"summary_{args.asof}.json", summ)

def process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir: Path):
    X = np.asarray(feature_rows, dtype=np.float32)

    # Use DMatrix for prediction (works with GPU when booster.device="cuda")
    dm = xgb.DMatrix(X, feature_names=NUMERIC_FEATS)
    delta_pred = bst.predict(dm)

    out = []
    commission = float(args.commission)
    for i, meta in enumerate(meta_rows):
        ts, mid, sid, ltp, p_now, back_sizes, lay_sizes = meta

        dp = float(delta_pred[i])
        p_pred = p_now + dp
        # clip probability to [0,1]
        p_pred = 0.0 if p_pred < 0.0 else (1.0 if p_pred > 1.0 else p_pred)
        if calibrator:
            try:
                p_pred = float(calibrator.transform(np.array([p_pred]))[0])
            except Exception:
                pass

        side = "back" if dp > 0 else ("lay" if dp < 0 else "none")
        if side == "none":
            continue

        # ----- EV per £1 -----
        if args.ev_mode == "mtm":
            ev_per_1 = _ev_mtm(p_now, p_pred, commission, side)
        else:
            ev_per_1 = _ev_settlement(p_pred, ltp, commission, side)

        if ev_per_1 < args.edge_thresh:
            continue

        # ----- Requested stake (pre-liquidity) -----
        if args.stake_mode == "flat":
            stake_req = 1.0
        else:
            if side == "back":
                f = kelly_fraction_back(p_pred, ltp, commission)
            else:
                f = kelly_fraction_lay(p_pred, ltp, commission)
            f = max(args.kelly_floor, min(args.kelly_cap, f))
            stake_req = f * float(args.bankroll_nom)

        # ----- Liquidity-aware cap -----
        if args.enforce_liquidity:
            avail = _sum_available_size(side, back_sizes, lay_sizes, args.liquidity_levels)
            stake_filled = min(stake_req, max(0.0, avail))
        else:
            stake_filled = stake_req

        if stake_filled <= 0.0:
            continue  # nothing filled

        fill_frac = (stake_filled / stake_req) if stake_req > 0 else 1.0

        exp_pnl = ev_per_1 * stake_filled
        out.append((
            ts, mid, sid, ltp, p_now, p_pred, dp, side, ev_per_1,
            stake_req, stake_filled, fill_frac, exp_pnl, args.stake_mode
        ))

    if not out:
        return
    part = pl.DataFrame(
        out,
        schema={
            "publishTimeMs": pl.Int64,
            "marketId": pl.Utf8,
            "selectionId": pl.Int64,
            "ltp": pl.Float64,
            "p_now": pl.Float64,
            "p_pred": pl.Float64,
            "delta_pred": pl.Float64,
            "side": pl.Utf8,
            "ev_per_1": pl.Float64,
            "stake_req": pl.Float64,
            "stake_filled": pl.Float64,
            "fill_frac": pl.Float64,
            "exp_pnl": pl.Float64,
            "stake_mode": pl.Utf8,
        },
        orient="row",
    )
    idx = len(list(outdir.glob("trades_part_*.parquet")))
    part.write_parquet(outdir / f"trades_part_{idx:05d}.parquet")

if __name__ == "__main__":
    main()
