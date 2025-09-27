#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
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
    p = argparse.ArgumentParser(description="Streaming-style simulator for price trend model")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)                 # inclusive valid end
    p.add_argument("--start-date", required=True)           # simulation window start
    p.add_argument("--valid-days", type=int, default=7)     # simulate over last N days up to asof
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120) # for documentation; model fixed
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--edge-thresh", type=float, default=0.002)  # realistic default (≈0.2%)
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--odds-min", type=float, default=None, help="optional min odds (ltp) filter")
    p.add_argument("--odds-max", type=float, default=None, help="optional max odds (ltp) filter")
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--calib-path", default="", help="optional isotonic.pkl for calibrating __p_pred (rare for delta model)")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])  # used by XGB for pred
    p.add_argument("--batch-size", type=int, default=200_000, help="batch predict to avoid RAM spikes")
    return p.parse_args()

def _make_features_row(bufs, last_row):
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

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    curated = Path(args.curated)

    asof_dt = parse_date(args.asof)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)
    start_dt = parse_date(args.start_date)

    # Load only from start_dt..valid_end
    lf = read_snapshots(curated, start_dt, valid_end, args.sport).with_columns([
        time_to_off_minutes_expr(),
    ])
    df = lf.collect()
    df = filter_preoff(df, args.preoff_max)

    # --- NEW: optional odds band filter on current LTP ---
    # drop null ltp, then apply min/max if provided
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

        feature_rows.append([feats.get(c, np.nan) for c in NUMERIC_FEATS])
        meta_rows.append((
            r["publishTimeMs"], r["marketId"], int(r["selectionId"]),
            float(r["ltp"]), float(feats["__p_now"])
        ))

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
    if trades_parts:
        all_trades = pl.concat([pl.read_parquet(p) for p in trades_parts], how="vertical_relaxed")
        trades_csv = outdir / f"trades_{args.asof}.csv"
        all_trades.write_csv(trades_csv)
        for p in trades_parts: p.unlink(missing_ok=True)

        # daily summary (+ ROI per day)
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
                pl.mean("ev_per_1").alias("avg_ev"),
                pl.mean("stake").alias("avg_stake"),
            ])
            .with_columns(
                (pl.col("exp_profit") / pl.lit(float(args.bankroll_nom))).alias("roi")
            )
            .sort("day")
        )
        daily_csv = outdir / f"daily_{args.asof}.csv"
        daily.write_csv(daily_csv)

        # summary json (also include overall ROI + odds band used)
        total_exp_profit = float(all_trades["exp_pnl"].sum())
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
            "rows": int(all_trades.height),
            "n_trades": int(all_trades.height),
            "total_exp_profit": total_exp_profit,
            "avg_ev_per_1": float(all_trades["ev_per_1"].mean()),
            "overall_roi": (total_exp_profit / float(args.bankroll_nom)) if args.bankroll_nom else None,
        }
        write_json(outdir / f"summary_{args.asof}.json", summ)

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
    """Settlement-style EV per £1 stake (only valid if p_pred is a true win probability)."""
    if side == "back":
        # p * (odds-1)*(1-comm) - (1-p)
        return p_pred * (ltp - 1.0) * (1.0 - commission) - (1.0 - p_pred)
    else:
        # (1-p)*(1-comm) - p*(odds-1)
        return (1.0 - p_pred) * (1.0 - commission) - p_pred * (ltp - 1.0)

def process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir: Path):
    X = np.asarray(feature_rows, dtype=np.float32)

    # Use DMatrix for prediction (works with GPU when booster.device="cuda")
    dm = xgb.DMatrix(X, feature_names=NUMERIC_FEATS)
    delta_pred = bst.predict(dm)

    out = []
    commission = float(args.commission)
    for i, (ts, mid, sid, ltp, p_now) in enumerate(meta_rows):
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

        # stake sizing
        if args.stake_mode == "flat":
            stake = 1.0
        else:
            if side == "back":
                f = kelly_fraction_back(p_pred, ltp, commission)
            else:
                f = kelly_fraction_lay(p_pred, ltp, commission)
            f = max(args.kelly_floor, min(args.kelly_cap, f))
            stake = f * float(args.bankroll_nom)

        exp_pnl = ev_per_1 * stake
        out.append((ts, mid, sid, ltp, p_now, p_pred, dp, side, ev_per_1, stake, exp_pnl,
                    args.stake_mode))

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
            "stake": pl.Float64,
            "exp_pnl": pl.Float64,
            "stake_mode": pl.Utf8,
        },
        orient="row",
    )
    idx = len(list(outdir.glob("trades_part_*.parquet")))
    part.write_parquet(outdir / f"trades_part_{idx:05d}.parquet")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
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
    p = argparse.ArgumentParser(description="Streaming-style simulator for price trend model")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)                 # inclusive valid end
    p.add_argument("--start-date", required=True)           # simulation window start
    p.add_argument("--valid-days", type=int, default=7)     # simulate over last N days up to asof
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120) # for documentation; model fixed
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--edge-thresh", type=float, default=0.002)  # realistic default (≈0.2%)
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--odds-min", type=float, default=None, help="optional min odds (ltp) filter")
    p.add_argument("--odds-max", type=float, default=None, help="optional max odds (ltp) filter")
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--calib-path", default="", help="optional isotonic.pkl for calibrating __p_pred (rare for delta model)")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])  # used by XGB for pred
    p.add_argument("--batch-size", type=int, default=200_000, help="batch predict to avoid RAM spikes")
    return p.parse_args()

def _make_features_row(bufs, last_row):
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

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    curated = Path(args.curated)

    asof_dt = parse_date(args.asof)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)
    start_dt = parse_date(args.start_date)

    # Load only from start_dt..valid_end
    lf = read_snapshots(curated, start_dt, valid_end, args.sport).with_columns([
        time_to_off_minutes_expr(),
    ])
    df = lf.collect()
    df = filter_preoff(df, args.preoff_max)

    # --- NEW: optional odds band filter on current LTP ---
    # drop null ltp, then apply min/max if provided
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

        feature_rows.append([feats.get(c, np.nan) for c in NUMERIC_FEATS])
        meta_rows.append((
            r["publishTimeMs"], r["marketId"], int(r["selectionId"]),
            float(r["ltp"]), float(feats["__p_now"])
        ))

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
    if trades_parts:
        all_trades = pl.concat([pl.read_parquet(p) for p in trades_parts], how="vertical_relaxed")
        trades_csv = outdir / f"trades_{args.asof}.csv"
        all_trades.write_csv(trades_csv)
        for p in trades_parts: p.unlink(missing_ok=True)

        # daily summary (+ ROI per day)
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
                pl.mean("ev_per_1").alias("avg_ev"),
                pl.mean("stake").alias("avg_stake"),
            ])
            .with_columns(
                (pl.col("exp_profit") / pl.lit(float(args.bankroll_nom))).alias("roi")
            )
            .sort("day")
        )
        daily_csv = outdir / f"daily_{args.asof}.csv"
        daily.write_csv(daily_csv)

        # summary json (also include overall ROI + odds band used)
        total_exp_profit = float(all_trades["exp_pnl"].sum())
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
            "rows": int(all_trades.height),
            "n_trades": int(all_trades.height),
            "total_exp_profit": total_exp_profit,
            "avg_ev_per_1": float(all_trades["ev_per_1"].mean()),
            "overall_roi": (total_exp_profit / float(args.bankroll_nom)) if args.bankroll_nom else None,
        }
        write_json(outdir / f"summary_{args.asof}.json", summ)

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
    """Settlement-style EV per £1 stake (only valid if p_pred is a true win probability)."""
    if side == "back":
        # p * (odds-1)*(1-comm) - (1-p)
        return p_pred * (ltp - 1.0) * (1.0 - commission) - (1.0 - p_pred)
    else:
        # (1-p)*(1-comm) - p*(odds-1)
        return (1.0 - p_pred) * (1.0 - commission) - p_pred * (ltp - 1.0)

def process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir: Path):
    X = np.asarray(feature_rows, dtype=np.float32)

    # Use DMatrix for prediction (works with GPU when booster.device="cuda")
    dm = xgb.DMatrix(X, feature_names=NUMERIC_FEATS)
    delta_pred = bst.predict(dm)

    out = []
    commission = float(args.commission)
    for i, (ts, mid, sid, ltp, p_now) in enumerate(meta_rows):
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

        # stake sizing
        if args.stake_mode == "flat":
            stake = 1.0
        else:
            if side == "back":
                f = kelly_fraction_back(p_pred, ltp, commission)
            else:
                f = kelly_fraction_lay(p_pred, ltp, commission)
            f = max(args.kelly_floor, min(args.kelly_cap, f))
            stake = f * float(args.bankroll_nom)

        exp_pnl = ev_per_1 * stake
        out.append((ts, mid, sid, ltp, p_now, p_pred, dp, side, ev_per_1, stake, exp_pnl,
                    args.stake_mode))

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
            "stake": pl.Float64,
            "exp_pnl": pl.Float64,
            "stake_mode": pl.Utf8,
        },
        orient="row",
    )
    idx = len(list(outdir.glob("trades_part_*.parquet")))
    part.write_parquet(outdir / f"trades_part_{idx:05d}.parquet")

if __name__ == "__main__":
    main()
