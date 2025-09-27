#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import timedelta, timezone
from collections import defaultdict, deque

import numpy as np
import polars as pl
import xgboost as xgb
import pickle
import json

# --- local imports (from your project)
from utils import (
    parse_date, to_ms, read_snapshots,
    implied_prob_from_ltp_expr, time_to_off_minutes_expr, filter_preoff,
    write_json, kelly_fraction_back, kelly_fraction_lay
)

UTC = timezone.utc

# 5s cadence lags (match build_features.py)
LAG_30S = 6
LAG_60S = 12
LAG_120S = 24

NUMERIC_FEATS = [
    "ltp", "tradedVolume", "spreadTicks", "imbalanceBest1", "mins_to_off", "__p_now",
    "ltp_diff_5s", "vol_diff_5s",
    "ltp_lag30s","ltp_lag60s","ltp_lag120s",
    "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
    "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
    "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
]

def parse_args():
    p = argparse.ArgumentParser(description="Streaming simulator (VWAP fills, liquidity, MTM exit + settlement P&L)")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--edge-thresh", type=float, default=0.002)
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--odds-min", type=float, default=None)
    p.add_argument("--odds-max", type=float, default=None)
    # Liquidity / VWAP
    p.add_argument("--liquidity-levels", type=int, default=1, help="sum size across top-L levels")
    p.add_argument("--enforce-liquidity", action="store_true", help="cap stake by available size and price at VWAP")
    # I/O & runtime
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--calib-path", default="")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=200_000)
    return p.parse_args()

# ---------- helpers: detect book columns ----------
def _has_book_columns(df: pl.DataFrame) -> bool:
    cols = set(df.columns)
    need = {"backTicks", "backSizes", "layTicks", "laySizes"}
    return need.issubset(cols)

# ---------- feature builder (inline, minimal) ----------
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
    # sanitize
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = np.nan
    return feats

# ---------- Betfair tick ladder ----------
_TICKS = [
    (1.01, 2.00, 0.01),
    (2.00, 3.00, 0.02),
    (3.00, 4.00, 0.05),
    (4.00, 6.00, 0.10),
    (6.00,10.00, 0.20),
    (10.0,20.00, 0.50),
    (20.0,30.00, 1.00),
    (30.0,50.00, 2.00),
    (50.0,100.0, 5.00),
    (100.0,1000.0,10.00),
]
_cum_ticks = []
cum = 0
for lo, hi, step in _TICKS:
    n = int(round((hi - lo) / step)) + 1
    _cum_ticks.append((lo, hi, step, cum, cum + n - 1))
    cum += n

def _odds_from_tick_index(tick_index: int) -> float:
    if tick_index < 0:
        tick_index = 0
    for lo, hi, step, t0, t1 in _cum_ticks:
        if t0 <= tick_index <= t1:
            k = tick_index - t0
            return round(lo + k * step, 2)
    lo, hi, step, t0, t1 = _cum_ticks[-1]
    k = min(t1, max(t0, tick_index)) - t0
    return round(lo + k * step, 2)

def _vwap_fill(side: str, ticks: list[int], sizes: list[float], need: float, levels: int) -> tuple[float,float]:
    if not ticks or not sizes or need <= 0.0:
        return 0.0, None
    L = max(1, int(levels))
    filled = 0.0
    notional = 0.0
    for i in range(min(L, len(ticks))):
        px = _odds_from_tick_index(int(ticks[i]))
        size_i = float(sizes[i])
        if size_i <= 0:
            continue
        take = min(need - filled, size_i)
        if take <= 0:
            break
        filled += take
        notional += take * px
        if filled >= need - 1e-12:
            break
    if filled <= 0.0:
        return 0.0, None
    return filled, float(notional / filled)

# ---------- EV helpers ----------
def _ev_mtm(p_now: float, p_pred: float, commission: float, side: str) -> float:
    eps = 1e-6
    p_now_c  = max(eps, min(1.0 - eps, p_now))
    p_pred_c = max(eps, min(1.0 - eps, p_pred))
    if side == "back":
        return ((p_pred_c / p_now_c) - 1.0) * (1.0 - commission)
    else:
        return (1.0 - (p_now_c / p_pred_c)) * (1.0 - commission)

def _ev_settlement(p_pred: float, ltp: float, commission: float, side: str) -> float:
    if side == "back":
        return p_pred * (ltp - 1.0) * (1.0 - commission) - (1.0 - p_pred)
    else:
        return (1.0 - p_pred) * (1.0 - commission) - p_pred * (ltp - 1.0)

# ---------- Results loader ----------
def _load_results(curated_root: Path, sport: str, start_dt, end_dt) -> pl.DataFrame:
    days = []
    day = start_dt
    while day <= end_dt:
        days.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)
    lfs = []
    for d in days:
        try:
            lfs.append(pl.scan_parquet(f"{curated_root}/results/sport={sport}/date={d}/*.parquet"))
        except Exception:
            continue
    if not lfs:
        return pl.DataFrame({"marketId": [], "selectionId": [], "winLabel": []})
    lf = pl.concat(lfs, how="vertical_relaxed")
    return (
        lf.select([
            pl.col("marketId").cast(pl.Utf8),
            pl.col("selectionId").cast(pl.Int64),
            pl.col("winLabel").cast(pl.Int32).fill_null(0),
        ]).unique().collect()
    )

def _prepare_exit_prices(df_snap: pl.DataFrame, horizon_secs: int) -> pl.DataFrame:
    """Exit lookup table (asof join). Falls back to LTP-only if no book arrays."""
    select_exprs = [
        pl.col("marketId").cast(pl.Utf8),
        pl.col("selectionId").cast(pl.Int64),
        pl.col("publishTimeMs").cast(pl.Int64).alias("ts"),
        pl.col("ltp").cast(pl.Float64).alias("ltp_exit_proxy"),
    ]
    if _has_book_columns(df_snap):
        select_exprs += [
            pl.col("backTicks"),
            pl.col("backSizes"),
            pl.col("layTicks"),
            pl.col("laySizes"),
        ]
    return df_snap.select(select_exprs).sort(["marketId","selectionId","ts"])

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # NEW: ensure no stale parts from previous schema versions
    for _p in outdir.glob("trades_part_*.parquet"):
        try:
            _p.unlink()
        except Exception:
            pass

    curated = Path(args.curated)

    asof_dt = parse_date(args.asof)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)
    start_dt = parse_date(args.start_date)

    # Load snapshots → filter pre-off → optional odds band
    lf = read_snapshots(curated, start_dt, valid_end, args.sport).with_columns([
        time_to_off_minutes_expr(),
    ])
    df = lf.collect()
    df = filter_preoff(df, args.preoff_max)
    df = df.filter(pl.col("ltp").is_not_null())
    if args.odds_min is not None:
        df = df.filter(pl.col("ltp") >= float(args.odds_min))
    if args.odds_max is not None:
        df = df.filter(pl.col("ltp") <= float(args.odds_max))

    # Book availability & effective liquidity
    book_available = _has_book_columns(df)
    effective_enforce = bool(args.enforce_liquidity and book_available)
    if args.enforce_liquidity and not book_available:
        print("[simulate] NOTE: depth columns (back/lay ticks/sizes) not found; "
              "falling back to LTP exec/exit and disabling liquidity enforcement.")
    print(f"[simulate] Liquidity: requested={'on' if args.enforce_liquidity else 'off'}, "
          f"effective={'on' if effective_enforce else 'off'}")

    # Sort for streaming + build exit table
    df = df.sort(["marketId","selectionId","publishTimeMs"])
    exit_df = _prepare_exit_prices(df, args.horizon_secs)

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

    # rolling buffers per runner
    bufs_ltp = defaultdict(lambda: deque(maxlen=LAG_120S))
    bufs_vol = defaultdict(lambda: deque(maxlen=LAG_120S))

    rows = df.iter_rows(named=True)
    feature_rows, meta_rows = [], []

    # stream over ticks
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
        if feats is None:
            continue

        meta_rows.append((
            r["publishTimeMs"],                    # 0 ts
            r["marketId"],                         # 1
            int(r["selectionId"]),                 # 2
            float(r["ltp"]),                       # 3 ltp_now
            float(feats["__p_now"]),               # 4 p_now
            (r.get("backTicks") or []) if book_available else [],  # 5
            (r.get("backSizes") or []) if book_available else [],  # 6
            (r.get("layTicks") or []) if book_available else [],   # 7
            (r.get("laySizes") or []) if book_available else [],   # 8
        ))
        feature_rows.append([feats.get(c, np.nan) for c in NUMERIC_FEATS])

        if len(feature_rows) >= args.batch_size:
            process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir, effective_enforce)
            feature_rows.clear(); meta_rows.clear()

    if feature_rows:
        process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir, effective_enforce)

    # collect parts
    trades_parts = sorted(outdir.glob("trades_part_*.parquet"))
    if not trades_parts:
        print("[simulate] no trades generated.")
        return

    # target schema (must match what you write in process_batch)
    TARGET_SCHEMA = {
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
        "exec_odds": pl.Float64,
        "exp_pnl": pl.Float64,
        "stake_mode": pl.Utf8,
    }

    aligned = []
    for p in trades_parts:
        dfp = pl.read_parquet(p)
        # add any missing columns as nulls, cast present ones
        for col, dtype in TARGET_SCHEMA.items():
            if col not in dfp.columns:
                dfp = dfp.with_columns(pl.lit(None, dtype=dtype).alias(col))
            else:
                dfp = dfp.with_columns(pl.col(col).cast(dtype, strict=False))
        # drop any extra columns and order consistently
        dfp = dfp.select(list(TARGET_SCHEMA.keys()))
        aligned.append(dfp)

    trades = pl.concat(aligned, how="vertical_relaxed")

    for p in trades_parts: p.unlink(missing_ok=True)

    # EXIT at t+H (asof join, forward)
    trades = trades.with_columns(
        (pl.col("publishTimeMs").cast(pl.Int64) + args.horizon_secs * 1000).alias("ts_exit")
    )
    trades_sorted = trades.sort(["marketId","selectionId","ts_exit"])
    exit_sorted = exit_df  # already sorted
    trades2 = trades_sorted.join_asof(
        exit_sorted,
        left_on="ts_exit", right_on="ts",
        by=["marketId","selectionId"],
        strategy="forward",
        tolerance=timedelta(minutes=10)
    ).with_columns(
        pl.col("ltp_exit_proxy").alias("exit_odds")
    ).drop(["ts","ltp_exit_proxy"])

    # Realised MTM P&L at exit (uses VWAP exec + exit LTP proxy)
    trades2 = trades2.with_columns([
        pl.when((pl.col("side") == "back") & pl.col("exit_odds").is_not_null())
          .then(((pl.col("exec_odds") / pl.col("exit_odds")) - 1.0) * pl.col("stake_filled"))
          .otherwise(
              pl.when((pl.col("side") == "lay") & pl.col("exit_odds").is_not_null())
               .then((1.0 - (pl.col("exit_odds") / pl.col("exec_odds"))) * pl.col("stake_filled"))
               .otherwise(pl.lit(None))
          ).alias("real_mtm_pnl")
    ])

    # Settlement P&L
    results_df = _load_results(curated, args.sport, start_dt, valid_end)
    if results_df.height > 0:
        trades2 = trades2.join(results_df, on=["marketId","selectionId"], how="left") \
                         .with_columns(pl.col("winLabel").fill_null(0).cast(pl.Int32))
        comm = float(args.commission)
        trades2 = trades2.with_columns(
            pl.when(pl.col("side") == "back")
              .then(
                  pl.when(pl.col("winLabel") == 1)
                    .then((pl.col("exec_odds") - 1.0) * pl.col("stake_filled") * (1.0 - comm))
                    .otherwise(-pl.col("stake_filled"))
              )
              .otherwise(
                  pl.when(pl.col("winLabel") == 1)
                    .then(-(pl.col("exec_odds") - 1.0) * pl.col("stake_filled"))
                    .otherwise(pl.col("stake_filled") * (1.0 - comm))
              ).alias("real_settle_pnl")
        )
    else:
        trades2 = trades2.with_columns(pl.lit(None, dtype=pl.Float64).alias("real_settle_pnl"))

    # Write trades CSV
    trades_csv = outdir / f"trades_{args.asof}.csv"
    trades2.write_csv(trades_csv)

    # Daily summary
    daily = (
        trades2
        .with_columns(
            pl.from_epoch(pl.col("publishTimeMs").cast(pl.Int64), time_unit="ms")
            .dt.replace_time_zone("UTC").dt.date().cast(pl.Utf8).alias("day")
        )
        .group_by("day","stake_mode")
        .agg([
            pl.len().alias("n_trades"),
            pl.sum("exp_pnl").alias("exp_profit"),
            pl.mean("ev_per_1").alias("avg_ev"),
            pl.mean("stake_filled").alias("avg_stake_filled"),
            pl.mean("fill_frac").alias("avg_fill_frac"),
            pl.sum("real_mtm_pnl").alias("real_mtm_profit"),
            pl.sum("real_settle_pnl").alias("real_settle_profit"),
        ])
        .with_columns([
            (pl.col("exp_profit")        / pl.lit(float(args.bankroll_nom))).alias("roi_exp"),
            (pl.col("real_mtm_profit")   / pl.lit(float(args.bankroll_nom))).alias("roi_real_mtm"),
            (pl.col("real_settle_profit")/ pl.lit(float(args.bankroll_nom))).alias("roi_real_settle"),
        ])
        .sort("day")
    )
    daily_csv = outdir / f"daily_{args.asof}.csv"
    daily.write_csv(daily_csv)

    # Summary + ROI for the run
    total_exp_profit   = float(trades2["exp_pnl"].sum())
    avg_ev             = float(trades2["ev_per_1"].mean())
    total_real_mtm     = float(trades2["real_mtm_pnl"].sum()) if trades2["real_mtm_pnl"].dtype != pl.Null else None
    total_real_settle  = float(trades2["real_settle_pnl"].sum()) if trades2["real_settle_pnl"].dtype != pl.Null else None

    roi_exp = (total_exp_profit / float(args.bankroll_nom)) if args.bankroll_nom else None
    roi_real_mtm = (total_real_mtm / float(args.bankroll_nom)) if (args.bankroll_nom and total_real_mtm is not None) else None
    roi_real_settle = (total_real_settle / float(args.bankroll_nom)) if (args.bankroll_nom and total_real_settle is not None) else None

    summ = {
        "asof": args.asof,
        "start_date": args.start_date,
        "valid_days": args.valid_days,
        "horizon_secs": args.horizon_secs,
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
        "enforce_liquidity_requested": bool(args.enforce_liquidity),
        "enforce_liquidity_effective": bool(effective_enforce),
        "rows": int(trades2.height),
        "n_trades": int(trades2.height),
        "total_exp_profit": total_exp_profit,
        "avg_ev_per_1": avg_ev,
        "total_real_mtm_profit": total_real_mtm,
        "total_real_settle_profit": total_real_settle,
        "overall_roi_exp": roi_exp,
        "overall_roi_real_mtm": roi_real_mtm,
        "overall_roi_real_settle": roi_real_settle,
    }
    write_json(outdir / f"summary_{args.asof}.json", summ)

    # Console print of ROI for this run
    def _fmt(x):
        return "n/a" if x is None else f"{x:.6f}"
    print(f"[simulate] ROI (exp)        : {_fmt(roi_exp)}")
    print(f"[simulate] ROI (real MTM)   : {_fmt(roi_real_mtm)}")
    print(f"[simulate] ROI (settlement) : {_fmt(roi_real_settle)}")
    print(f"[simulate] Trades: {summ['n_trades']:,}  Avg EV/£1: {avg_ev:.6f}")

def process_batch(feature_rows, meta_rows, bst, calibrator, args, outdir: Path, effective_enforce: bool):
    X = np.asarray(feature_rows, dtype=np.float32)
    dm = xgb.DMatrix(X, feature_names=NUMERIC_FEATS)
    delta_pred = bst.predict(dm)

    out = []
    commission = float(args.commission)
    for i, meta in enumerate(meta_rows):
        ts, mid, sid, ltp_now, p_now, back_ticks, back_sizes, lay_ticks, lay_sizes = meta

        dp = float(delta_pred[i])
        p_pred = p_now + dp
        p_pred = 0.0 if p_pred < 0.0 else (1.0 if p_pred > 1.0 else p_pred)
        if calibrator:
            try:
                p_pred = float(calibrator.transform(np.array([p_pred]))[0])
            except Exception:
                pass

        side = "back" if dp > 0 else ("lay" if dp < 0 else "none")
        if side == "none":
            continue

        # EV per £1
        if args.ev_mode == "mtm":
            ev_per_1 = _ev_mtm(p_now, p_pred, commission, side)
        else:
            ev_per_1 = _ev_settlement(p_pred, ltp_now, commission, side)
        if ev_per_1 < args.edge_thresh:
            continue

        # Requested stake
        if args.stake_mode == "flat":
            stake_req = 1.0
        else:
            if side == "back":
                f = kelly_fraction_back(p_pred, ltp_now, commission)
            else:
                f = kelly_fraction_lay(p_pred, ltp_now, commission)
            f = max(args.kelly_floor, min(args.kelly_cap, f))
            stake_req = f * float(args.bankroll_nom)

        # Liquidity/VWAP if book present and enforcement effective; else LTP & full fill
        has_book = bool(back_ticks or lay_ticks)
        if effective_enforce and has_book:
            avail_sizes = back_sizes if side == "back" else lay_sizes
            avail_ticks = back_ticks if side == "back" else lay_ticks
            stake_filled, exec_vwap = _vwap_fill(side, avail_ticks, avail_sizes, stake_req, args.liquidity_levels)
        else:
            stake_filled = stake_req
            exec_vwap = float(ltp_now)

        if stake_filled <= 0.0 or exec_vwap is None:
            continue

        fill_frac = stake_filled / stake_req if stake_req > 0 else 1.0
        exp_pnl = ev_per_1 * stake_filled

        out.append((
            ts, mid, sid, ltp_now, p_now, p_pred, dp, side, ev_per_1,
            stake_req, stake_filled, fill_frac, float(exec_vwap), exp_pnl, args.stake_mode
        ))

    if not out:
        return
    part = pl.DataFrame(
        out,
        schema={
            "publishTimeMs": pl.Int64,
            "marketId": pl.Utf8,
            "selectionId": pl.Int64,
            "ltp": pl.Float64,           # LTP at entry (info)
            "p_now": pl.Float64,
            "p_pred": pl.Float64,
            "delta_pred": pl.Float64,
            "side": pl.Utf8,
            "ev_per_1": pl.Float64,
            "stake_req": pl.Float64,
            "stake_filled": pl.Float64,
            "fill_frac": pl.Float64,
            "exec_odds": pl.Float64,     # VWAP entry odds or LTP fallback
            "exp_pnl": pl.Float64,
            "stake_mode": pl.Utf8,
        },
        orient="row",
    )
    idx = len(list(outdir.glob("trades_part_*.parquet")))
    part.write_parquet(outdir / f"trades_part_{idx:05d}.parquet")

if __name__ == "__main__":
    main()
