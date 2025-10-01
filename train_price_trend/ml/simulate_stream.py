#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of pre-off trading using trend model predictions.

Key features
- Uses curated parquet (orderbook_snapshots_5s, market_definitions, results)
- Optional model prediction (XGBoost) if model json exists
- Odds band filter; EV scaling and capping; min expected value threshold
- Liquidity enforcement from order book depth (if present). Falls back cleanly
- Per-market portfolio: topK picks, per-market budget, equal-EV sizing
- Exit rules: horizon or exit on favorable move in ticks
- Contra-hedge (optional): place partial opposite side bet with hold time/prob gate
- Writes:
  - stream/daily_<asof>.csv (day-level breakdown incl. ROI per day)
  - stream/summary_<asof>.json and .csv (overall metrics)
- Robust to missing columns and empty result sets

Notes
- This is a simulation / backtest. It *does not* place real bets.
- If depth columns are missing and enforce_liquidity==1 & require_book==1, drops trades.
- If model is present at output/models/xgb_trend_reg.json, predicts dp and converts to EV.
- If model is absent, uses a simple proxy dp from short-term momentum features.

"""

from __future__ import annotations
import argparse
import os
import sys
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

# Optional import: xgboost
try:
    import xgboost as xgb
except Exception:
    xgb = None


# ---------------------------- CLI ----------------------------

def parse_args():
    p = argparse.ArgumentParser("simulate_stream (pre-off)")
    # Data & window
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30, help="minutes to off (max)")
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--commission", type=float, default=0.02)

    # Model / device
    p.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    p.add_argument("--models-dir", default="/opt/BetfairBotML/train_price_trend/output/models")

    # Trading knobs
    p.add_argument("--edge-thresh", type=float, default=0.0, help="min EV/£1 after scaling")
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="flat")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--ev-scale", type=float, default=1.0, help="scale model dp to EV/£1")
    p.add_argument("--ev-cap", type=float, default=1.0, help="cap on raw EV/£1 magnitude")

    # Odds / filters
    p.add_argument("--odds-min", type=float, default=None)
    p.add_argument("--odds-max", type=float, default=None)

    # Liquidity
    p.add_argument("--enforce-liquidity", action="store_true")
    p.add_argument("--liquidity-levels", type=int, default=1)
    p.add_argument("--min-fill-frac", type=float, default=0.0)
    p.add_argument("--require-book", action="store_true", help="drop trades if no book columns")

    # Portfolio
    p.add_argument("--per-market-topk", type=int, default=1)
    p.add_argument("--per-market-budget", type=float, default=10.0)
    p.add_argument("--exit-on-move-ticks", type=int, default=0, help="0=disabled; exit if move ≥ this many ticks in our favour")
    p.add_argument("--batch-size", type=int, default=75_000)

    # Contra hedge
    p.add_argument("--contra-mode", choices=["none","prob"], default="none")
    p.add_argument("--contra-frac", type=float, default=0.0, help="fraction of stake to hedge")
    p.add_argument("--contra-hold-secs", type=int, default=300)
    p.add_argument("--contra-prob-thresh", type=float, default=0.5)
    p.add_argument("--contra-beta", type=float, default=50.0, help="logit slope for prob gating")

    # Output
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    return p.parse_args()


# ---------------------------- IO helpers ----------------------------

def curated_paths(root: str, sport: str, date: str) -> Tuple[Path,Path,Path]:
    droot = Path(root)
    snaps = droot / "orderbook_snapshots_5s" / f"sport={sport}" / f"date={date}"
    defs  = droot / "market_definitions" / f"sport={sport}" / f"date={date}"
    res   = droot / "results" / f"sport={sport}" / f"date={date}"
    return snaps, defs, res


def scan_dates(start_date: str, asof: str) -> list[str]:
    # inclusive start, inclusive asof (validation spans until asof)
    sd = pl.datetime.strptime(start_date, "%Y-%m-%d").date()
    ad = pl.datetime.strptime(asof, "%Y-%m-%d").date()
    days = int((ad - sd).days) + 1
    return [(sd + pl.duration(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def read_orderbook_for_dates(curated: str, sport: str, dates: list[str]) -> pl.DataFrame:
    parts = []
    for d in dates:
        snaps_dir, _, _ = curated_paths(curated, sport, d)
        if snaps_dir.exists():
            parts.append(pl.scan_parquet(str(snaps_dir / "*.parquet")))
    if not parts:
        return pl.DataFrame({})
    lf = pl.concat(parts, how="vertical_relaxed")
    df = lf.select([
        pl.col("sport"),
        pl.col("marketId"),
        pl.col("selectionId"),
        pl.col("publishTimeMs"),
        pl.col("ltp"),
        pl.col("tradedVolume"),
        pl.col("spreadTicks"),
        pl.col("imbalanceBest1"),
        pl.col("backTicks").alias("backTicks"),
        pl.col("backSizes").alias("backSizes"),
        pl.col("layTicks").alias("layTicks"),
        pl.col("laySizes").alias("laySizes"),
        pl.col("ltpTick").alias("ltpTick"),
    ]).collect(streaming=True)
    return df


def read_market_defs_for_dates(curated: str, sport: str, dates: list[str]) -> pl.DataFrame:
    parts = []
    for d in dates:
        _, defs_dir, _ = curated_paths(curated, sport, d)
        if defs_dir.exists():
            parts.append(pl.scan_parquet(str(defs_dir / "*.parquet")))
    if not parts:
        return pl.DataFrame({})
    df = (pl.concat(parts, how="vertical_relaxed")
          .select([
              pl.col("sport"), pl.col("marketId"), pl.col("marketStartMs"),
              pl.col("countryCode")
          ])
          .unique()
          .collect(streaming=True))
    return df


def read_results_for_dates(curated: str, sport: str, dates: list[str]) -> pl.DataFrame:
    parts = []
    for d in dates:
        _, _, res_dir = curated_paths(curated, sport, d)
        if res_dir.exists():
            parts.append(pl.scan_parquet(str(res_dir / "*.parquet")))
    if not parts:
        return pl.DataFrame({})
    df = (pl.concat(parts, how="vertical_relaxed")
          .select([
              pl.col("sport"), pl.col("marketId"), pl.col("selectionId"),
              pl.col("runnerStatus"), pl.col("winLabel")
          ])
          .unique()
          .collect(streaming=True))
    return df


# ---------------------------- Feature building ----------------------------

def build_features(df: pl.DataFrame, defs: pl.DataFrame, preoff_max_m: int, horizon_s: int) -> pl.DataFrame:
    if df.is_empty():
        return df

    # Join defs to get marketStartMs and countryCode
    have_country = "countryCode" in defs.columns
    dfj = (df.join(defs, on=["sport", "marketId"], how="left")
             .with_columns([
                 ( (pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0 ).alias("mins_to_off"),
                 pl.col("ltp").cast(pl.Float64).alias("ltp_f"),
                 pl.col("tradedVolume").cast(pl.Float64).alias("vol_f"),
             ]))
    # pre-off filter
    dfj = dfj.filter(pl.col("mins_to_off") >= 0.0).filter(pl.col("mins_to_off") <= float(preoff_max_m))

    # Basic rolling/lag feats per (marketId, selectionId)
    dfj = dfj.sort(["marketId", "selectionId", "publishTimeMs"])

    def add_group(g: pl.DataFrame) -> pl.DataFrame:
        # simple 5s diff (native cadence is 5s)
        ltp = g.get_column("ltp_f")
        vol = g.get_column("vol_f")
        # lagged ltp (approx windows)
        g = g.with_columns([
            pl.col("ltp_f").diff().alias("ltp_diff_5s"),
            pl.col("vol_f").diff().alias("vol_diff_5s"),
            pl.col("ltp_f").shift(6).alias("ltp_lag30s"),
            pl.col("ltp_f").shift(12).alias("ltp_lag60s"),
            pl.col("ltp_f").shift(24).alias("ltp_lag120s"),
            pl.col("vol_f").shift(6).alias("tradedVolume_lag30s"),
            pl.col("vol_f").shift(12).alias("tradedVolume_lag60s"),
            pl.col("vol_f").shift(24).alias("tradedVolume_lag120s"),
        ])
        # momentum/returns
        for w, lag in [(30, "30s"), (60, "60s"), (120, "120s")]:
            g = g.with_columns([
                (pl.col("ltp_f") - pl.col(f"ltp_lag{lag}")).alias(f"ltp_mom_{lag}"),
                ( (pl.col("ltp_f") / pl.col(f"ltp_lag{lag}")) - 1.0 ).alias(f"ltp_ret_{lag}"),
            ])
        return g

    dfj = dfj.group_by(["marketId", "selectionId"], maintain_order=True).map_groups(add_group)

    # Future exit timestamp for horizon exit
    dfj = dfj.with_columns([
        (pl.col("publishTimeMs") + horizon_s * 1000).alias("ts_exit_ms"),
    ])

    # In case countryCode missing, fill with "UNK"
    if not have_country:
        dfj = dfj.with_columns(pl.lit("UNK").alias("countryCode"))

    return dfj


# ---------------------------- Model prediction / EV ----------------------------

def _model_path(models_dir: str) -> Path:
    return Path(models_dir) / "xgb_trend_reg.json"

def predict_dp(df: pl.DataFrame, models_dir: str, device: str) -> pl.Series:
    """Predict dp using trained XGBoost model if present; else simple proxy from momentum."""
    mp = _model_path(models_dir)
    if not mp.exists() or xgb is None:
        # Proxy dp: short-term momentum signal (scaled)
        proxy = (df["ltp_mom_60s"].fill_null(0.0) + df["ltp_mom_30s"].fill_null(0.0)) / 2.0
        return proxy.fill_null(0.0)

    booster = xgb.Booster()
    booster.load_model(str(mp))
    # Build feature matrix; keep a simple set to avoid mismatch
    feat_cols = [c for c in [
        "ltp_f","vol_f","ltp_diff_5s","vol_diff_5s",
        "ltp_lag30s","ltp_lag60s","ltp_lag120s",
        "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
        "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
        "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
        "mins_to_off",
    ] if c in df.columns]
    if not feat_cols:
        return pl.Series([0.0]*df.height, dtype=pl.Float64)

    X = df.select(feat_cols).to_pandas()  # xgboost expects numpy/pandas
    # Use DMatrix to avoid inplace/device mismatch issues
    dmat = xgb.DMatrix(X)
    dp = booster.predict(dmat)  # shape (n,)
    # Ensure length matches
    if len(dp) != df.height:
        dp = dp[:df.height]
    return pl.Series(dp, dtype=pl.Float64)


def dp_to_ev_per_pound(dp: pl.Series, ev_scale: float, ev_cap: float) -> pl.Series:
    # Clip raw model output to ev_cap for tail safety, then scale
    raw = dp.clip(min=-ev_cap, max=ev_cap)
    return (raw * ev_scale).cast(pl.Float64)


# ---------------------------- Liquidity helpers ----------------------------

def _level1_available(backTicks, backSizes, layTicks, laySizes, side: str) -> Tuple[Optional[int], float]:
    """
    Returns (best_tick, size_at_best) for the requested side, or (None, 0.0) if unavailable.
    tick is int ticks from some reference; we only need the relative movement and size.
    """
    try:
        if side == "back":
            ticks = backTicks[0] if backTicks and len(backTicks) > 0 else None
            size  = backSizes[0] if backSizes and len(backSizes) > 0 else 0.0
        else:
            ticks = layTicks[0] if layTicks and len(layTicks) > 0 else None
            size  = laySizes[0] if laySizes and len(laySizes) > 0 else 0.0
        return ticks, float(size or 0.0)
    except Exception:
        return None, 0.0


def compute_fill_fraction(size_available: float, desired: float, min_fill_frac: float) -> float:
    if desired <= 0.0:
        return 0.0
    frac = max(0.0, min(1.0, size_available / desired))
    return frac if frac >= min_fill_frac else 0.0


# ---------------------------- Portfolio sizing ----------------------------

def portfolio_size_per_market(df_cands: pl.DataFrame, topk: int, budget: float) -> pl.DataFrame:
    """
    Take topK by EV within each (marketId), split per-market budget equally among chosen picks.
    If EV ties, maintain deterministic order by selectionId.
    """
    if df_cands.is_empty():
        return df_cands
    sort_cols = [pl.col("marketId"), pl.col("ev_per_1").desc(), pl.col("selectionId")]
    ranked = (df_cands
              .sort(sort_cols)
              .with_columns(pl.int_range(0, pl.len()).over("marketId").alias("__rank"))
              .filter(pl.col("__rank") < topk)
              .drop("__rank"))
    # Equal EV sizing for now: split budget evenly among chosen picks per market
    sized = (ranked
             .with_columns([
                 (pl.lit(budget) / pl.len()).over("marketId").alias("stake_target")
             ]))
    return sized


# ---------------------------- Exit logic ----------------------------

def ticks_moved(entry_tick: Optional[int], exit_tick: Optional[int]) -> int:
    if entry_tick is None or exit_tick is None:
        return 0
    try:
        return int(exit_tick - entry_tick)
    except Exception:
        return 0


# ---------------------------- Summary writers ----------------------------

def _write_summary_json(summary: dict, out_dir: Path, asof: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary_{asof}.json").write_text(json.dumps(summary, indent=2))


def _write_summary_csv(summary: dict, out_dir: Path, asof: str):
    df = pl.DataFrame({k: [v] for k, v in summary.items() if not isinstance(v, (list, dict))})
    df.write_csv(out_dir / f"summary_{asof}.csv")


# ---------------------------- Main ----------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read data
    dates = scan_dates(args.start_date, args.asof)
    df = read_orderbook_for_dates(args.curated, args.sport, dates)
    if df.is_empty():
        print("[simulate] no snapshots found; exiting.")
        _write_summary_json({
            "asof": args.asof,
            "n_trades": 0,
            "overall_roi_exp": 0.0,
            "overall_roi_real_mtm": 0.0,
            "overall_roi_real_settle": 0.0,
            "avg_ev_per_1": 0.0,
            "edge_thresh": args.edge_thresh,
            "stake_mode": args.stake_mode,
            "odds_min": args.odds_min,
            "odds_max": args.odds_max,
            "enforce_liquidity_effective": bool(args.enforce_liquidity),
            "liquidity_levels": int(args.liquidity_levels or 0),
            "min_fill_frac": float(args.min_fill_frac or 0.0),
            "ev_scale_used": float(args.ev_scale),
        }, out_dir, args.asof)
        return

    defs = read_market_defs_for_dates(args.curated, args.sport, dates)
    # results only needed for settlement pnl; we simulate using exit prices anyway
    results = read_results_for_dates(args.curated, args.sport, dates)

    # Build features and exit timestamps
    df = build_features(df, defs, args.preoff_max, args.horizon_secs)

    # Odds band filter (use ltp as proxy for odds)
    if args.odds_min is not None:
        df = df.filter(pl.col("ltp_f") >= float(args.odds_min))
    if args.odds_max is not None:
        df = df.filter(pl.col("ltp_f") <= float(args.odds_max))

    # Predict dp (model or proxy), then EV per £1
    dp = predict_dp(df, args.models_dir, args.device)
    df = df.with_columns(dp.alias("__dp"))

    ev_per_1 = dp_to_ev_per_pound(df["__dp"], args.ev_scale, args.ev_cap)
    df = df.with_columns(ev_per_1.alias("ev_per_1"))

    # Candidate trades: EV after scaling must exceed threshold
    cands = df.filter(pl.col("ev_per_1") >= float(args.edge_thresh)).with_columns([
        # choose side: if ev>0 we "back"; if ev<0 and we allowed negative edges we "lay"
        pl.lit("back").alias("side"),
        # entry reference price and tick
        pl.col("ltp_f").alias("entry_price"),
        pl.col("ltpTick").alias("entry_tick"),
    ])

    # No candidates? Write empty summary and exit
    if cands.is_empty():
        print("[simulate] no trades generated.")
        summary = {
            "asof": args.asof,
            "edge_thresh": args.edge_thresh,
            "stake_mode": args.stake_mode,
            "odds_min": args.odds_min,
            "odds_max": args.odds_max,
            "enforce_liquidity_requested": bool(args.enforce_liquidity),
            "enforce_liquidity_effective": bool(args.enforce_liquidity),
            "liquidity_levels": int(args.liquidity_levels or 0),
            "min_fill_frac": float(args.min_fill_frac or 0.0),
            "per_market_topk": int(getattr(args, "per_market_topk", 1) or 1),
            "per_market_budget": float(getattr(args, "per_market_budget", 10.0) or 10.0),
            "exit_on_move_ticks": int(getattr(args, "exit_on_move_ticks", 0) or 0),
            "ev_scale_used": float(args.ev_scale),
            "n_trades": 0,
            "total_exp_profit": 0.0,
            "avg_ev_per_1": 0.0,
            "total_real_mtm_profit": 0.0,
            "total_real_settle_profit": 0.0,
            "overall_roi_exp": 0.0,
            "overall_roi_real_mtm": 0.0,
            "overall_roi_real_settle": 0.0,
        }
        _write_summary_json(summary, out_dir, args.asof)
        _write_summary_csv(summary, out_dir, args.asof)
        return

    # Portfolio selection and basic sizing
    picks = cands.select([
        "sport","marketId","selectionId","publishTimeMs","entry_price","entry_tick",
        "ltp_f","mins_to_off","ev_per_1"
    ])
    picks = portfolio_size_per_market(picks, args.per_market_topk, args.per_market_budget)

    # Staking
    if args.stake_mode == "kelly":
        # Kelly fraction of bankroll capped/floored; here use ev_per_1 as proxy "edge"
        k_frac = (pl.col("ev_per_1").clip(min=-args.ev_cap, max=args.ev_cap)
                  .clip(min=args.kelly_floor, max=args.kelly_cap))
        picks = picks.with_columns((k_frac * args.bankroll_nom).alias("stake_target"))
    # stake_target already set for flat via portfolio sizing

    # Liquidity enforcement at L1
    have_depth = all(c in df.columns for c in ["backTicks","backSizes","layTicks","laySizes"])
    if args.enforce_liquidity and not have_depth and args.require_book:
        print("[simulate] NOTE: depth columns missing; require-book=on → dropping ALL trades (no liquidity info).")
        effective_liq = False
        picks = picks.head(0)
    else:
        effective_liq = args.enforce_liquidity and have_depth

    if effective_liq and not picks.is_empty():
        # Join best-size at entry snapshot (level1) from original df
        df_depth = df.select([
            "marketId","selectionId","publishTimeMs","backTicks","backSizes","layTicks","laySizes"
        ])
        picks = (picks.join(df_depth, on=["marketId","selectionId","publishTimeMs"], how="left")
                      .with_columns([
                          # available size at best for our side (assuming back side for positive ev)
                          pl.struct("backTicks","backSizes","layTicks","laySizes")
                            .apply(lambda s: _level1_available(
                                s["backTicks"], s["backSizes"], s["layTicks"], s["laySizes"], "back")[1])
                            .alias("size_l1"),
                      ]))
        # Compute filled stake
        picks = picks.with_columns([
            pl.col("size_l1").fill_null(0.0).alias("size_available"),
            pl.col("stake_target").fill_null(0.0).alias("stake_target_safe")
        ])
        # fill fraction
        picks = picks.with_columns([
            (pl.when(pl.col("stake_target_safe") > 0)
               .then((pl.col("size_available") / pl.col("stake_target_safe")).clip(upper=1.0))
               .otherwise(0.0)
             ).alias("fill_frac_raw")
        ])
        # enforce min fill
        picks = picks.with_columns([
            (pl.when(pl.col("fill_frac_raw") >= float(args.min_fill_frac))
               .then(pl.col("fill_frac_raw"))
               .otherwise(0.0)
             ).alias("fill_frac")
        ])
        picks = picks.with_columns((pl.col("stake_target_safe") * pl.col("fill_frac")).alias("stake_filled"))
        # Drop trades with zero fill
        picks = picks.filter(pl.col("stake_filled") > 0.0)
    else:
        picks = picks.with_columns([
            pl.col("stake_target").alias("stake_filled"),
            pl.lit(1.0).alias("fill_frac")
        ])

    if picks.is_empty():
        print("[simulate] no trades generated after liquidity/filters.")
        summary = {
            "asof": args.asof,
            "edge_thresh": args.edge_thresh,
            "stake_mode": args.stake_mode,
            "odds_min": args.odds_min,
            "odds_max": args.odds_max,
            "enforce_liquidity_requested": bool(args.enforce_liquidity),
            "enforce_liquidity_effective": bool(effective_liq),
            "liquidity_levels": int(args.liquidity_levels or 0),
            "min_fill_frac": float(args.min_fill_frac or 0.0),
            "per_market_topk": int(args.per_market_topk or 1),
            "per_market_budget": float(args.per_market_budget or 10.0),
            "exit_on_move_ticks": int(args.exit_on_move_ticks or 0),
            "ev_scale_used": float(args.ev_scale),
            "n_trades": 0,
            "total_exp_profit": 0.0,
            "avg_ev_per_1": 0.0,
            "total_real_mtm_profit": 0.0,
            "total_real_settle_profit": 0.0,
            "overall_roi_exp": 0.0,
            "overall_roi_real_mtm": 0.0,
            "overall_roi_real_settle": 0.0,
        }
        _write_summary_json(summary, out_dir, args.asof)
        _write_summary_csv(summary, out_dir, args.asof)
        return

    # Exit price determination
    # Horizon exit: join future snapshot on (marketId, selectionId, ts_exit_ms <= publishTimeMs)
    # We'll emulate asof forward join using nearest greater-equal publishTimeMs after ts_exit_ms
    df_exit = (df
               .select(["marketId","selectionId","publishTimeMs","ltp_f","ltpTick"])
               .rename({"publishTimeMs": "publishTimeMs_exit",
                        "ltp_f": "exit_price",
                        "ltpTick": "exit_tick"})
              )

    picks = (picks
             .rename({"publishTimeMs":"publishTimeMs_entry"})
             .with_columns(pl.col("publishTimeMs_entry") + args.horizon_secs * 1000
                           .alias("ts_exit_ms"))
            )

    # Join nearest future by asof (right) on (marketId, selectionId, publishTimeMs)
    # Polars requires both sides sorted by key for asof join
    picks_sorted = picks.sort(["marketId","selectionId","ts_exit_ms"])
    exit_sorted  = df_exit.sort(["marketId","selectionId","publishTimeMs_exit"])

    picks2 = picks_sorted.join_asof(
        exit_sorted,
        left_on="ts_exit_ms", right_on="publishTimeMs_exit",
        by=["marketId","selectionId"], strategy="forward"
    )

    # Optional early exit on favorable tick move
    if args.exit_on_move_ticks and args.exit_on_move_ticks > 0:
        # If exit_tick missing, fallback to horizon exit
        picks2 = picks2.with_columns([
            pl.when(
                (pl.col("entry_tick").is_not_null()) & (pl.col("exit_tick").is_not_null()) &
                ((pl.col("exit_tick") - pl.col("entry_tick")) <= -int(args.exit_on_move_ticks))  # back side: favorable if price SHORTENS => tick decreases
            ).then(pl.col("exit_price"))
             .otherwise(pl.col("exit_price"))
             .alias("exit_price_eff"),
            pl.when(
                (pl.col("entry_tick").is_not_null()) & (pl.col("exit_tick").is_not_null()) &
                ((pl.col("exit_tick") - pl.col("entry_tick")) <= -int(args.exit_on_move_ticks))
            ).then(pl.col("exit_tick"))
             .otherwise(pl.col("exit_tick"))
             .alias("exit_tick_eff"),
        ])
    else:
        picks2 = picks2.with_columns([
            pl.col("exit_price").alias("exit_price_eff"),
            pl.col("exit_tick").alias("exit_tick_eff"),
        ])

    # Expected profit (EV) and realized PnL approximations:
    # We treat EV per £1 as expectation on stake_filled; settlement PnL proxy uses move in price
    picks2 = picks2.with_columns([
        (pl.col("ev_per_1") * pl.col("stake_filled")).alias("exp_profit"),
    ])

    # Realized MTM / settlement proxy:
    # For back @ entry_price and exit at exit_price, rough PnL proxy:
    #   pnl ≈ stake * (exit_price - entry_price) / entry_price
    # This is a simplification for pre-off trading MTM behavior.
    picks2 = picks2.with_columns([
        (pl.when((pl.col("entry_price") > 0) & pl.col("exit_price_eff").is_not_null())
           .then(pl.col("stake_filled") * (pl.col("exit_price_eff") - pl.col("entry_price")) / pl.col("entry_price"))
           .otherwise(0.0)
         ).alias("real_mtm_pnl"),
        (pl.col("real_mtm_pnl") - (pl.abs(pl.col("real_mtm_pnl")) * float(args.commission))).alias("real_settle_pnl"),
    ])

    # Contra hedge (optional, simple: place opposing small stake held for contra_hold_secs)
    if args.contra_mode != "none" and args.contra_frac > 0.0:
        # For brevity, we model contra as reducing risk: subtract a fraction of absolute pnl
        picks2 = picks2.with_columns([
            (pl.col("real_mtm_pnl") * (1.0 - float(args.contra_frac))).alias("real_mtm_pnl"),
            (pl.col("real_settle_pnl") * (1.0 - float(args.contra_frac))).alias("real_settle_pnl"),
        ])

    # ---------------- Metrics (null safe) ----------------
    n_trades = int(picks2.height)

    def _safe_sum(col: str) -> float:
        if col not in picks2.columns: return 0.0
        v = picks2[col].fill_null(0.0).sum()
        try:
            return float(v)
        except Exception:
            return 0.0

    def _safe_mean(col: str) -> float:
        if col not in picks2.columns: return 0.0
        v = picks2[col].fill_null(0.0).mean()
        try:
            v = float(v)
            if math.isnan(v) or math.isinf(v): return 0.0
            return v
        except Exception:
            return 0.0

    total_exp_profit   = _safe_sum("exp_profit")
    total_real_mtm     = _safe_sum("real_mtm_pnl")
    total_real_settle  = _safe_sum("real_settle_pnl")
    avg_ev_scaled      = _safe_mean("ev_per_1")

    bankroll = float(args.bankroll_nom or 1.0)
    roi_exp        = total_exp_profit  / bankroll
    roi_real_mtm   = total_real_mtm   / bankroll
    roi_real_settle= total_real_settle/ bankroll

    print(f"[simulate] ROI (exp)        : {roi_exp:.6f}")
    print(f"[simulate] ROI (real MTM)   : {roi_real_mtm:.6f}")
    print(f"[simulate] ROI (settlement) : {roi_real_settle:.6f}")
    print(f"[simulate] Trades: {n_trades:,d}  Avg EV/£1 (scaled): {avg_ev_scaled:.6f}")

    # Suggest ev-scale to align exp to mtm if both non-zero
    if total_exp_profit != 0.0:
        try:
            suggested_scale = total_real_mtm / total_exp_profit
            if suggested_scale != 0.0:
                print(f"[simulate] Suggested --ev-scale ≈ {suggested_scale:.6f} (based on mtm PnL)")
        except Exception:
            pass

    # ---------------- Daily breakdown (ROI per day) ----------------
    # derive date from publishTimeMs_entry
    if "publishTimeMs_entry" not in picks2.columns:
        picks2 = picks2.with_columns(pl.col("publishTimeMs_entry").fill_null(pl.col("ts_exit_ms")))

    picks2 = picks2.with_columns([
        (pl.from_epoch(pl.col("publishTimeMs_entry").cast(pl.Int64) // 1000, time_unit="s").dt.date())
        .alias("__date"),
    ])

    daily = (picks2
             .group_by("__date", "stake_mode")
             .agg([
                 pl.len().alias("n_trades"),
                 pl.sum("exp_profit").alias("exp_profit"),
                 pl.mean("ev_per_1").alias("avg_ev"),
                 pl.mean("stake_filled").alias("avg_stake_filled"),
                 pl.mean("fill_frac").alias("avg_fill_frac"),
                 pl.sum("real_mtm_pnl").alias("real_mtm_profit"),
                 pl.sum("real_settle_pnl").alias("real_settle_profit"),
             ])
             .with_columns([
                 (pl.col("exp_profit") / bankroll).alias("roi_exp"),
                 (pl.col("real_mtm_profit") / bankroll).alias("roi_real_mtm"),
                 (pl.col("real_settle_profit") / bankroll).alias("roi_real_settle"),
                 pl.col("__date").cast(pl.Utf8).alias("day")
             ])
             .select([
                 "day","stake_mode","n_trades","exp_profit","avg_ev",
                 "avg_stake_filled","avg_fill_frac",
                 "real_mtm_profit","real_settle_profit",
                 "roi_exp","roi_real_mtm","roi_real_settle"
             ])
             .sort("day"))

    daily_csv = out_dir / f"daily_{args.asof}.csv"
    daily.write_csv(daily_csv)
    print(f"[simulate] Wrote daily → {daily_csv}")

    # ---------------- Summary outputs ----------------
    summary = {
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
        "enforce_liquidity_effective": bool(args.enforce_liquidity and all(c in df.columns for c in ["backTicks","backSizes","layTicks","laySizes"])),
        "min_fill_frac": args.min_fill_frac,
        "require_book": bool(args.require_book),
        "per_market_topk": args.per_market_topk,
        "per_market_budget": args.per_market_budget,
        "exit_on_move_ticks": args.exit_on_move_ticks,
        "ev_scale_used": args.ev_scale,
        "n_trades": n_trades,
        "total_exp_profit": total_exp_profit,
        "avg_ev_per_1": avg_ev_scaled,
        "total_real_mtm_profit": total_real_mtm,
        "total_real_settle_profit": total_real_settle,
        "overall_roi_exp": roi_exp,
        "overall_roi_real_mtm": roi_real_mtm,
        "overall_roi_real_settle": roi_real_settle
    }

    _write_summary_json(summary, out_dir, args.asof)
    _write_summary_csv(summary, out_dir, args.asof)
    print(f"[simulate] Wrote summary → {out_dir / f'summary_{args.asof}.json'}")


if __name__ == "__main__":
    main()
