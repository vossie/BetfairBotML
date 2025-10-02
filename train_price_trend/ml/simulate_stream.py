#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast, robust pre-off simulator (per-day, grid-join; no join_asof).

- Processes one day at a time and appends to daily CSV (progress is visible).
- Exit prices via exact equi-join on a 5s grid timestamp (ceil to next grid).
- Uses schema_loader to pin expected columns & avoid schema stalls.
- Liquidity enforcement (L1), odds band, EV scaling/cap, per-market topK,
  basket sizing, exit-on-move ticks (simplified), contra placeholders.

Outputs:
  output/stream/daily_<ASOF>.csv
  output/stream/summary_<ASOF>.json and .csv
"""

from __future__ import annotations
import argparse, json, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import polars as pl

try:
    import xgboost as xgb
except Exception:
    xgb = None

GRID_MS = 5_000  # 5s snapshots

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("simulate_stream (pre-off, grid-join)")
    # data window
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--commission", type=float, default=0.02)

    # device/model
    p.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    p.add_argument("--models-dir", default="/opt/BetfairBotML/train_price_trend/output/models")

    # trading
    p.add_argument("--edge-thresh", type=float, default=0.0)
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="flat")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--ev-scale", type=float, default=1.0)
    p.add_argument("--ev-cap", type=float, default=1.0)

    # odds band
    p.add_argument("--odds-min", type=float, default=None)
    p.add_argument("--odds-max", type=float, default=None)

    # liquidity
    p.add_argument("--enforce-liquidity", action="store_true")
    p.add_argument("--liquidity-levels", type=int, default=1)  # reserved
    p.add_argument("--min-fill-frac", type=float, default=0.0)
    p.add_argument("--require-book", action="store_true")

    # portfolio
    p.add_argument("--per-market-topk", type=int, default=1)
    p.add_argument("--per-market-budget", type=float, default=10.0)
    p.add_argument("--basket-sizing", choices=["equal_stake","prop_ev","all_to_top"], default="equal_stake")
    p.add_argument("--exit-on-move-ticks", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=75_000)  # currently unused; placeholder

    # contra (placeholder)
    p.add_argument("--contra-mode", choices=["none","prob"], default="none")
    p.add_argument("--contra-frac", type=float, default=0.0)
    p.add_argument("--contra-hold-secs", type=int, default=300)
    p.add_argument("--contra-prob-thresh", type=float, default=0.7)
    p.add_argument("--contra-beta", type=float, default=80.0)

    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    return p.parse_args()

# ---------- schema helpers ----------
def load_schema(kind: str) -> list[str]:
    try:
        from ml.schema_loader import load_schema as _load
        return _load(kind)
    except Exception:
        if kind == "orderbook":
            return ["sport","marketId","selectionId","publishTimeMs","ltp","ltpTick",
                    "tradedVolume","spreadTicks","imbalanceBest1","backTicks","backSizes","layTicks","laySizes"]
        if kind == "marketdef":
            return ["sport","marketId","marketStartMs","countryCode"]
        if kind == "results":
            return ["sport","marketId","selectionId","runnerStatus","winLabel"]
        return []

def intersect_existing(requested: list[str], have: list[str]) -> list[str]:
    hs = set(have)
    return [c for c in requested if c in hs]

# ---------- dates / IO ----------
def scan_dates(start_date: str, asof: str) -> List[str]:
    sd = datetime.strptime(start_date, "%Y-%m-%d").date()
    ad = datetime.strptime(asof, "%Y-%m-%d").date()
    return [(sd + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((ad - sd).days + 1)]

def curated_paths(root: str, sport: str, date: str) -> Tuple[Path,Path,Path]:
    base = Path(root)
    return (base / "orderbook_snapshots_5s" / f"sport={sport}" / f"date={date}",
            base / "market_definitions"      / f"sport={sport}" / f"date={date}",
            base / "results"                  / f"sport={sport}" / f"date={date}")

def list_parquets(dirpath: Path) -> list[str]:
    return sorted(dirpath.glob("*.parquet"))

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

def read_day(curated: str, sport: str, date: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    sdir, ddir, _ = curated_paths(curated, sport, date)
    want_order = load_schema("orderbook")
    want_defs  = load_schema("marketdef")

    snaps = pl.DataFrame({})
    defs  = pl.DataFrame({})

    sfiles = list_parquets(sdir)
    if sfiles:
        lf = pl.scan_parquet([str(p) for p in sfiles])
        have = lf.collect_schema().names()
        cols = intersect_existing(want_order, have)
        if cols:
            snaps = collect_streaming(lf.select([pl.col(c) for c in cols]))
    dfiles = list_parquets(ddir)
    if dfiles:
        lf = pl.scan_parquet([str(p) for p in dfiles])
        have = lf.collect_schema().names()
        cols = intersect_existing(want_defs, have)
        if cols:
            defs = collect_streaming(lf.select([pl.col(c) for c in cols]))
    if "marketStartMs" not in defs.columns:
        defs = defs.with_columns(pl.lit(None).alias("marketStartMs"))
    if "countryCode" not in defs.columns:
        defs = defs.with_columns(pl.lit("UNK").alias("countryCode"))
    return snaps, defs

# ---------- features ----------
def build_features(df: pl.DataFrame, defs: pl.DataFrame, preoff_max_m: int, horizon_s: int) -> pl.DataFrame:
    if df.is_empty(): return df
    out = (df.join(defs, on=["sport","marketId"], how="left")
             .with_columns([
                 ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off"),
                 pl.col("ltp").cast(pl.Float64).alias("ltp_f"),
                 pl.col("tradedVolume").cast(pl.Float64).alias("vol_f"),
             ])
             .filter(pl.col("mins_to_off").is_not_null() & (pl.col("mins_to_off") >= 0.0) & (pl.col("mins_to_off") <= float(preoff_max_m)))
             .sort(["marketId","selectionId","publishTimeMs"])
          )
    grp = ["marketId","selectionId"]
    out = out.with_columns([
        pl.col("ltp_f").diff().over(grp).alias("ltp_diff_5s"),
        pl.col("vol_f").diff().over(grp).alias("vol_diff_5s"),
        pl.col("ltp_f").shift(6).over(grp).alias("ltp_lag30s"),
        pl.col("ltp_f").shift(12).over(grp).alias("ltp_lag60s"),
        pl.col("ltp_f").shift(24).over(grp).alias("ltp_lag120s"),
        pl.col("vol_f").shift(6).over(grp).alias("tradedVolume_lag30s"),
        pl.col("vol_f").shift(12).over(grp).alias("tradedVolume_lag60s"),
        pl.col("vol_f").shift(24).over(grp).alias("tradedVolume_lag120s"),
        (pl.col("publishTimeMs") + horizon_s * 1000).alias("ts_exit_ms"),
    ]).with_columns([
        (pl.col("ltp_f") - pl.col("ltp_lag30s")).alias("ltp_mom_30s"),
        (pl.col("ltp_f") - pl.col("ltp_lag60s")).alias("ltp_mom_60s"),
        (pl.col("ltp_f") - pl.col("ltp_lag120s")).alias("ltp_mom_120s"),
        ((pl.col("ltp_f") / pl.col("ltp_lag30s")) - 1.0).alias("ltp_ret_30s"),
        ((pl.col("ltp_f") / pl.col("ltp_lag60s")) - 1.0).alias("ltp_ret_60s"),
        ((pl.col("ltp_f") / pl.col("ltp_lag120s")) - 1.0).alias("ltp_ret_120s"),
    ])
    # ensure optional arrays exist
    for c in ["backTicks","backSizes","layTicks","laySizes","ltpTick"]:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).alias(c))
    return out

# ---------- model / EV ----------
def model_path(models_dir: str) -> Path:
    return Path(models_dir) / "xgb_trend_reg.json"

def predict_dp(df: pl.DataFrame, models_dir: str) -> pl.Series:
    mp = model_path(models_dir)
    if not mp.exists() or xgb is None:
        # proxy if no model: mid-momentum
        proxy = (df["ltp_mom_60s"].fill_null(0.0) + df["ltp_mom_30s"].fill_null(0.0)) / 2.0
        return proxy.fill_null(0.0)
    booster = xgb.Booster(); booster.load_model(str(mp))
    feats = [c for c in [
        "ltp_f","vol_f","ltp_diff_5s","vol_diff_5s",
        "ltp_lag30s","ltp_lag60s","ltp_lag120s",
        "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
        "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
        "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
        "mins_to_off"
    ] if c in df.columns]
    X = df.select(feats).to_pandas()
    dp = booster.predict(xgb.DMatrix(X))
    return pl.Series(dp[:df.height], dtype=pl.Float64)

def dp_to_ev(dp: pl.Series, ev_scale: float, ev_cap: float) -> pl.Series:
    return (dp.clip(min=-ev_cap, max=ev_cap) * ev_scale).cast(pl.Float64)

# ---------- liquidity ----------
def level1_size(row, side: str) -> float:
    try:
        if side == "back":
            sizes = row["backSizes"]
        else:
            sizes = row["laySizes"]
        if sizes and len(sizes) > 0:
            return float(sizes[0] or 0.0)
    except Exception:
        pass
    return 0.0

# ---------- portfolio ----------
def pick_topk(df: pl.DataFrame, k: int) -> pl.DataFrame:
    if df.is_empty(): return df
    return (df.sort([pl.col("marketId"), pl.col("ev_per_1").desc(), pl.col("selectionId")])
              .with_columns(pl.int_range(0, pl.len()).over("marketId").alias("__rk"))
              .filter(pl.col("__rk") < k)
              .drop("__rk"))

def size_basket(df: pl.DataFrame, sizing: str, budget: float, kelly: bool,
                cap: float, floor: float, bankroll: float) -> pl.DataFrame:
    if df.is_empty(): return df
    if kelly:
        # naive Kelly using EV as an edge proxy; capped/floored
        k_frac = (pl.col("ev_per_1").clip(min=-cap, max=cap)).clip(min=floor, max=cap)
        return df.with_columns((k_frac * bankroll).alias("stake_target"))
    if sizing == "equal_stake":
        return df.with_columns((pl.lit(budget) / pl.len()).over("marketId").alias("stake_target"))
    if sizing == "prop_ev":
        return (df.with_columns((pl.col("ev_per_1") / pl.sum("ev_per_1").over("marketId")).alias("__w"))
                  .with_columns((pl.lit(budget) * pl.col("__w").fill_null(0.0)).alias("stake_target"))
                  .drop("__w"))
    # all_to_top
    return (df.with_columns(pl.int_range(0, pl.len()).over("marketId").alias("__rk"))
              .with_columns(pl.when(pl.col("__rk")==0).then(pl.lit(budget)).otherwise(0.0).alias("stake_target"))
              .drop("__rk"))

# ---------- per-day runner ----------
def run_day(date: str, args, out_dir: Path, rollup: dict):
    print(f"[simulate] {date} … loading")
    snaps, defs = read_day(args.curated, args.sport, date)
    if snaps.is_empty():
        print(f"[simulate] {date} … no snapshots")
        return

    df = build_features(snaps, defs, args.preoff_max, args.horizon_secs)

    # odds band
    if args.odds_min is not None:
        df = df.filter(pl.col("ltp_f") >= float(args.odds_min))
    if args.odds_max is not None:
        df = df.filter(pl.col("ltp_f") <= float(args.odds_max))
    if df.is_empty():
        print(f"[simulate] {date} … no rows after odds filter")
        return

    # EV
    dp = predict_dp(df, args.models_dir)
    df = df.with_columns(dp.alias("__dp"))
    df = df.with_columns(dp_to_ev(pl.col("__dp"), args.ev_scale, args.ev_cap).alias("ev_per_1"))

    # candidates
    cands = (df.filter(pl.col("ev_per_1") >= float(args.edge_thresh))
               .select(["marketId","selectionId","publishTimeMs","ltp_f","ltpTick","ev_per_1"]))
    if cands.is_empty():
        print(f"[simulate] {date} … no candidates after edge filter")
        return

    cands = cands.with_columns([
        pl.lit("back").alias("side"),
        pl.col("ltp_f").alias("entry_price"),
        pl.col("ltpTick").alias("entry_tick"),
        (pl.col("publishTimeMs") + args.horizon_secs * 1000).alias("ts_exit_ms"),
        (((pl.col("publishTimeMs") + args.horizon_secs * 1000 + (GRID_MS-1)) // GRID_MS) * GRID_MS).alias("ts_exit_grid_ms"),
    ])

    # per-market selection + sizing
    picks = cands.select(["marketId","selectionId","publishTimeMs","entry_price","entry_tick","ev_per_1"])
    picks = pick_topk(picks, args.per_market_topk)
    kelly = (args.stake_mode == "kelly")
    picks = size_basket(picks, args.basket_sizing, args.per_market_budget, kelly,
                        args.kelly_cap, args.kelly_floor, args.bankroll_nom)

    # liquidity (L1) if depth exists
    have_depth = all(c in df.columns for c in ["backTicks","backSizes","layTicks","laySizes"])
    if args.enforce_liquidity and not have_depth and args.require_book:
        print(f"[simulate] {date} … require_book=on but no depth → dropping day")
        return
    if args.enforce_liquidity and have_depth:
        depth = df.select(["marketId","selectionId","publishTimeMs","backTicks","backSizes","layTicks","laySizes"])
        picks = (picks.join(depth, on=["marketId","selectionId","publishTimeMs"], how="left")
                      .with_columns(pl.struct("backTicks","backSizes","layTicks","laySizes")
                        .apply(lambda s: _level1_size_struct(s, "back")).alias("size_l1"))
                      .with_columns([
                          pl.col("size_l1").fill_null(0.0).alias("size_available"),
                          pl.col("stake_target").fill_null(0.0).alias("stake_target_safe"),
                      ])
                      .with_columns(
                          pl.when(pl.col("stake_target_safe")>0)
                           .then((pl.col("size_available")/pl.col("stake_target_safe")).clip(upper=1.0))
                           .otherwise(0.0)
                           .alias("fill_frac"))
                      .with_columns((pl.col("stake_target_safe") * pl.col("fill_frac")).alias("stake_filled"))
                      .filter(pl.col("stake_filled")>0.0)
               )
    else:
        picks = picks.with_columns([
            pl.col("stake_target").alias("stake_filled"),
            pl.lit(1.0).alias("fill_frac")
        ])
    if picks.is_empty():
        print(f"[simulate] {date} … no trades after liquidity/filters")
        return

    # EXIT prices via grid equi-join
    exits = (snaps.select(["marketId","selectionId","publishTimeMs","ltp"])
                  .rename({"publishTimeMs":"ts_exit_grid_ms","ltp":"exit_price"}))
    picks = (picks.rename({"publishTimeMs":"publishTimeMs_entry"})
                  .with_columns((((pl.col("publishTimeMs_entry") + args.horizon_secs*1000 + (GRID_MS-1)) // GRID_MS) * GRID_MS)
                                 .alias("ts_exit_grid_ms")))

    picks2 = picks.join(exits, on=["marketId","selectionId","ts_exit_grid_ms"], how="left")
    picks2 = picks2.with_columns(pl.col("exit_price").fill_null(pl.col("entry_price")).alias("exit_price_eff"))

    # Exit-on-move ticks (simplified: if tick info missing, fallback to horizon exit)
    # We keep the interface; detailed tick-runner would require ladder & tick tables.
    # Here we leave exit_price_eff as horizon exit price.

    # PnL
    picks2 = picks2.with_columns([
        (pl.col("ev_per_1") * pl.col("stake_filled")).alias("exp_profit"),
        (pl.when((pl.col("entry_price")>0) & pl.col("exit_price_eff").is_not_null())
           .then(pl.col("stake_filled") * (pl.col("exit_price_eff") - pl.col("entry_price")) / pl.col("entry_price"))
           .otherwise(0.0)).alias("real_mtm_pnl")
    ]).with_columns([
        (pl.col("real_mtm_pnl") - (pl.abs(pl.col("real_mtm_pnl")) * float(args.commission))).alias("real_settle_pnl"),
        pl.lit(args.stake_mode).alias("stake_mode"),
        (pl.from_epoch((pl.col("publishTimeMs_entry").cast(pl.Int64)//1000), time_unit="s").dt.date()).alias("__date")
    ])

    bankroll = float(args.bankroll_nom or 1.0)
    day_agg = (picks2.group_by("__date","stake_mode").agg([
                    pl.len().alias("n_trades"),
                    pl.sum("exp_profit").alias("exp_profit"),
                    pl.mean("ev_per_1").alias("avg_ev"),
                    pl.mean("stake_filled").alias("avg_stake_filled"),
                    pl.mean("fill_frac").alias("avg_fill_frac"),
                    pl.sum("real_mtm_pnl").alias("real_mtm_profit"),
                    pl.sum("real_settle_pnl").alias("real_settle_profit"),
               ])
               .with_columns([
                    (pl.col("exp_profit")/bankroll).alias("roi_exp"),
                    (pl.col("real_mtm_profit")/bankroll).alias("roi_real_mtm"),
                    (pl.col("real_settle_profit")/bankroll).alias("roi_real_settle"),
                    pl.col("__date").cast(pl.Utf8).alias("day")
               ])
               .select(["day","stake_mode","n_trades","exp_profit","avg_ev",
                        "avg_stake_filled","avg_fill_frac",
                        "real_mtm_profit","real_settle_profit",
                        "roi_exp","roi_real_mtm","roi_real_settle"])
               .sort("day"))

    daily_csv = out_dir / f"daily_{args.asof}.csv"
    if daily_csv.exists():
        prev = pl.read_csv(daily_csv)
        day_agg = pl.concat([prev, day_agg], how="vertical_relaxed").unique(subset=["day","stake_mode"]).sort("day")
    day_agg.write_csv(daily_csv)

    # rollup
    rollup["n_trades"] += int(picks2.height)
    rollup["total_exp_profit"] += float(picks2["exp_profit"].fill_null(0.0).sum())
    rollup["total_real_mtm_profit"] += float(picks2["real_mtm_pnl"].fill_null(0.0).sum())
    rollup["total_real_settle_profit"] += float(picks2["real_settle_pnl"].fill_null(0.0).sum())
    rollup["ev_sum"] += float(picks2["ev_per_1"].fill_null(0.0).sum())
    rollup["ev_cnt"] += int(picks2.height)

    print(f"[simulate] {date} ✓ trades={picks2.height:,d}  daily_csv={daily_csv}")

def _level1_size_struct(s, side: str) -> float:
    try:
        sizes = s["backSizes"] if side == "back" else s["laySizes"]
        if sizes and len(sizes) > 0:
            return float(sizes[0] or 0.0)
    except Exception:
        pass
    return 0.0

# ---------- main ----------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Streaming Backtest ===")
    print(f"EV threshold:    {args.edge_thresh} per £1")
    print(f"EV mode:         {args.ev_mode} (scale={args.ev_scale}, cap={args.ev_cap})")
    if args.odds_min is not None or args.odds_max is not None:
        print(f"Odds band:       {args.odds_min if args.odds_min is not None else '-'} .. {args.odds_max if args.odds_max is not None else '-'}")
    else:
        print("Odds band:       - .. -")
    print(f"Stake mode:      {args.stake_mode} (cap={args.kelly_cap} floor={args.kelly_floor})  bankroll={args.bankroll_nom}")
    print(f"Liquidity:       enforce={1 if args.enforce_liquidity else 0} levels={args.liquidity_levels} require_book={1 if args.require_book else 0} min_fill_frac={args.min_fill_frac}")
    print(f"Portfolio:       topK={args.per_market_topk} budget={args.per_market_budget} sizing={args.basket_sizing} exit_ticks={args.exit_on_move_ticks}")
    print(f"Sim device:      {args.device}")
    print(f"Output dir:      {out_dir}")

    dates = scan_dates(args.start_date, args.asof)

    rollup = {
        "n_trades": 0,
        "total_exp_profit": 0.0,
        "total_real_mtm_profit": 0.0,
        "total_real_settle_profit": 0.0,
        "ev_sum": 0.0,
        "ev_cnt": 0,
    }
    for d in dates:
        run_day(d, args, out_dir, rollup)

    avg_ev = (rollup["ev_sum"] / rollup["ev_cnt"]) if rollup["ev_cnt"] else 0.0
    bankroll = float(args.bankroll_nom or 1.0)
    roi_exp = rollup["total_exp_profit"] / bankroll
    roi_mtm = rollup["total_real_mtm_profit"] / bankroll
    roi_settle = rollup["total_real_settle_profit"] / bankroll

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
        "enforce_liquidity_effective": bool(args.enforce_liquidity),  # depth check is per-day
        "min_fill_frac": args.min_fill_frac,
        "require_book": bool(args.require_book),
        "per_market_topk": args.per_market_topk,
        "per_market_budget": args.per_market_budget,
        "basket_sizing": args.basket_sizing,
        "exit_on_move_ticks": args.exit_on_move_ticks,
        "ev_scale_used": args.ev_scale,
        "n_trades": rollup["n_trades"],
        "total_exp_profit": rollup["total_exp_profit"],
        "avg_ev_per_1": avg_ev,
        "total_real_mtm_profit": rollup["total_real_mtm_profit"],
        "total_real_settle_profit": rollup["total_real_settle_profit"],
        "overall_roi_exp": roi_exp,
        "overall_roi_real_mtm": roi_mtm,
        "overall_roi_real_settle": roi_settle,
    }

    (out_dir / f"summary_{args.asof}.json").write_text(json.dumps(summary, indent=2))
    pl.DataFrame({k:[v] for k,v in summary.items() if not isinstance(v,(list,dict))}).write_csv(out_dir / f"summary_{args.asof}.csv")

    print(f"[simulate] ROI (exp)        : {roi_exp:.6f}")
    print(f"[simulate] ROI (real MTM)   : {roi_mtm:.6f}")
    print(f"[simulate] ROI (settlement) : {roi_settle:.6f}")
    print(f"[simulate] Trades: {rollup['n_trades']:,d}  Avg EV/£1 (scaled): {avg_ev:.6f}")
    print(f"[simulate] Wrote daily & summary to {out_dir}")

if __name__ == "__main__":
    main()
