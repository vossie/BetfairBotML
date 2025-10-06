#!/usr/bin/env python3
"""
simulate_stream.py — Backtest price-trend model on Betfair-curated data.

Compatible with Polars ≥ 1.0 (no deprecated `clip`, `map_groups`, or streaming args).
"""

import argparse
import polars as pl
import xgboost as xgb
from pathlib import Path
import numpy as np
import json
import sys
from datetime import datetime, timedelta


# ---------------------------------------------------------------------
# Safe clamp (Polars < 1.4 doesn’t support clip(lower, upper))
# ---------------------------------------------------------------------
def dp_to_ev(dp: pl.Expr, ev_scale: float, ev_cap: float) -> pl.Expr:
    """Clamp dp to [-ev_cap, +ev_cap] then scale."""
    x = pl.when(dp > ev_cap).then(pl.lit(ev_cap)).otherwise(dp)
    x = pl.when(x < -ev_cap).then(pl.lit(-ev_cap)).otherwise(x)
    return (x * ev_scale).cast(pl.Float64)


# ---------------------------------------------------------------------
# Load Parquet helper
# ---------------------------------------------------------------------
def load_parquets(path_glob: str, columns: list[str]) -> pl.DataFrame:
    paths = list(Path().glob(path_glob))
    if not paths:
        raise FileNotFoundError(f"No parquet files found for pattern: {path_glob}")
    parts = []
    for p in paths:
        lf = pl.scan_parquet(str(p))
        lf = lf.select([pl.col(c) for c in lf.columns if c in columns])
        parts.append(lf)
    df = pl.concat(parts, how="vertical_relaxed").collect()
    return df


# ---------------------------------------------------------------------
# Feature preparation (safe groupby-apply)
# ---------------------------------------------------------------------
def build_features(df: pl.DataFrame, preoff_max: int, horizon_secs: int) -> pl.DataFrame:
    """Simple derived metrics for testing; extend as needed."""
    # Basic deltas / momentums
    df = df.with_columns([
        (pl.col("ltp").diff().over(["marketId", "selectionId"])).alias("ltp_diff"),
        (pl.col("tradedVolume").diff().over(["marketId", "selectionId"])).alias("vol_diff"),
        (pl.col("ltp").pct_change().over(["marketId", "selectionId"])).alias("ltp_ret"),
    ])
    df = df.fill_null(0)
    df = df.filter(pl.col("ltp") > 0)
    return df


# ---------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------
def run_day(day: str, args, out_dir: Path):
    print(f"[simulate] Processing day {day}")
    date_path = Path(args.curated) / "orderbook_snapshots_5s" / f"sport={args.sport}" / f"date={day}"
    order_cols = ["marketId", "selectionId", "publishTimeMs", "ltp", "tradedVolume"]
    df = load_parquets(str(date_path / "*.parquet"), order_cols)

    df = build_features(df, args.preoff_max, args.horizon_secs)
    if "__dp" not in df.columns:
        df = df.with_columns((pl.col("ltp_ret") * 100).alias("__dp"))

    # EV computation (clamped)
    df = df.with_columns(dp_to_ev(pl.col("__dp"), args.ev_scale, args.ev_cap).alias("ev_per_1"))

    # Simple statistics
    avg_ev = df["ev_per_1"].mean()
    print(f"[simulate] {day} rows={len(df):,} avg_ev/£1={avg_ev:.6f}")

    out_file = out_dir / f"sim_{day}.parquet"
    df.write_parquet(out_file)
    print(f"[simulate] Saved {out_file}")
    return avg_ev


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("simulate_stream (pre-off, grid-join)")
    ap.add_argument("--curated", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--valid-days", type=int, default=7)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--preoff-max", type=int, default=30)
    ap.add_argument("--horizon-secs", type=int, default=120)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    ap.add_argument("--ev-scale", type=float, default=1.0)
    ap.add_argument("--ev-cap", type=float, default=1.0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asof_date = datetime.strptime(args.asof, "%Y-%m-%d").date()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    all_days = [(start_date + timedelta(days=i)).isoformat()
                for i in range((asof_date - start_date).days + 1)][-args.valid_days:]

    results = []
    for d in all_days:
        try:
            ev = run_day(d, args, out_dir)
            results.append((d, ev))
        except Exception as e:
            print(f"[simulate] ERROR {d}: {e}")

    print(f"[simulate] Completed {len(results)} days.")
    for d, ev in results:
        print(f"  {d}: EV={ev:.6f}")


if __name__ == "__main__":
    main()
