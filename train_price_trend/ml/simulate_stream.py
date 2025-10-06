#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_stream.py — lightweight, streaming backtester for the price-trend model.

Key points:
- Single scan_parquet(glob) per day; no per-file loops.
- Streaming collect to avoid RAM spikes.
- Version-safe EV clamping (no Expr.clip).
- Verbose progress logging; optional caps for files/rows.

This version intentionally focuses on robustness/perf of IO. It doesn’t
simulate order book microstructure; it estimates EV from a proxy dp signal.
Integrate with your full trade fill logic later once loading is stable.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import glob
import polars as pl

# -------------------------
# Helpers
# -------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def clamp_to_ev(dp: pl.Expr, ev_scale: float, ev_cap: float) -> pl.Expr:
    # Avoid .clip keyword differences across Polars versions
    x = pl.when(dp > ev_cap).then(pl.lit(ev_cap)).otherwise(dp)
    x = pl.when(x < -ev_cap).then(pl.lit(-ev_cap)).otherwise(x)
    return (x * ev_scale).cast(pl.Float64)

def date_list(start: str, end: str) -> list[str]:
    sd = datetime.strptime(start, "%Y-%m-%d").date()
    ed = datetime.strptime(end, "%Y-%m-%d").date()
    return [(sd + timedelta(days=i)).isoformat() for i in range((ed - sd).days + 1)]

def scan_one_day(glob_path: str, columns: list[str], max_files: int | None) -> pl.LazyFrame | None:
    # Count files quickly for logging and optional cap
    files = sorted(glob.glob(glob_path))
    if not files:
        return None
    if max_files is not None and len(files) > max_files:
        files = files[:max_files]
    # Single lazy scan on either the full glob or the truncated list
    lf = pl.scan_parquet(files if max_files else glob_path)
    # Select only columns that exist
    have = lf.collect_schema().names()
    take = [c for c in columns if c in have]
    if not take:
        return None
    return lf.select([pl.col(c) for c in take])

def add_basic_features(lf: pl.LazyFrame, preoff_max_min: int, horizon_secs: int, row_sample_secs: int | None) -> pl.LazyFrame:
    # marketStartMs may be missing; mins_to_off will be null in that case and filtered out
    lf = lf.with_columns([
        ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off"),
        pl.col("ltp").cast(pl.Float32).alias("ltp_f"),
        pl.col("tradedVolume").cast(pl.Float32).alias("vol_f"),
    ]).filter(
        pl.col("mins_to_off").is_not_null()
        & (pl.col("mins_to_off") >= 0)
        & (pl.col("mins_to_off") <= float(preoff_max_min))
    ).sort(["marketId", "selectionId", "publishTimeMs"])

    # Optional uniform row sampling by time grid to reduce compute
    if row_sample_secs and row_sample_secs > 0:
        step = max(1, int(round(row_sample_secs / 5)))  # base grid=5s
        lf = lf.with_columns(((pl.col("publishTimeMs") // 5000) % step).alias("__mod")).filter(pl.col("__mod") == 0).drop("__mod")

    grp = ["marketId", "selectionId"]
    steps = max(1, int(round(horizon_secs / 5)))

    # Add deltas/momentum and proxy dp (future - now)
    lf = lf.with_columns([
        pl.col("ltp_f").diff().over(grp).alias("ltp_diff_5s"),
        pl.col("vol_f").diff().over(grp).alias("vol_diff_5s"),
        pl.col("ltp_f").shift(6).over(grp).alias("ltp_lag30s"),
        pl.col("ltp_f").shift(12).over(grp).alias("ltp_lag60s"),
        pl.col("ltp_f").shift(24).over(grp).alias("ltp_lag120s"),
    ]).with_columns([
        (pl.col("ltp_f") - pl.col("ltp_lag30s")).alias("ltp_mom_30s"),
        (pl.col("ltp_f") - pl.col("ltp_lag60s")).alias("ltp_mom_60s"),
        (pl.col("ltp_f") - pl.col("ltp_lag120s")).alias("ltp_mom_120s"),
        pl.when(pl.col("ltp_lag30s") > 0).then((pl.col("ltp_f")/pl.col("ltp_lag30s") - 1.0)).otherwise(0.0).alias("ltp_ret_30s"),
        pl.when(pl.col("ltp_lag60s") > 0).then((pl.col("ltp_f")/pl.col("ltp_lag60s") - 1.0)).otherwise(0.0).alias("ltp_ret_60s"),
        pl.when(pl.col("ltp_lag120s") > 0).then((pl.col("ltp_f")/pl.col("ltp_lag120s") - 1.0)).otherwise(0.0).alias("ltp_ret_120s"),
        pl.col("ltp_f").shift(-steps).over(grp).alias("ltp_future"),
    ]).with_columns([
        (pl.col("ltp_future") - pl.col("ltp_f")).alias("__dp")
    ])

    # Keep a compact set to collect
    keep = [
        "marketId","selectionId","publishTimeMs","marketStartMs",
        "ltp_f","vol_f",
        "ltp_diff_5s","vol_diff_5s",
        "ltp_lag30s","ltp_lag60s","ltp_lag120s",
        "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
        "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
        "mins_to_off","ltp_future","__dp"
    ]
    have2 = lf.collect_schema().names()
    take2 = [c for c in keep if c in have2]
    return lf.select([pl.col(c) for c in take2])

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("simulate_stream (pre-off, streaming)")
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

    ap.add_argument("--output-dir", required=True)

    # Perf/control knobs
    ap.add_argument("--max-files-per-day", type=int, default=0, help="cap number of parquet files read per day (0=off)")
    ap.add_argument("--row-sample-secs", type=int, default=0, help="downsample rows by keeping 1 per N seconds (0=off)")
    ap.add_argument("--polars-max-threads", type=int, default=0, help="override POLARS_MAX_THREADS (0=leave)")

    args = ap.parse_args()

    if args.polars_max_threads > 0:
        os.environ["POLARS_MAX_THREADS"] = str(args.polars_max_threads)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asof_d = datetime.strptime(args.asof, "%Y-%m-%d").date()
    days = date_list(args.start_date, args.asof)[-args.valid_days:]

    # What columns we will try to read
    base_cols = [
        "sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume"
    ]
    # defs columns are optional but help filtering to pre-off
    def_cols = ["marketId","marketStartMs"]

    total_days = 0
    ev_acc = []

    for day in days:
        ob_dir = Path(args.curated) / "orderbook_snapshots_5s" / f"sport={args.sport}" / f"date={day}"
        md_dir = Path(args.curated) / "market_definitions"      / f"sport={args.sport}" / f"date={day}"
        ob_glob = str(ob_dir / "*.parquet")
        md_glob = str(md_dir / "*.parquet")

        files = sorted(glob.glob(ob_glob))
        log(f"[simulate] {day} … files={len(files)} (cap={args.max_files_per_day or '∞'})")

        lf_obs = scan_one_day(ob_glob, base_cols, args.max_files_per_day if args.max_files_per_day>0 else None)
        if lf_obs is None:
            log(f"[simulate] {day} … no snapshots; skipping")
            continue

        # defs are optional; if present we’ll join to get marketStartMs
        lf_defs = scan_one_day(md_glob, def_cols, None)
        if lf_defs is not None:
            lf = lf_obs.join(lf_defs, on="marketId", how="left")
        else:
            # if no defs, create a null start time to preserve schema
            lf = lf_obs.with_columns(pl.lit(None).alias("marketStartMs"))

        # Feature add
        lf = add_basic_features(lf, args.preoff_max, args.horizon_secs, args.row_sample_secs)

        # Collect streaming (fallback for older Polars)
        try:
            df = lf.collect(engine="streaming")
        except TypeError:
            df = lf.collect(streaming=True)

        n_rows = df.height
        if n_rows == 0:
            log(f"[simulate] {day} … 0 rows post-filter; skipping")
            continue

        # EV from __dp (proxy)
        df = df.with_columns(clamp_to_ev(pl.col("__dp"), args.ev_scale, args.ev_cap).alias("ev_per_1"))
        avg_ev = float(df["ev_per_1"].mean())
        ev_acc.append(avg_ev)

        # Write a compact per-day parquet to allow further analysis
        out_file = out_dir / f"sim_{day}.parquet"
        df.select([
            "marketId","selectionId","publishTimeMs","mins_to_off","ltp_f","vol_f",
            "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s","ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
            "ltp_future","__dp","ev_per_1"
        ]).write_parquet(out_file)

        log(f"[simulate] {day} … rows={n_rows:,}  avg_ev/£1={avg_ev:.6f}  → {out_file.name}")
        total_days += 1

    log(f"[simulate] Completed {total_days} day(s).")
    if ev_acc:
        log(f"[simulate] Mean EV/£1 over days = {sum(ev_acc)/len(ev_acc):.6f}")


if __name__ == "__main__":
    main()
