#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_stream.py — streaming backtester scaffold for price-trend model.

Fixes:
- Join market definitions over a DATE RANGE (not same-day) to get marketStartMs.
- Streaming single-scan parquet IO (no per-file loops).
- Version-safe EV clamping (no Expr.clip).
- Defensive knobs for IO/rows/threads.

This file intentionally focuses on robust IO + pre-off filtering. Plug your full
fill/portfolio logic on top once loading is stable.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import glob
import polars as pl

# -------------------------
# Small utilities
# -------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def parse_date(s: str) -> datetime.date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def date_iter(start: str, end: str):
    sd, ed = parse_date(start), parse_date(end)
    d = sd
    while d <= ed:
        yield d.isoformat()
        d += timedelta(days=1)

def clamp_to_ev(dp: pl.Expr, ev_scale: float, ev_cap: float) -> pl.Expr:
    # No .clip(...) to avoid Polars keyword differences
    x = pl.when(dp > ev_cap).then(pl.lit(ev_cap)).otherwise(dp)
    x = pl.when(x < -ev_cap).then(pl.lit(-ev_cap)).otherwise(x)
    return (x * ev_scale).cast(pl.Float64)

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

# -------------------------
# Scanners
# -------------------------

def scan_obs_glob(glob_path: str, want_cols: list[str], max_files: int | None) -> pl.LazyFrame | None:
    files = sorted(glob.glob(glob_path))
    if not files:
        return None
    if max_files is not None and max_files > 0 and len(files) > max_files:
        files = files[:max_files]
    lf = pl.scan_parquet(files)
    have = lf.collect_schema().names()
    take = [c for c in want_cols if c in have]
    if not take:
        return None
    return lf.select([pl.col(c) for c in take])

def scan_defs_range(curated: str, sport: str, start_date: str, end_date: str) -> pl.LazyFrame | None:
    """Scan market_definitions for [start_date .. end_date] and reduce to latest per marketId."""
    base = Path(curated) / "market_definitions" / f"sport={sport}"
    patterns = [str(base / f"date={d}" / "*.parquet") for d in date_iter(start_date, end_date)]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        return None
    lf = pl.scan_parquet(files).select([
        pl.col("marketId"),
        pl.col("marketStartMs"),
        pl.col("publishTimeMs").alias("__def_pub")
    ])
    # Reduce to latest definition per marketId (max publishTimeMs)
    lf = (lf
          .filter(pl.col("marketId").is_not_null())
          .group_by("marketId")
          .agg([
              pl.col("marketStartMs").last().alias("marketStartMs"),  # if multiple rows same publish, last ok
              pl.col("__def_pub").max().alias("__def_pub_max")
          ])
          .drop("__def_pub_max"))
    return lf

# -------------------------
# Feature builder
# -------------------------

def add_basic_features(lf: pl.LazyFrame, preoff_max_min: int, horizon_secs: int, row_sample_secs: int | None) -> pl.LazyFrame:
    # Sample down by time-grid if requested (base grid = 5s)
    if row_sample_secs and row_sample_secs > 0:
        step = max(1, int(round(row_sample_secs / 5)))
        lf = lf.with_columns(((pl.col("publishTimeMs") // 5000) % step).alias("__mod")).filter(pl.col("__mod") == 0).drop("__mod")

    # Pre-off mins and sorting
    lf = (lf
          .with_columns([
              ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off"),
              pl.col("ltp").cast(pl.Float32).alias("ltp_f"),
              pl.col("tradedVolume").cast(pl.Float32).alias("vol_f")
          ])
          .filter(
              pl.col("marketStartMs").is_not_null()
              & (pl.col("mins_to_off") >= 0)
              & (pl.col("mins_to_off") <= float(preoff_max_min))
          )
          .sort(["marketId", "selectionId", "publishTimeMs"])
          )

    grp = ["marketId", "selectionId"]
    steps = max(1, int(round(horizon_secs / 5)))

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

    keep = [
        "marketId","selectionId","publishTimeMs","marketStartMs","mins_to_off",
        "ltp_f","vol_f",
        "ltp_diff_5s","vol_diff_5s",
        "ltp_lag30s","ltp_lag60s","ltp_lag120s",
        "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
        "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
        "ltp_future","__dp"
    ]
    have = lf.collect_schema().names()
    take = [c for c in keep if c in have]
    return lf.select([pl.col(c) for c in take])

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("simulate_stream (pre-off, streaming, defs-range)")
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

    # IO/perf knobs
    ap.add_argument("--max-files-per-day", type=int, default=0, help="cap #parquet files for snapshots per day (0=off)")
    ap.add_argument("--row-sample-secs", type=int, default=0, help="downsample rows by time (0=off)")
    ap.add_argument("--polars-max-threads", type=int, default=0, help="override POLARS_MAX_THREADS (0=leave)")

    # NEW: definition scan control
    ap.add_argument("--defs-days-back", type=int, default=21, help="scan market_definitions this many days BEFORE start-date")
    ap.add_argument("--defs-days-forward", type=int, default=7, help="scan market_definitions this many days AFTER asof")
    ap.add_argument("--allow-missing-defs", action="store_true", help="if no defs found, skip pre-off filter (dangerous but useful for debugging)")

    args = ap.parse_args()

    if args.polars_max_threads > 0:
        os.environ["POLARS_MAX_THREADS"] = str(args.polars_max_threads)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of validation days
    asof_d = parse_date(args.asof)
    all_days = list(date_iter(args.start_date, args.asof))
    days = all_days[-args.valid_days:]

    # ---- pre-scan market definitions over a range ----
    defs_start = (parse_date(args.start_date) - timedelta(days=args.defs_days_back)).isoformat()
    defs_end   = (parse_date(args.asof) + timedelta(days=args.defs_days_forward)).isoformat()
    log(f"[simulate] scanning definitions {defs_start} .. {defs_end}")
    lf_defs = scan_defs_range(args.curated, args.sport, defs_start, defs_end)

    # If defs absent but allowed → create stub with nulls (skip pre-off filter)
    if lf_defs is None:
        msg = "[simulate] WARNING: no market_definitions found in range."
        if args.allow_missing_defs:
            log(msg + " allow-missing-defs=on → proceeding WITHOUT pre-off filter.")
        else:
            log(msg + " Try increasing --defs-days-back/forward or pass --allow-missing-defs.")
            return

    # ---- process each day ----
    base_cols = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume"]
    total_days = 0
    mean_evs = []

    for day in days:
        ob_dir = Path(args.curated) / "orderbook_snapshots_5s" / f"sport={args.sport}" / f"date={day}"
        ob_glob = str(ob_dir / "*.parquet")
        n_files = len(glob.glob(ob_glob))
        log(f"[simulate] {day} … files={n_files} (cap={args.max_files_per_day or '∞'})")

        lf_obs = scan_obs_glob(ob_glob, base_cols, args.max_files_per_day if args.max_files_per_day>0 else None)
        if lf_obs is None:
            log(f"[simulate] {day} … no snapshots; skipping")
            continue

        # Join with definitions (if available)
        if lf_defs is not None:
            lf = lf_obs.join(lf_defs, on="marketId", how="left")
        else:
            lf = lf_obs.with_columns(pl.lit(None).alias("marketStartMs"))

        # Features + filter
        lf = add_basic_features(lf, args.preoff_max, args.horizon_secs, args.row_sample_secs)

        # If defs missing and not allowed, skip days with null start time
        if lf_defs is None and not args.allow_missing_defs:
            log(f"[simulate] {day} … defs missing and allow-missing-defs=off → skipped")
            continue

        # Collect (streaming)
        df = collect_streaming(lf)
        if df.height == 0:
            log(f"[simulate] {day} … 0 rows post-filter; skipping")
            continue

        # EV proxy
        df = df.with_columns(clamp_to_ev(pl.col("__dp"), args.ev_scale, args.ev_cap).alias("ev_per_1"))
        avg_ev = float(df["ev_per_1"].mean())
        mean_evs.append(avg_ev)

        # Persist compact daily slice
        out_file = out_dir / f"sim_{day}.parquet"
        df.select([
            "marketId","selectionId","publishTimeMs","marketStartMs","mins_to_off",
            "ltp_f","vol_f",
            "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
            "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
            "ltp_future","__dp","ev_per_1"
        ]).write_parquet(out_file)

        log(f"[simulate] {day} … rows={df.height:,}  avg_ev/£1={avg_ev:.6f} → {out_file.name}")
        total_days += 1

    log(f"[simulate] Completed {total_days} day(s).")
    if mean_evs:
        log(f"[simulate] Mean EV/£1 over days = {sum(mean_evs)/len(mean_evs):.6f}")


if __name__ == "__main__":
    main()
