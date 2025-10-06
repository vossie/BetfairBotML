#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_stream.py — streaming backtester scaffold for price-trend model.

Fixes/Features:
- Joins market definitions over a date RANGE.
- Per-day diagnostics: files, rows, join coverage (non-null marketStartMs).
- Auto-fallback to skip pre-off filtering if join coverage too low.
- Streaming IO, Polars-version-safe EV clamping (no Expr.clip).

This is an IO + filtering scaffold. Plug your full fill/portfolio logic on top.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import glob
import polars as pl


# -------------------------
# Utilities
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
    # No .clip(...) to avoid Polars keyword changes
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

def scan_defs_range(curated: str, sport: str, start_date: str, end_date: str) -> tuple[pl.LazyFrame | None, int]:
    """Scan market_definitions for [start_date .. end_date] and reduce to latest per marketId."""
    base = Path(curated) / "market_definitions" / f"sport={sport}"
    patterns = [str(base / f"date={d}" / "*.parquet") for d in date_iter(start_date, end_date)]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        return None, 0
    lf = pl.scan_parquet(files).select([
        pl.col("marketId"),
        pl.col("marketStartMs"),
        pl.col("publishTimeMs").alias("__def_pub")
    ])
    lf = (
        lf.filter(pl.col("marketId").is_not_null())
          .group_by("marketId")
          .agg([
              # take last marketStartMs; we only need one (the latest def wins)
              pl.col("marketStartMs").last().alias("marketStartMs"),
              pl.col("__def_pub").max().alias("__def_pub_max")
          ])
          .drop("__def_pub_max")
    )
    return lf, len(files)


# -------------------------
# Feature builder
# -------------------------

def add_basic_features(
    lf: pl.LazyFrame,
    preoff_max_min: int,
    horizon_secs: int,
    row_sample_secs: int | None,
    apply_preoff_filter: bool
) -> pl.LazyFrame:
    # Optional uniform sampling by time grid (base grid: 5s)
    if row_sample_secs and row_sample_secs > 0:
        step = max(1, int(round(row_sample_secs / 5)))
        lf = lf.with_columns(((pl.col("publishTimeMs") // 5000) % step).alias("__mod")).filter(pl.col("__mod") == 0).drop("__mod")

    # Prepare basics
    lf = lf.with_columns([
        pl.col("ltp").cast(pl.Float32).alias("ltp_f"),
        pl.col("tradedVolume").cast(pl.Float32).alias("vol_f"),
    ])

    if apply_preoff_filter:
        # Need marketStartMs; filter to [0, preoff_max]
        lf = lf.with_columns(
            ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off")
        ).filter(
            pl.col("marketStartMs").is_not_null()
            & (pl.col("mins_to_off") >= 0)
            & (pl.col("mins_to_off") <= float(preoff_max_min))
        )
    else:
        # No pre-off filtering; mins_to_off may be null
        lf = lf.with_columns(
            ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off")
        )

    lf = lf.sort(["marketId", "selectionId", "publishTimeMs"])

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
    ap = argparse.ArgumentParser("simulate_stream (pre-off, streaming, defs-range + fallback)")
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

    # Definition scan window
    ap.add_argument("--defs-days-back", type=int, default=30, help="scan market_definitions this many days BEFORE start-date")
    ap.add_argument("--defs-days-forward", type=int, default=7, help="scan market_definitions this many days AFTER asof")

    # Fallback if defs coverage is poor
    ap.add_argument("--fallback-coverage-thresh", type=float, default=0.01, help="if < this fraction of rows have marketStartMs post-join, skip pre-off filter for the day")
    ap.add_argument("--force-skip-preoff", action="store_true", help="always skip pre-off filter (debug)")

    args = ap.parse_args()

    if args.polars_max_threads > 0:
        os.environ["POLARS_MAX_THREADS"] = str(args.polars_max_threads)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build validation days
    all_days = list(date_iter(args.start_date, args.asof))
    days = all_days[-args.valid_days:]

    # ---- pre-scan definitions over a range ----
    defs_start = (parse_date(args.start_date) - timedelta(days=args.defs_days_back)).isoformat()
    defs_end   = (parse_date(args.asof) + timedelta(days=args.defs_days_forward)).isoformat()
    log(f"[simulate] scanning definitions {defs_start} .. {defs_end}")
    lf_defs, n_def_files = scan_defs_range(args.curated, args.sport, defs_start, defs_end)
    if lf_defs is None:
        log("[simulate] WARNING: no market_definitions found in range — will need to skip pre-off filtering.")

    # ---- process per day ----
    base_cols = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume"]
    total_days = 0
    mean_evs = []

    for day in days:
        ob_dir = Path(args.curated) / "orderbook_snapshots_5s" / f"sport={args.sport}" / f"date={day}"
        ob_glob = str(ob_dir / "*.parquet")
        n_files = len(glob.glob(ob_glob))
        cap = args.max_files_per_day or "∞"
        log(f"[simulate] {day} … files={n_files} (cap={cap})")

        lf_obs = scan_obs_glob(ob_glob, base_cols, args.max_files_per_day if args.max_files_per_day>0 else None)
        if lf_obs is None:
            log(f"[simulate] {day} … no snapshots; skipping")
            continue

        # Join with definitions (if available)
        if lf_defs is not None:
            lf_join = lf_obs.join(lf_defs, on="marketId", how="left")
            # Coverage check: how many rows have marketStartMs?
            cov_lf = lf_join.select([
                pl.count().alias("__n"),
                pl.col("marketStartMs").is_not_null().sum().alias("__nn")
            ])
            cov = collect_streaming(cov_lf).to_dict(as_series=False)
            n_total = int(cov["__n"][0]) if cov["__n"] else 0
            n_nonnull = int(cov["__nn"][0]) if cov["__nn"] else 0
            pct = (n_nonnull / n_total) if n_total else 0.0
            log(f"[simulate] {day} join coverage: rows={n_total:,} marketStartMs non-null={n_nonnull:,} ({pct:.2%})")
        else:
            lf_join = lf_obs.with_columns(pl.lit(None).alias("marketStartMs"))
            n_total, n_nonnull, pct = 0, 0, 0.0
            log(f"[simulate] {day} join coverage: defs missing")

        # Decide whether to apply pre-off filter
        apply_preoff = not args.force_skip_preoff and (pct >= args.fallback_coverage_thresh)
        if not apply_preoff:
            if args.force_skip_preoff:
                log(f"[simulate] {day} pre-off filter: FORCE SKIPPED (debug)")
            else:
                log(f"[simulate] {day} pre-off filter: SKIPPED (coverage {pct:.2%} < {args.fallback_coverage_thresh:.2%})")

        # Features + (maybe) pre-off filter
        lf = add_basic_features(lf_join, args.preoff_max, args.horizon_secs, args.row_sample_secs, apply_preoff)

        # Collect (streaming)
        df = collect_streaming(lf)
        if df.height == 0:
            log(f"[simulate] {day} … 0 rows post-feature; skipping")
            continue

        # EV proxy from dp
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
