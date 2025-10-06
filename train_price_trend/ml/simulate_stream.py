#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, glob
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl

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
    x = pl.when(dp > ev_cap).then(pl.lit(ev_cap)).otherwise(dp)
    x = pl.when(x < -ev_cap).then(pl.lit(-ev_cap)).otherwise(x)
    return (x * ev_scale).cast(pl.Float64)

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

def uniform_sample(seq: list[str], k: int) -> list[str]:
    if k <= 0 or k >= len(seq): return seq
    step = len(seq) / k
    # spread picks across day; round to nearest indices
    idxs = {min(len(seq)-1, int(round(i*step))) for i in range(k)}
    # guard: ensure exactly k unique indices
    while len(idxs) < k:
        idxs.add(len(idxs))
    return [seq[i] for i in sorted(idxs)]

def scan_obs_glob(glob_path: str, want_cols: list[str], max_files: int | None, sample_mode: str) -> tuple[pl.LazyFrame | None, int]:
    files = sorted(glob.glob(glob_path))
    total = len(files)
    if not files:
        return None, 0
    if max_files is not None and max_files > 0 and total > max_files:
        if sample_mode == "uniform":
            files = uniform_sample(files, max_files)
        elif sample_mode == "tail":
            files = files[-max_files:]
        else:  # head
            files = files[:max_files]
    lf = pl.scan_parquet(files)
    have = lf.collect_schema().names()
    take = [c for c in want_cols if c in have]
    if not take:
        return None, total
    return lf.select([pl.col(c) for c in take]), total

def scan_defs_range(curated: str, sport: str, start_date: str, end_date: str) -> tuple[pl.LazyFrame | None, int]:
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
    ]).filter(pl.col("marketId").is_not_null())
    lf = (lf
          .group_by("marketId")
          .agg([
              pl.col("marketStartMs").last().alias("marketStartMs"),
              pl.col("__def_pub").max().alias("__def_pub_max")
          ])
          .drop("__def_pub_max"))
    return lf, len(files)

def add_basic_features(
    lf: pl.LazyFrame,
    preoff_max_min: int,
    horizon_secs: int,
    row_sample_secs: int | None,
    apply_preoff_filter: bool
) -> pl.LazyFrame:
    if row_sample_secs and row_sample_secs > 0:
        step = max(1, int(round(row_sample_secs / 5)))
        lf = lf.with_columns(((pl.col("publishTimeMs") // 5000) % step).alias("__mod")).filter(pl.col("__mod") == 0).drop("__mod")

    lf = lf.with_columns([
        pl.col("ltp").cast(pl.Float32).alias("ltp_f"),
        pl.col("tradedVolume").cast(pl.Float32).alias("vol_f"),
        ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off"),
    ])

    if apply_preoff_filter:
        lf = lf.filter(
            pl.col("marketStartMs").is_not_null()
            & (pl.col("mins_to_off") >= 0)
            & (pl.col("mins_to_off") <= float(preoff_max_min))
        )

    grp = ["marketId", "selectionId"]
    steps = max(1, int(round(horizon_secs / 5)))
    lf = lf.sort(["marketId", "selectionId", "publishTimeMs"]).with_columns([
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
        (pl.col("ltp_f").shift(-steps).over(grp) - pl.col("ltp_f")).alias("__dp"),
    ])

    keep = [
        "marketId","selectionId","publishTimeMs","marketStartMs","mins_to_off",
        "ltp_f","vol_f",
        "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
        "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
        "ltp_future","__dp"
    ]
    have = lf.collect_schema().names()
    take = [c for c in keep if c in have]
    return lf.select([pl.col(c) for c in take])

def main():
    ap = argparse.ArgumentParser("simulate_stream (pre-off, streaming, defs-range + uniform sampling)")
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

    # IO/perf
    ap.add_argument("--max-files-per-day", type=int, default=0)
    ap.add_argument("--file-sample-mode", choices=["uniform","head","tail"], default="uniform",
                    help="how to pick files when capping per day (default uniform)")
    ap.add_argument("--row-sample-secs", type=int, default=0)
    ap.add_argument("--polars-max-threads", type=int, default=0)

    # defs window
    ap.add_argument("--defs-days-back", type=int, default=30)
    ap.add_argument("--defs-days-forward", type=int, default=7)

    # fallback
    ap.add_argument("--fallback-coverage-thresh", type=float, default=0.01)
    ap.add_argument("--force-skip-preoff", action="store_true")

    args = ap.parse_args()
    if args.polars_max_threads > 0:
        os.environ["POLARS_MAX_THREADS"] = str(args.polars_max_threads)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_days = list(date_iter(args.start_date, args.asof))
    days = all_days[-args.valid_days:]

    defs_start = (parse_date(args.start_date) - timedelta(days=args.defs_days_back)).isoformat()
    defs_end   = (parse_date(args.asof) + timedelta(days=args.defs_days_forward)).isoformat()
    log(f"[simulate] scanning definitions {defs_start} .. {defs_end}")
    lf_defs, n_def_files = scan_defs_range(args.curated, args.sport, defs_start, defs_end)
    if lf_defs is None:
        log("[simulate] WARNING: no market_definitions found in range — pre-off filter may be skipped.")

    base_cols = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume"]
    total_days = 0
    mean_evs = []

    for day in days:
        ob_dir = Path(args.curated) / "orderbook_snapshots_5s" / f"sport={args.sport}" / f"date={day}"
        ob_glob = str(ob_dir / "*.parquet")
        lf_obs, total_files = scan_obs_glob(ob_glob, base_cols,
                                            args.max_files_per_day if args.max_files_per_day>0 else None,
                                            args.file_sample_mode)
        cap = args.max_files_per_day or "∞"
        log(f"[simulate] {day} … files={total_files} (cap={cap}, mode={args.file_sample_mode})")
        if lf_obs is None:
            log(f"[simulate] {day} … no snapshots; skipping")
            continue

        lf_join = lf_obs.join(lf_defs, on="marketId", how="left") if lf_defs is not None else lf_obs.with_columns(pl.lit(None).alias("marketStartMs"))

        # coverage
        cov_lf = lf_join.select([pl.len().alias("__n"), pl.col("marketStartMs").is_not_null().sum().alias("__nn")])
        cov = collect_streaming(cov_lf).to_dict(as_series=False)
        n_total = int(cov["__n"][0]) if cov["__n"] else 0
        n_nonnull = int(cov["__nn"][0]) if cov["__nn"] else 0
        pct = (n_nonnull / n_total) if n_total else 0.0
        log(f"[simulate] {day} join coverage: rows={n_total:,} marketStartMs non-null={n_nonnull:,} ({pct:.2%})")

        apply_preoff = (not args.force_skip_preoff) and (pct >= args.fallback_coverage_thresh)
        if not apply_preoff:
            why = "force" if args.force_skip_preoff else f"coverage {pct:.2%} < {args.fallback_coverage_thrash:.2%}"
            log(f"[simulate] {day} pre-off filter: SKIPPED ({why})")

        lf_feats = add_basic_features(lf_join, args.preoff_max, args.horizon_secs, args.row_sample_secs, apply_preoff)

        # log pre-off window coverage
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

        df = df.with_columns(clamp_to_ev(pl.col("__dp"), args.ev_scale, args.ev_cap).alias("ev_per_1"))
        avg_ev = float(df["ev_per_1"].mean())
        mean_evs.append(avg_ev)

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
