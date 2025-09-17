# ml/features.py
from __future__ import annotations

from typing import List, Tuple, Optional

import polars as pl
import pyarrow.dataset as pads


# -------------------------
# Path helpers
# -------------------------

def _paths_for_dates(curated_root: str, sport: str, dates: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Build partitioned paths for snapshots, market definitions, and results
    under the curated root (local or s3). Expected layout:
      <root>/orderbook_snapshots_5s/sport=<sport>/date=YYYY-MM-DD/part-*.parquet
      <root>/market_definitions/sport=<sport>/date=YYYY-MM-DD/part-*.parquet
      <root>/results/sport=<sport>/date=YYYY-MM-DD/part-*.parquet
    """
    def pfx(name: str) -> str:
        return f"{curated_root.rstrip('/')}/{name}/sport={sport}"
    snaps = [f"{pfx('orderbook_snapshots_5s')}/date={d}" for d in dates]
    defs  = [f"{pfx('market_definitions')}/date={d}" for d in dates]
    res   = [f"{pfx('results')}/date={d}" for d in dates]
    return snaps, defs, res


def _scan_parquet(paths: List[str]) -> pl.LazyFrame:
    """
    Read a list of partition directories/files (local or s3://...) using PyArrow dataset.
    Relies on environment for S3 configuration (unchanged behavior).
    Missing paths are skipped quietly.
    """
    frames: List[pl.LazyFrame] = []
    for p in paths:
        try:
            ds = pads.dataset(p, format="parquet", partitioning="hive", ignore_missing_files=True)
        except Exception:
            continue
        tbl = ds.to_table()
        if tbl.num_rows > 0:
            frames.append(pl.from_arrow(tbl).lazy())
    if not frames:
        return pl.DataFrame([]).lazy()
    return pl.concat(frames, how="vertical")


# -------------------------
# Feature building
# -------------------------

def _with_basic_transforms(snaps: pl.LazyFrame) -> pl.LazyFrame:
    """Add timestamps and simple transforms expected by downstream code."""
    schema = snaps.collect_schema().names()

    # publishTimeMs -> ts (datetime) and seconds
    if "publishTimeMs" in schema:
        snaps = snaps.with_columns(
            (pl.col("publishTimeMs") / 1000).alias("ts_s")
        ).with_columns(
            pl.from_epoch(pl.col("ts_s")).alias("ts")
        )
    else:
        snaps = snaps.with_columns(pl.lit(None).alias("ts_s"), pl.lit(None).alias("ts"))

    # implied_prob = 1/ltp (guarded)
    if "ltp" in schema:
        snaps = snaps.with_columns(
            pl.when((pl.col("ltp").is_not_null()) & (pl.col("ltp") > 0))
              .then(1.0 / pl.col("ltp"))
              .otherwise(None)
              .alias("implied_prob")
        )
    else:
        snaps = snaps.with_columns(pl.lit(None).alias("implied_prob"))

    # Rolling momentum and volume windows: 10s (2 * 5s), 60s (12 * 5s)
    by_keys = ["marketId", "selectionId"]
    schema = snaps.collect_schema().names()
    if all(k in schema for k in by_keys) and "publishTimeMs" in schema:
        snaps = snaps.sort(by_keys + ["publishTimeMs"])  # ensure order

        if "ltp" in schema:
            snaps = snaps.with_columns([
                (pl.col("ltp") - pl.col("ltp").shift(2)).over(by_keys).alias("mom_10s"),
                (pl.col("ltp") - pl.col("ltp").shift(12)).over(by_keys).alias("mom_60s"),
            ])
        else:
            snaps = snaps.with_columns([pl.lit(None).alias("mom_10s"), pl.lit(None).alias("mom_60s")])

        if "tradedVolume" in schema:
            snaps = snaps.with_columns([
                (pl.col("tradedVolume") - pl.col("tradedVolume").shift(2)).over(by_keys).alias("vol_10s"),
                (pl.col("tradedVolume") - pl.col("tradedVolume").shift(12)).over(by_keys).alias("vol_60s"),
            ])
        else:
            snaps = snaps.with_columns([pl.lit(None).alias("vol_10s"), pl.lit(None).alias("vol_60s")])
    else:
        snaps = snaps.with_columns([
            pl.lit(None).alias("mom_10s"),
            pl.lit(None).alias("mom_60s"),
            pl.lit(None).alias("vol_10s"),
            pl.lit(None).alias("vol_60s"),
        ])
    return snaps


def build_features_streaming(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int = 30,
    batch_markets: int = 100,     # API compat; not used in this single-pass builder
    downsample_secs: Optional[int] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Build features by scanning partitioned Parquet under curated_root for given dates.
    Returns (features_df, total_raw_rows).
    - Includes 'countryCode' from market definitions.
    - Computes time-to-off in minutes from marketStartMs.
    - Computes basic momentum/volume windows from snapshots.
    - Applies the same pre-off filter as the classic pipeline: 0 <= tto_minutes <= preoff_minutes.
    """
    snap_paths, def_paths, res_paths = _paths_for_dates(curated_root, sport, dates)

    lf_snap = _scan_parquet(snap_paths)
    lf_def  = _scan_parquet(def_paths)
    lf_res  = _scan_parquet(res_paths)

    # Nothing to do if no snapshots
    if len(lf_snap.collect_schema().names()) == 0:
        return pl.DataFrame([]), 0

    # Minimal subset from snapshots
    keep_snap_wanted = [
        "sport", "marketId", "selectionId", "publishTimeMs",
        "ltp", "tradedVolume", "spreadTicks", "imbalanceBest1",
    ]
    keep_snap = [c for c in keep_snap_wanted if c in lf_snap.collect_schema().names()]
    lf_snap = lf_snap.select(keep_snap)
    lf_snap = _with_basic_transforms(lf_snap)

    # Market definitions: need marketStartMs and countryCode (if present)
    if len(lf_def.collect_schema().names()) > 0:
        keep_def = [c for c in ["marketId", "marketStartMs", "countryCode"] if c in lf_def.collect_schema().names()]
        lf_def = lf_def.select(keep_def)
    else:
        lf_def = pl.DataFrame({"marketId": [], "marketStartMs": [], "countryCode": []}).lazy()

    # Results: join for winLabel if available
    if len(lf_res.collect_schema().names()) > 0:
        keep_res = [c for c in ["marketId", "selectionId", "winLabel"] if c in lf_res.collect_schema().names()]
        lf_res = lf_res.select(keep_res)
    else:
        lf_res = pl.DataFrame({"marketId": [], "selectionId": [], "winLabel": []}).lazy()

    # Join snapshots with definitions
    left = lf_snap.join(lf_def, on="marketId", how="left")

    # tto_minutes from marketStartMs and publishTimeMs (may be NULL if marketStartMs missing)
    left = left.with_columns(
        ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / (1000 * 60)).alias("tto_minutes")
    )

    # Optional downsample: floor to a time bucket, then take last row in bucket
    if downsample_secs:
        left = left.with_columns(((pl.col("publishTimeMs") // (downsample_secs * 1000)) * (downsample_secs * 1000)).alias("_buck"))
        left = (
            left
            .group_by(["marketId", "selectionId", "_buck"])
            .agg([pl.all().last()])
            .drop("_buck")
        )

    # Pre-off filter: keep rows with 0 <= tto_minutes <= preoff_minutes
    # If tto_minutes is null (missing marketStartMs), we leave the row out here to match classic behavior.
    left = left.filter((pl.col("tto_minutes") >= 0) & (pl.col("tto_minutes") <= preoff_minutes))

    # Join results for label
    left = left.join(lf_res, on=["marketId", "selectionId"], how="left")

    # Final feature selection
    wanted = [
        "sport",
        "marketId",
        "selectionId",
        "publishTimeMs",
        "ts",
        "tto_minutes",
        "ltp",
        "tradedVolume",
        "spreadTicks",
        "imbalanceBest1",
        "implied_prob",
        "mom_10s",
        "mom_60s",
        "vol_10s",
        "vol_60s",
        "countryCode",
        "winLabel",
    ]
    final_names = set(left.collect_schema().names())
    have = [c for c in wanted if c in final_names]

    feat = left.select(have).collect()
    total_raw = feat.height

    return feat, total_raw
