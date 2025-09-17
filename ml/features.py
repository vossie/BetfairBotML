# ml/features.py
from __future__ import annotations

from typing import List, Tuple, Optional

import polars as pl
import pyarrow.dataset as pads
import pyarrow.fs as pafs


# -------------------------
# Path helpers
# -------------------------

def _paths_for_dates(curated_root: str, sport: str, dates: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Build partitioned paths for snapshots, market definitions, and results
    under the curated root (local or s3). Example layout (from user's data):
      betfair-curated/market_definitions/sport=<sport>/date=YYYY-MM-DD/part-*.parquet
      betfair-curated/orderbook_snapshots_5s/sport=<sport>/date=YYYY-MM-DD/part-*.parquet
      betfair-curated/results/sport=<sport>/date=YYYY-MM-DD/part-*.parquet
    """
    def pfx(name: str) -> str:
        return f"{curated_root.rstrip('/')}/{name}/sport={sport}"

    snaps = [f"{pfx('orderbook_snapshots_5s')}/date={d}" for d in dates]
    defs = [f"{pfx('market_definitions')}/date={d}" for d in dates]
    res  = [f"{pfx('results')}/date={d}" for d in dates]
    return snaps, defs, res


def _fs_and_path(uri: str):
    """Return (filesystem, path_without_scheme) for a given local/S3/other URI."""
    fs, path = pafs.FileSystem.from_uri(uri)
    return fs, path


def _expand_uri(uri: str) -> List[str]:
    """Expand a file or directory URI into file paths compatible with its filesystem."""
    fs, path = _fs_and_path(uri)
    info = fs.get_file_info([path])[0]
    if info.type == pafs.FileType.Directory:
        selector = pafs.FileSelector(path, recursive=True)
        return [fi.path for fi in fs.get_file_info(selector) if fi.is_file]
    elif info.is_file:
        return [path]
    else:
        return []


def _scan_parquet(uris: List[str]) -> pl.LazyFrame:
    """
    Build a LazyFrame from a list of URIs (local or s3://...). All URIs must belong
    to the same filesystem (e.g., same S3 endpoint). Returns an empty LazyFrame if none.
    """
    if not uris:
        return pl.DataFrame([]).lazy()

    base_fs, _ = _fs_and_path(uris[0])
    all_files: List[str] = []
    for u in uris:
        fs_u, _ = _fs_and_path(u)
        if type(fs_u) != type(base_fs):
            raise ValueError(f"Mixed filesystems not supported: {type(base_fs)} vs {type(fs_u)} for {u}")
        all_files.extend(_expand_uri(u))

    if not all_files:
        return pl.DataFrame([]).lazy()

    ds = pads.dataset(all_files, format="parquet", filesystem=base_fs)
    tbl = ds.to_table()
    return pl.from_arrow(tbl).lazy()


# -------------------------
# Feature building
# -------------------------

def _safe_col(df: pl.LazyFrame, name: str) -> pl.Expr:
    """Return a column if present; else a null placeholder."""
    if name in df.collect_schema().names():
        return pl.col(name)
    return pl.lit(None).alias(name)


def _with_basic_transforms(snaps: pl.LazyFrame) -> pl.LazyFrame:
    """Add timestamps and simple transforms expected by downstream code."""
    # publishTimeMs -> ts (datetime) and seconds
    snaps = snaps.with_columns([
        (pl.col("publishTimeMs") / 1000).alias("ts_s"),
    ]).with_columns([
        pl.from_epoch("ts_s").alias("ts")
    ])

    # implied_prob = 1/ltp (guarded)
    if "ltp" in snaps.collect_schema().names():
        snaps = snaps.with_columns(
            pl.when((pl.col("ltp").is_not_null()) & (pl.col("ltp") > 0))
              .then(1.0 / pl.col("ltp"))
              .otherwise(None)
              .alias("implied_prob")
        )
    else:
        snaps = snaps.with_columns(pl.lit(None).alias("implied_prob"))

    # Rolling momentum and volume windows: 10s (2 * 5s), 60s (12 * 5s)
    # These are simplistic diffs over fixed lag counts.
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
    batch_markets: int = 100,     # kept for API compatibility; not used in this single-pass builder
    downsample_secs: Optional[int] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Build features by scanning partitioned Parquet under curated_root for given dates.
    Returns (features_df, total_raw_rows).
    - Includes 'countryCode' from market definitions.
    - Computes simple time-to-off in minutes from marketStartMs.
    - Computes basic momentum/volume windows from snapshots.
    """
    snap_paths, def_paths, res_paths = _paths_for_dates(curated_root, sport, dates)

    lf_snap = _scan_parquet(snap_paths)
    lf_def  = _scan_parquet(def_paths)
    lf_res  = _scan_parquet(res_paths)

    if lf_snap.collect_schema().is_empty():
        return pl.DataFrame([]), 0

    # Minimal subset from snapshots
    keep_snap = [c for c in [
        "sport", "marketId", "selectionId", "publishTimeMs",
        "ltp", "tradedVolume", "spreadTicks", "imbalanceBest1",
    ] if c in lf_snap.collect_schema().names()]
    lf_snap = lf_snap.select(keep_snap)
    lf_snap = _with_basic_transforms(lf_snap)

    # Market definitions: need marketStartMs and countryCode (if present)
    if not lf_def.collect_schema().is_empty():
        keep_def = [c for c in ["marketId", "marketStartMs", "countryCode"] if c in lf_def.collect_schema().names()]
        lf_def = lf_def.select(keep_def)
    else:
        lf_def = pl.DataFrame({"marketId": [], "marketStartMs": [], "countryCode": []}).lazy()

    # Results: join for winLabel if available
    if not lf_res.collect_schema().is_empty():
        keep_res = [c for c in ["marketId", "selectionId", "winLabel"] if c in lf_res.collect_schema().names()]
        lf_res = lf_res.select(keep_res)
    else:
        lf_res = pl.DataFrame({"marketId": [], "selectionId": [], "winLabel": []}).lazy()

    # Join snapshots with definitions
    left = lf_snap.join(lf_def, on="marketId", how="left")

    # tto_minutes from marketStartMs and publishTimeMs
    left = left.with_columns(
        (
            (pl.col("marketStartMs") - pl.col("publishTimeMs")) / (1000 * 60)
        ).alias("tto_minutes")
    )

    # Optional downsample: floor to a time bucket, then take last row in bucket
    if downsample_secs:
        # Compute bucket per market/selection
        left = left.with_columns(((pl.col("publishTimeMs") // (downsample_secs * 1000)) * (downsample_secs * 1000)).alias("_buck"))
        left = (
            left
            .group_by(["marketId", "selectionId", "_buck"])
            .agg([pl.all().last()])
            .drop("_buck")
        )

    # Filter to pre-off window (keep rows with 0 <= tto_minutes <= preoff_minutes)
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
