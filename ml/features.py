# ml/features.py
from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict

import polars as pl
import pyarrow.dataset as pads
import pyarrow.fs as pafs


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def _storage_options() -> Dict[str, str]:
    # Allow polars/pyarrow to talk to MinIO directly, if env vars are set.
    opts: Dict[str, str] = {}
    endpoint = os.getenv("AWS_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    if endpoint:
        opts["endpoint_override"] = endpoint
    # s3fs-style flags that pyarrow respects via env:
    #   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
    return opts


def _paths_for_dates(curated_root: str, sport: str, dates: List[str]) -> Tuple[List[str], List[str], List[str]]:
    def pfx(name: str) -> str:
        return f"{curated_root.rstrip('/')}/{name}/sport={sport}"

    snaps = [f"{pfx('orderbook_snapshots_5s')}/date={d}" for d in dates]
    defs = [f"{pfx('market_definitions')}/date={d}" for d in dates]
    res = [f"{pfx('results')}/date={d}" for d in dates]
    return snaps, defs, res


# PATCH 1/1: replace _scan_parquet in ml/features.py (robustly handle directories by expanding to files)

def _scan_parquet(paths: List[str]) -> pl.LazyFrame:
    # Expand each partition directory into concrete parquet file paths, then
    # build a single dataset. Works for Local FS and S3/MinIO.
    fs, _ = pafs.FileSystem.from_uri(paths[0])

    def _expand(dir_or_file: str) -> List[str]:
        info = fs.get_file_info([dir_or_file])[0]
        if info.type == pafs.FileType.File:
            return [dir_or_file] if dir_or_file.endswith(".parquet") else []
        elif info.type in (pafs.FileType.Directory, pafs.FileType.NotFound):  # NotFound can still be a selector root on S3
            selector = pafs.FileSelector(dir_or_file, recursive=True)
            infos = fs.get_file_info(selector)
            return [i.path for i in infos if i.type == pafs.FileType.File and i.path.endswith(".parquet")]
        else:
            return []

    all_files: List[str] = []
    for p in paths:
        all_files.extend(_expand(p))

    if not all_files:
        # Return empty LazyFrame with no rows; caller can handle empty concat.
        return pl.DataFrame([]).lazy()

    ds = pads.dataset(all_files, format="parquet", filesystem=fs)
    tbl = ds.to_table()
    return pl.from_arrow(tbl).lazy()


def build_features_streaming(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int = 30,
    batch_markets: int = 100,
    downsample_secs: Optional[int] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Build feature frame from curated parquet (local or S3/MinIO) for given sport/dates.

    Returns:
        (features_df, total_raw_rows_scanned)
    """

    snap_paths, def_paths, res_paths = _paths_for_dates(curated_root, sport, dates)

    lf_snap = _scan_parquet(snap_paths)
    lf_def = _scan_parquet(def_paths)
    lf_res = _scan_parquet(res_paths)

    # Basic schema normalization
    lf_snap = lf_snap.select(
        pl.col("sport"),
        pl.col("marketId"),
        pl.col("selectionId").cast(pl.Int64),
        pl.col("publishTimeMs").cast(pl.Int64),
        pl.col("backTicks"),
        pl.col("backSizes"),
        pl.col("layTicks"),
        pl.col("laySizes"),
        pl.col("ltp").cast(pl.Float64).alias("ltp"),
        pl.col("ltpTick").cast(pl.Int32),
        pl.col("tradedVolume").cast(pl.Float64),
        pl.col("spreadTicks").cast(pl.Int32),
        pl.col("imbalanceBest1").cast(pl.Float64),
    )

    lf_def = lf_def.select(
        pl.col("sport"),
        pl.col("marketId"),
        pl.col("marketStartMs").cast(pl.Int64),
        pl.col("inPlay").cast(pl.Boolean),
        pl.col("status"),
        pl.col("marketType"),
        pl.col("countryCode"),
    )

    lf_res = lf_res.select(
        pl.col("sport"),
        pl.col("marketId"),
        pl.col("selectionId").cast(pl.Int64),
        pl.col("winLabel").cast(pl.Int32),
        pl.col("runnerStatus"),
        pl.col("settledTimeMs").cast(pl.Int64),
    )

    # Join definitions (static per marketId)
    left = lf_snap.join(lf_def, on=["sport", "marketId"], how="left")

    # Filter by pre-off window: keep snapshots where marketStartMs - publishTimeMs in (0, preoff_minutes]
    left = left.filter(
        (pl.col("marketStartMs").is_not_null())
        & (pl.col("publishTimeMs").is_not_null())
        & ((pl.col("marketStartMs") - pl.col("publishTimeMs")) > 0)
        & ((pl.col("marketStartMs") - pl.col("publishTimeMs")) <= preoff_minutes * 60_000)
    )

    # Compute time-to-off in minutes (bounded) – NEW FEATURE
    left = left.with_columns(
        (
            ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60_000.0)
            .clip(lower_bound=0.0, upper_bound=float(preoff_minutes))
            .alias("tto_minutes")
        ),
        pl.from_epoch((pl.col("publishTimeMs") / 1000).cast(pl.Int64)).alias("ts"),
    )

    # Optional downsampling by time to reduce rows (nearest snapshot per selection per second bucket)
    if downsample_secs and downsample_secs > 0:
        left = left.with_columns((pl.col("publishTimeMs") // (downsample_secs * 1000)).alias("bucket"))
        left = (
            left.sort(["marketId", "selectionId", "publishTimeMs"])
            .group_by(["marketId", "selectionId", "bucket"])
            .agg(pl.all().last())  # last snapshot in each bucket
            .drop("bucket")
            .lazy()
        )

    # Join results to bring the label
    # We want the result per (marketId, selectionId)
    left = left.join(lf_res, on=["sport", "marketId", "selectionId"], how="left")

    # Core microstructure features
    # Overround approximation: use ltp-implied prob; if missing, derive from best prices where possible (ltp used here).
    left = left.with_columns(
        pl.when(pl.col("ltp") > 0)
        .then(1.0 / pl.col("ltp"))
        .otherwise(pl.lit(None))
        .alias("implied_prob")
    )

    # Simple momentum / volatility windows (10s & 60s) using join_asof on self
    base = (
        left.select("marketId", "selectionId", "publishTimeMs", "ltp")
        .sort(["marketId", "selectionId", "publishTimeMs"])
    )

    def _lag_join(delta_ms: int, suffix: str) -> pl.LazyFrame:
        j = base.join_asof(
            base.rename({"publishTimeMs": f"publishTimeMs_{suffix}", "ltp": f"ltp_{suffix}"}),
            left_on="publishTimeMs",
            right_on=f"publishTimeMs_{suffix}",
            by=["marketId", "selectionId"],
            strategy="backward",
            tolerance=delta_ms,
        )
        return j.lazy()

    j10 = _lag_join(10_000, "10s")
    j60 = _lag_join(60_000, "60s")

    # Merge back deltas
    left = (
        left.join(j10, on=["marketId", "selectionId", "publishTimeMs"], how="left")
        .join(j60, on=["marketId", "selectionId", "publishTimeMs"], how="left")
        .with_columns(
            (pl.col("ltp") - pl.col("ltp_10s")).alias("mom_10s"),
            (pl.col("ltp") - pl.col("ltp_60s")).alias("mom_60s"),
        )
        .with_columns(
            (pl.col("mom_10s").abs()).alias("vol_10s"),
            (pl.col("mom_60s").abs()).alias("vol_60s"),
        )
    )

    # Final select of feature columns + identifiers + label
    feat = left.select(
        "sport",
        "marketId",
        "selectionId",
        "publishTimeMs",
        "ts",
        "tto_minutes",  # <-- NEW
        "ltp",
        "tradedVolume",
        "spreadTicks",
        "imbalanceBest1",
        "implied_prob",
        "mom_10s",
        "mom_60s",
        "vol_10s",
        "vol_60s",
        "winLabel",
    ).collect()  # collect once; we rely on pyarrow.dataset above to batch scans robustly

    total_raw = feat.height  # approximate; if you need exact raw count, track via snapshots scan before joins.

    return feat, total_raw
