# ml/features.py
from __future__ import annotations

from typing import List, Tuple, Optional

import polars as pl
import pyarrow.dataset as pads
import pyarrow.fs as pafs


def _paths_for_dates(curated_root: str, sport: str, dates: List[str]) -> Tuple[List[str], List[str], List[str]]:
    def pfx(name: str) -> str:
        return f"{curated_root.rstrip('/')}/{name}/sport={sport}"

    snaps = [f"{pfx('orderbook_snapshots_5s')}/date={d}" for d in dates]
    defs = [f"{pfx('market_definitions')}/date={d}" for d in dates]
    res = [f"{pfx('results')}/date={d}" for d in dates]
    return snaps, defs, res


def _scan_parquet(paths: List[str]) -> pl.LazyFrame:
    """
    Expand directory-like inputs to concrete parquet file paths and build a single dataset.
    Works for Local FS and S3/MinIO.
    """
    if not paths:
        return pl.DataFrame([]).lazy()

    fs, _ = pafs.FileSystem.from_uri(paths[0])

    def _expand(dir_or_file: str) -> List[str]:
        info = fs.get_file_info([dir_or_file])[0]
        if info.type == pafs.FileType.File:
            return [dir_or_file] if dir_or_file.endswith(".parquet") else []
        elif info.type in (pafs.FileType.Directory, pafs.FileType.NotFound):
            # NotFound can still be a selector root on object stores
            selector = pafs.FileSelector(dir_or_file, recursive=True)
            infos = fs.get_file_info(selector)
            return [i.path for i in infos if i.type == pafs.FileType.File and i.path.endswith(".parquet")]
        else:
            return []

    all_files: List[str] = []
    for p in paths:
        all_files.extend(_expand(p))

    if not all_files:
        return pl.DataFrame([]).lazy()

    ds = pads.dataset(all_files, format="parquet", filesystem=fs)
    tbl = ds.to_table()
    return pl.from_arrow(tbl).lazy()


def build_features_streaming(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int = 30,
    batch_markets: int = 100,     # kept for API compatibility
    downsample_secs: Optional[int] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Build feature frame from curated parquet (local or S3/MinIO) for given sport/dates.

    Returns:
        (features_df, total_raw_rows_scanned_approx)
    """
    snap_paths, def_paths, res_paths = _paths_for_dates(curated_root, sport, dates)

    lf_snap = _scan_parquet(snap_paths)
    lf_def = _scan_parquet(def_paths)
    lf_res = _scan_parquet(res_paths)

    # Resolve schemas once to avoid repeated expensive lookups and warnings.
    snap_names = set(lf_snap.collect_schema().names())
    def_names = set(lf_def.collect_schema().names())
    res_names = set(lf_res.collect_schema().names())

    if not snap_names or not def_names or not res_names:
        return pl.DataFrame([]), 0

    # --- Normalize/select columns from snapshots ---
    snap_keep_order = (
        "sport", "marketId", "selectionId", "publishTimeMs",
        "backTicks", "backSizes", "layTicks", "laySizes",
        "ltpTick", "ltp", "tradedVolume", "spreadTicks", "imbalanceBest1"
    )
    keep_snap = [c for c in snap_keep_order if c in snap_names]

    lf_snap = lf_snap.select(
        *[
            (pl.col("selectionId").cast(pl.Int64) if c == "selectionId"
             else pl.col("publishTimeMs").cast(pl.Int64) if c == "publishTimeMs"
             else pl.col("ltp").cast(pl.Float64) if c == "ltp"
             else pl.col("ltpTick").cast(pl.Int32) if c == "ltpTick"
             else pl.col("tradedVolume").cast(pl.Float64) if c == "tradedVolume"
             else pl.col("spreadTicks").cast(pl.Int32) if c == "spreadTicks"
             else pl.col("imbalanceBest1").cast(pl.Float64) if c == "imbalanceBest1"
             else pl.col(c))
            for c in keep_snap
        ]
    )

    # --- Definitions ---
    def_keep_order = ("sport", "marketId", "marketStartMs", "inPlay", "status", "marketType", "countryCode")
    keep_def = [c for c in def_keep_order if c in def_names]

    lf_def = lf_def.select(
        *[
            (pl.col("marketStartMs").cast(pl.Int64) if c == "marketStartMs"
             else pl.col("inPlay").cast(pl.Boolean) if c == "inPlay"
             else pl.col(c))
            for c in keep_def
        ]
    )

    # --- Results ---
    res_keep_order = ("sport", "marketId", "selectionId", "winLabel", "runnerStatus", "settledTimeMs")
    keep_res = [c for c in res_keep_order if c in res_names]

    lf_res = lf_res.select(
        *[
            (pl.col("selectionId").cast(pl.Int64) if c == "selectionId"
             else pl.col("winLabel").cast(pl.Int32) if c == "winLabel"
             else pl.col("settledTimeMs").cast(pl.Int64) if c == "settledTimeMs"
             else pl.col(c))
            for c in keep_res
        ]
    )

    # Join definitions (static per market)
    left = lf_snap.join(lf_def, on=["sport", "marketId"], how="left")

    # Filter by pre-off window: keep snapshots where marketStartMs - publishTimeMs in (0, preoff_minutes]
    left = left.filter(
        (pl.col("marketStartMs").is_not_null())
        & (pl.col("publishTimeMs").is_not_null())
        & ((pl.col("marketStartMs") - pl.col("publishTimeMs")) > 0)
        & ((pl.col("marketStartMs") - pl.col("publishTimeMs")) <= preoff_minutes * 60_000)
    )

    # Compute time-to-off in minutes (bounded) + timestamp
    left = left.with_columns(
        (
            ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60_000.0)
            .clip(lower_bound=0.0, upper_bound=float(preoff_minutes))
            .alias("tto_minutes")
        ),
        pl.from_epoch((pl.col("publishTimeMs") / 1000).cast(pl.Int64)).alias("ts"),
    )

    # Optional downsampling: one snapshot per bucket per (marketId, selectionId)
    if downsample_secs and downsample_secs > 0:
        left = left.with_columns((pl.col("publishTimeMs") // (downsample_secs * 1000)).alias("bucket"))
        left = (
            left.sort(["marketId", "selectionId", "publishTimeMs"])
            .group_by(["marketId", "selectionId", "bucket"])
            .agg(pl.all().last())
            .drop("bucket")
            .lazy()
        )

    # Join results to bring the label
    left = left.join(lf_res, on=["sport", "marketId", "selectionId"], how="left")

    # Implied prob from ltp
    left = left.with_columns(
        pl.when(pl.col("ltp").is_not_null() & (pl.col("ltp") > 0.0))
        .then(1.0 / pl.col("ltp"))
        .otherwise(None)
        .alias("implied_prob")
    )

    # --- Momentum / volatility using as-of self-joins (avoid duplicate 'ltp' collisions) ---
    base = (
        left.select(["marketId", "selectionId", "publishTimeMs", "ltp"])
            .sort(["marketId", "selectionId", "publishTimeMs"])
    )

    def _lag_frame(delta_ms: int, suffix: str) -> pl.LazyFrame:
        j = base.join_asof(
            base.rename({"publishTimeMs": f"publishTimeMs_{suffix}", "ltp": f"ltp_{suffix}"}),
            left_on="publishTimeMs",
            right_on=f"publishTimeMs_{suffix}",
            by=["marketId", "selectionId"],
            strategy="backward",
            tolerance=delta_ms,
        ).select(["marketId", "selectionId", "publishTimeMs", f"ltp_{suffix}"])
        return j.lazy()

    j10 = _lag_frame(10_000, "10s")
    j60 = _lag_frame(60_000, "60s")

    left = (
        left.join(j10, on=["marketId", "selectionId", "publishTimeMs"], how="left")
            .join(j60, on=["marketId", "selectionId", "publishTimeMs"], how="left")
            .with_columns(
                (pl.col("ltp") - pl.col("ltp_10s")).alias("mom_10s"),
                (pl.col("ltp") - pl.col("ltp_60s")).alias("mom_60s"),
            )
            .with_columns(
                pl.col("mom_10s").abs().alias("vol_10s"),
                pl.col("mom_60s").abs().alias("vol_60s"),
            )
    )

    # Final selection
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
    # Avoid resolving schema repeatedly
    final_names = set(left.collect_schema().names())
    have = [c for c in wanted if c in final_names]

    feat = left.select(have).collect()

    total_raw = feat.height  # approximate scanned rows (post-filters)

    return feat, total_raw
