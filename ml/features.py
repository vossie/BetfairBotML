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
    Expected layout:
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


def _scan_many(paths: List[str]):
    """
    Read a list of directories/files (local or s3://...) with PyArrow.
    We do NOT override any S3 config here; environment drives behavior.
    Missing/non-existent paths are skipped quietly.
    Returns a Polars LazyFrame (possibly empty).
    """
    frames: List[pl.LazyFrame] = []
    if not paths:
        return pl.DataFrame([]).lazy()
    try:
        # Let Arrow handle multiple roots at once; more robust than per-path loops
        ds = pads.dataset(paths, format="parquet", ignore_missing_files=True)
        tbl = ds.to_table()
        if tbl.num_rows > 0:
            frames.append(pl.from_arrow(tbl).lazy())
    except Exception:
        # Fallback: try one-by-one so a single bad path doesn't nuke everything
        for p in paths:
            try:
                ds = pads.dataset(p, format="parquet", ignore_missing_files=True)
                tbl = ds.to_table()
                if tbl.num_rows > 0:
                    frames.append(pl.from_arrow(tbl).lazy())
            except Exception:
                continue

    return pl.concat(frames, how="vertical") if frames else pl.DataFrame([]).lazy()


# -------------------------
# Feature building
# -------------------------

def _with_basic_transforms(snaps: pl.LazyFrame) -> pl.LazyFrame:
    """
    Adds:
      ts (datetime from publishTimeMs),
      implied_prob (1/ltp), simple momentum/volume windows (10s/60s).
    Does NOT drop any rows here.
    """
    schema = snaps.collect_schema().names()

    # publishTimeMs -> ts (datetime)
    if "publishTimeMs" in schema:
        snaps = snaps.with_columns(
            (pl.col("publishTimeMs") / 1000).alias("ts_s")
        ).with_columns(
            pl.from_epoch(pl.col("ts_s")).alias("ts")
        )
    else:
        snaps = snaps.with_columns(pl.lit(None).alias("ts_s"), pl.lit(None).alias("ts"))

    # implied_prob = 1/ltp
    if "ltp" in schema:
        snaps = snaps.with_columns(
            pl.when((pl.col("ltp").is_not_null()) & (pl.col("ltp") > 0))
              .then(1.0 / pl.col("ltp"))
              .otherwise(None)
              .alias("implied_prob")
        )
    else:
        snaps = snaps.with_columns(pl.lit(None).alias("implied_prob"))

    # Momentum/volume over 10s (2*5s) and 60s (12*5s)
    by_keys = ["marketId", "selectionId"]
    schema = snaps.collect_schema().names()
    if all(k in schema for k in by_keys) and "publishTimeMs" in schema:
        snaps = snaps.sort(by_keys + ["publishTimeMs"])
        snaps = snaps.with_columns([
            (pl.col("ltp") - pl.col("ltp").shift(2)).over(by_keys).alias("mom_10s") if "ltp" in schema else pl.lit(None).alias("mom_10s"),
            (pl.col("ltp") - pl.col("ltp").shift(12)).over(by_keys).alias("mom_60s") if "ltp" in schema else pl.lit(None).alias("mom_60s"),
            (pl.col("tradedVolume") - pl.col("tradedVolume").shift(2)).over(by_keys).alias("vol_10s") if "tradedVolume" in schema else pl.lit(None).alias("vol_10s"),
            (pl.col("tradedVolume") - pl.col("tradedVolume").shift(12)).over(by_keys).alias("vol_60s") if "tradedVolume" in schema else pl.lit(None).alias("vol_60s"),
        ])
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
    batch_markets: int = 100,     # kept for API compatibility; not used here
    downsample_secs: Optional[int] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Single-pass builder that:
      - reads snapshots, defs, results via Arrow Dataset
      - joins defs (adds marketStartMs + countryCode) and results (winLabel)
      - computes helper fields (ts, tto_minutes, momentum/volume)
      - DOES NOT pre-off filter here (keeps all rows)
    Returns (features_df, total_raw_rows)
    """
    snap_paths, def_paths, res_paths = _paths_for_dates(curated_root, sport, dates)

    lf_snap = _scan_many(snap_paths)
    lf_def  = _scan_many(def_paths)
    lf_res  = _scan_many(res_paths)

    # Early exit if no snapshots
    if len(lf_snap.collect_schema().names()) == 0:
        return pl.DataFrame([]), 0

    # Keep snapshots untouched (avoid dropping columns relied on elsewhere)
    snaps = lf_snap
    snaps = _with_basic_transforms(snaps)

    # Defs: only the fields we need; if missing, create empties
    if len(lf_def.collect_schema().names()) > 0:
        cols_def = [c for c in ["marketId", "marketStartMs", "countryCode"] if c in lf_def.collect_schema().names()]
        defs = lf_def.select(cols_def)
    else:
        defs = pl.DataFrame({"marketId": [], "marketStartMs": [], "countryCode": []}).lazy()

    # Results: winLabel
    if len(lf_res.collect_schema().names()) > 0:
        cols_res = [c for c in ["marketId", "selectionId", "winLabel"] if c in lf_res.collect_schema().names()]
        res = lf_res.select(cols_res)
    else:
        res = pl.DataFrame({"marketId": [], "selectionId": [], "winLabel": []}).lazy()

    # Join snaps + defs
    left = snaps.join(defs, on="marketId", how="left")

    # tto_minutes from marketStartMs (may be NULL)
    if "publishTimeMs" in left.collect_schema().names():
        left = left.with_columns(
            ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / (1000 * 60)).alias("tto_minutes")
        )
    else:
        left = left.with_columns(pl.lit(None).alias("tto_minutes"))

    # Optional downsample
    if downsample_secs:
        left = left.with_columns(((pl.col("publishTimeMs") // (downsample_secs * 1000)) * (downsample_secs * 1000)).alias("_buck"))
        left = (
            left
            .group_by(["marketId", "selectionId", "_buck"])
            .agg([pl.all().last()])
            .drop("_buck")
        )

    # Join results
    left = left.join(res, on=["marketId", "selectionId"], how="left")

    # Final selection (don’t fail if some columns are absent)
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
    have = [c for c in wanted if c in left.collect_schema().names()]
    feat = left.select(have).collect()
    total_raw = feat.height
    return feat, total_raw
