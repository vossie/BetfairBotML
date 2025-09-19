#!/usr/bin/env python3
"""
Drop-in replacement for price-move label building.

Usage in train_edge_temporal.py:
    from pm_labels import add_price_move_labels

Key properties:
- Works with ~5s snapshots that are not exactly on a grid.
- Zero "peek ahead": uses as-of join at t+h with BACKWARD strategy.
- Allows small drift around the horizon via tolerance = horizon_secs + slack.
- Drops rows with no matching future snapshot (no NaN labels).
- Produces:
    - pm_delta_ticks (float): future minus current ticks (or approx from ltp)
    - pm_up (int8): 1 if delta >= tick_threshold else 0

Tune `slack_secs` (default 3) if your collector is noisier or tighter.
"""
from __future__ import annotations

from typing import Optional
import polars as pl


def _detect_ts_col(df: pl.DataFrame) -> str:
    for c in ("ts", "ts_ms", "publishTimeMs"):
        if c in df.columns:
            return c
    raise KeyError("No timestamp column found (expected one of 'ts', 'ts_ms', 'publishTimeMs').")


def _ensure_ts_seconds(df: pl.DataFrame) -> pl.DataFrame:
    c = _detect_ts_col(df)
    if c == "ts":
        df2 = df.with_columns(pl.col(c).cast(pl.Datetime(time_unit="us")).alias("ts_dt"))
        return df2.with_columns((pl.col("ts_dt").cast(pl.Int64) // 1_000_000).alias("ts_s"))
    elif c in ("ts_ms", "publishTimeMs"):
        return df.with_columns(((pl.col(c).cast(pl.Int64)) // 1000).alias("ts_s"))
    else:  # pragma: no cover
        raise AssertionError


def add_price_move_labels(
    df: pl.DataFrame,
    horizon_secs: int = 60,
    tick_threshold: int = 1,
    slack_secs: int = 3,
    clip_abs_ticks: Optional[int] = 200,  # guard against absurd moves if data is sparse
) -> pl.DataFrame:
    """Return df with price-move labels joined in without leakage.

    Strategy:
      - Let t be the current snapshot time (seconds).
      - We target t+h (horizon_secs), but snapshots are irregular, so
        we join to the *latest snapshot <= t+h* within tolerance (h + slack).
      - Use join_asof BACKWARD with tolerance = horizon_secs + slack_secs.

    Rows with no matching future snapshot are dropped to avoid NaNs.
    """
    if "ltpTick" not in df.columns and "ltp" not in df.columns:
        raise ValueError("Expected ltpTick or ltp in features for price-move labels.")

    tol = int(horizon_secs + max(0, slack_secs))

    base = _ensure_ts_seconds(df)

    right_cols = [
        pl.col("marketId"),
        pl.col("selectionId"),
        pl.col("ts_s").alias("ts_s_right"),
    ]

    use_ticks = "ltpTick" in base.columns
    if use_ticks:
        right_cols.append(pl.col("ltpTick").alias("ltpTick_fut"))
    else:
        right_cols.append(pl.col("ltp").alias("ltp_fut"))

    right = base.select(right_cols).sort(["marketId", "selectionId", "ts_s_right"])  # required sorted

    left = base.with_columns((pl.col("ts_s") + horizon_secs).alias("ts_s_join")).sort(
        ["marketId", "selectionId", "ts_s_join"]
    )

    joined = left.join_asof(
        right,
        left_on="ts_s_join",
        right_on="ts_s_right",
        strategy="backward",  # only snapshots at or BEFORE t+h
        by=["marketId", "selectionId"],
        tolerance=tol,
    )

    # keep only futures that are close to the target horizon
    # delta_sec = (t+h) - matched_future_time  ∈ [0, slack_secs]
    joined = joined.with_columns(
        (pl.col("ts_s_join") - pl.col("ts_s_right")).alias("future_delta_sec")
    ).filter(
        (pl.col("future_delta_sec") >= 0) & (pl.col("future_delta_sec") <= slack_secs)
    )

    # compute delta in ticks (or approximate from ltp buckets)
    if use_ticks:
        delta_ticks = (pl.col("ltpTick_fut") - pl.col("ltpTick"))
    else:
        # approximate mapping: Betfair tick ladders simplified
        delta_ticks = (pl.col("ltp_fut") - pl.col("ltp")) / pl.when(pl.col("ltp") <= 2).then(0.01) \
                                                      .when(pl.col("ltp") <= 3).then(0.02) \
                                                      .otherwise(0.1)

    out = joined.with_columns([
        delta_ticks.alias("pm_delta_ticks"),
    ]).filter(
        pl.col("pm_delta_ticks").is_not_null()
    )

    if clip_abs_ticks is not None:
        out = out.with_columns(
            pl.col("pm_delta_ticks").clip(-clip_abs_ticks, clip_abs_ticks).alias("pm_delta_ticks")
        )

    out = out.with_columns(
        (pl.col("pm_delta_ticks") >= tick_threshold).cast(pl.Int8).alias("pm_up")
    )

    return out
