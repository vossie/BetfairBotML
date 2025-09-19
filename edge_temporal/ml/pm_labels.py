#!/usr/bin/env python3
"""
Price-move label builder with strict anti-leak guard.
- Joins as-of at t+h (BACKWARD, slack)
- Filters to futures within [0, slack_secs]
- Drops *_fut, *_right, *_join, future_* helper cols
"""
from __future__ import annotations
import polars as pl
from typing import Optional


def _detect_ts_col(df: pl.DataFrame) -> str:
    for c in ("ts", "ts_ms", "publishTimeMs"):
        if c in df.columns:
            return c
    raise KeyError("No timestamp column found (expected one of ts/ts_ms/publishTimeMs).")


def _ensure_ts_seconds(df: pl.DataFrame) -> pl.DataFrame:
    c = _detect_ts_col(df)
    if c == "ts":
        df2 = df.with_columns(pl.col(c).cast(pl.Datetime(time_unit="us")).alias("ts_dt"))
        return df2.with_columns((pl.col("ts_dt").cast(pl.Int64) // 1_000_000).alias("ts_s"))
    elif c in ("ts_ms", "publishTimeMs"):
        return df.with_columns(((pl.col(c).cast(pl.Int64)) // 1000).alias("ts_s"))
    else:
        raise AssertionError


def add_price_move_labels(
    df: pl.DataFrame,
    horizon_secs: int = 60,
    tick_threshold: int = 1,
    slack_secs: int = 3,
    clip_abs_ticks: Optional[int] = 200,
) -> pl.DataFrame:
    if "ltpTick" not in df.columns and "ltp" not in df.columns:
        raise ValueError("Need ltpTick or ltp for price-move labels.")

    tol = int(horizon_secs + max(0, slack_secs))
    base = _ensure_ts_seconds(df)

    right_cols = [pl.col("marketId"), pl.col("selectionId"), pl.col("ts_s").alias("ts_s_right")]
    if "ltpTick" in base.columns:
        right_cols.append(pl.col("ltpTick").alias("ltpTick_fut"))
    else:
        right_cols.append(pl.col("ltp").alias("ltp_fut"))
    right = base.select(right_cols).sort(["marketId", "selectionId", "ts_s_right"])

    left = base.with_columns((pl.col("ts_s") + horizon_secs).alias("ts_s_join")).sort(
        ["marketId", "selectionId", "ts_s_join"]
    )

    joined = left.join_asof(
        right,
        left_on="ts_s_join",
        right_on="ts_s_right",
        strategy="backward",
        by=["marketId", "selectionId"],
        tolerance=tol,
    )

    joined = joined.with_columns(
        (pl.col("ts_s_join") - pl.col("ts_s_right")).alias("future_delta_sec")
    ).filter(
        (pl.col("future_delta_sec") >= 0) & (pl.col("future_delta_sec") <= slack_secs)
    )

    if "ltpTick_fut" in joined.columns:
        delta_ticks = pl.col("ltpTick_fut") - pl.col("ltpTick")
    else:
        delta_ticks = (pl.col("ltp_fut") - pl.col("ltp")) / pl.when(pl.col("ltp") <= 2).then(0.01) \
                                                       .when(pl.col("ltp") <= 3).then(0.02) \
                                                       .otherwise(0.1)

    out = joined.with_columns(delta_ticks.alias("pm_delta_ticks"))
    if clip_abs_ticks is not None:
        out = out.with_columns(pl.col("pm_delta_ticks").clip(-clip_abs_ticks, clip_abs_ticks))

    out = out.with_columns((pl.col("pm_delta_ticks") >= tick_threshold).cast(pl.Int8).alias("pm_up"))

    # drop leakage-prone helper columns
    drop_cols = [c for c in out.columns if c.endswith("_fut")
                 or c.endswith("_right")
                 or c.endswith("_join")
                 or c.startswith("future_")]
    out = out.drop(drop_cols)
    return out
