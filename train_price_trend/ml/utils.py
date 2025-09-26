#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

UTC = timezone.utc

# ---------- time helpers ----------
def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD into a UTC datetime (00:00:00)."""
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)

def to_ms(dt: datetime) -> int:
    """UTC datetime -> epoch milliseconds."""
    return int(dt.timestamp() * 1000)

# ---------- robust path scanners ----------
def _scan_paths(root: Path, rels: list[str]) -> str:
    """
    Return a glob string for the first plausible layout.
    We intentionally keep it as a glob even if it doesn't exist yet—Polars Lazy handles it.
    """
    for rel in rels:
        p = root / rel
        return str(p / "*.parquet")
    return str(root / "**/*.parquet")

def _snap_glob(root: Path, sport: str) -> str:
    # expected curated layout
    return _scan_paths(root, [f"orderbook_snapshots_5s/sport={sport}/date=*"])

def _defs_glob(root: Path, sport: str) -> str:
    # allow a few common layouts for market definitions
    candidates = [
        f"market_definitions/sport={sport}/date=*",
        f"marketdefinitions/sport={sport}/date=*",
        f"market_definitions/date=*/sport={sport}",
    ]
    return _scan_paths(root, candidates)

# ---------- expressions ----------
def implied_prob_from_ltp_expr(col: str) -> pl.Expr:
    """
    Return implied probability 1/ltp with a small floor to avoid inf / NaN.
    Result column name: "__p_now".
    """
    return (1.0 / pl.when(pl.col(col) < 1e-12).then(1e-12).otherwise(pl.col(col))).alias("__p_now")

def time_to_off_minutes_expr() -> pl.Expr:
    """
    (marketStartMs - publishTimeMs) / 60_000 → minutes to off.
    Result column name: "mins_to_off".
    """
    return (
        (pl.col("marketStartMs").cast(pl.Int64) - pl.col("publishTimeMs").cast(pl.Int64)) / 60000.0
    ).alias("mins_to_off")

# ---------- IO ----------
def read_defs(curated_root: Path, start_dt: datetime, end_dt: datetime, sport: str) -> pl.LazyFrame:
    """
    Read market definitions, keep minimal columns and unique marketId,
    restricted to the required time window and sport.
    """
    ms_lo = to_ms(start_dt)
    ms_hi = to_ms(end_dt + timedelta(days=1))
    defs_glob = _defs_glob(curated_root, sport)
    lf = (
        pl.scan_parquet(defs_glob)
        .select([
            pl.col("sport"),
            pl.col("marketId"),
            pl.col("marketStartMs").cast(pl.Int64),
        ])
        .filter(pl.col("sport") == sport)
        .filter((pl.col("marketStartMs") >= ms_lo) & (pl.col("marketStartMs") < ms_hi))
        .unique(subset=["marketId"], keep="first")
    )
    return lf

def read_snapshots(curated_root: Path, start_dt: datetime, end_dt: datetime, sport: str) -> pl.LazyFrame:
    """
    Read order book snapshots for a sport and date window, join marketStartMs,
    and add mins_to_off + __p_now. Microstructure columns are null-safe.
    """
    ms_lo = to_ms(start_dt)
    ms_hi = to_ms(end_dt + timedelta(days=1))
    snap_glob = _snap_glob(curated_root, sport)

    # Load minimal numeric snapshot columns. Keep integer precision where appropriate.
    base = (
        pl.scan_parquet(snap_glob)
        .select([
            pl.col("sport"),
            pl.col("marketId"),
            pl.col("selectionId").cast(pl.Int64),
            pl.col("publishTimeMs").cast(pl.Int64),
            pl.col("ltp").cast(pl.Float64),             # optional in Avro → may be null
            pl.col("tradedVolume").cast(pl.Float64),    # optional in Avro → may be null
            pl.col("spreadTicks").cast(pl.Int32),       # keep as Int32 for precision
            pl.col("imbalanceBest1").cast(pl.Float64),  # optional → may be null
        ])
        .filter(pl.col("sport") == sport)
        .filter((pl.col("publishTimeMs") >= ms_lo) & (pl.col("publishTimeMs") < ms_hi))
    )

    defs = read_defs(curated_root, start_dt, end_dt, sport)

    # Left join to bring marketStartMs; compute mins_to_off and __p_now.
    # Make microstructure columns null-safe: fill spreadTicks/imbalanceBest1 defaults.
    lf = (
        base.join(defs.select(["marketId", "marketStartMs"]), on="marketId", how="left")
        .with_columns([
            # microstructure: fill defaults
            pl.col("spreadTicks").fill_null(0).cast(pl.Float64),
            pl.col("imbalanceBest1").fill_null(0.0),

            # derived columns
            time_to_off_minutes_expr(),
            implied_prob_from_ltp_expr("ltp"),
        ])
    )
    return lf

def filter_preoff(df: pl.DataFrame, preoff_max_minutes: int) -> pl.DataFrame:
    """
    Keep rows with known marketStartMs (mins_to_off not null) and within [0, preoff_max_minutes].
    """
    return df.filter(
        pl.col("mins_to_off").is_not_null()
        & (pl.col("mins_to_off") >= 0.0)
        & (pl.col("mins_to_off") <= float(preoff_max_minutes))
    )

# ---------- misc ----------
def write_json(path: Path, obj) -> None:
    """Write JSON with pretty indentation, ensuring parent dir exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# Kelly fractions with commission (exchange take on net win)
def kelly_fraction_back(p: float, odds: float, comm: float) -> float:
    """
    Kelly fraction for a back bet with exchange commission on net win.
    Returns fraction of bankroll to stake. Clipped to [0, ∞) by caller.
    """
    b = (odds - 1.0) * (1.0 - comm)  # net payoff per £1 when selection wins
    q = 1.0 - p
    denom = b
    return max(0.0, (b * p - q) / denom) if denom > 0 else 0.0

def kelly_fraction_lay(p: float, odds: float, comm: float) -> float:
    """
    Kelly fraction for a lay bet (exposure is odds-1), with exchange commission on net win.
    We approximate the net win as £1*(1-comm) when selection loses.
    """
    q = 1.0 - p
    b = 1.0 * (1.0 - comm)  # net win ~£1 after commission when selection loses
    denom = odds - 1.0      # exposure per £1 lay
    return max(0.0, (b * q - p) / denom) if denom > 0 else 0.0
