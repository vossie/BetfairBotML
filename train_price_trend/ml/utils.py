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
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

# ---------- robust path scanners ----------
def _scan_paths(root: Path, rels: list[str]) -> str:
    # returns a glob string that may match multiple partitions
    for rel in rels:
        p = root / rel
        # keep it as a glob even if not existing now (Polars lazy handles it)
        return str(p / "*.parquet")
    # fallback (never hit)
    return str(root / "**/*.parquet")

def _snap_glob(root: Path, sport: str) -> str:
    return _scan_paths(
        root,
        [f"orderbook_snapshots_5s/sport={sport}/date=*"]
    )

def _defs_glob(root: Path, sport: str) -> str:
    # try common layouts
    candidates = [
        f"market_definitions/sport={sport}/date=*",
        f"marketdefinitions/sport={sport}/date=*",
        f"market_definitions/date=*/sport={sport}",
    ]
    return _scan_paths(root, candidates)

# ---------- expressions ----------
def implied_prob_from_ltp_expr(col: str) -> pl.Expr:
    return (1.0 / pl.when(pl.col(col) < 1e-12).then(1e-12).otherwise(pl.col(col))).alias("__p_now")

def time_to_off_minutes_expr() -> pl.Expr:
    return ((pl.col("marketStartMs").cast(pl.Int64) - pl.col("publishTimeMs").cast(pl.Int64)) / 60000.0).alias("mins_to_off")

# ---------- IO ----------
def read_defs(curated_root: Path, start_dt: datetime, end_dt: datetime, sport: str) -> pl.LazyFrame:
    ms_lo = to_ms(start_dt)
    ms_hi = to_ms(end_dt + timedelta(days=1))
    defs_glob = _defs_glob(curated_root, sport)
    lf = (
        pl.scan_parquet(defs_glob)
        .select([
            pl.col("sport"),
            pl.col("marketId"),
            pl.col("marketStartMs").cast(pl.Int64)
        ])
        .filter(pl.col("sport") == sport)
        .filter((pl.col("marketStartMs") >= ms_lo) & (pl.col("marketStartMs") < ms_hi))
        .unique(subset=["marketId"], keep="first")
    )
    return lf

def read_snapshots(curated_root: Path, start_dt: datetime, end_dt: datetime, sport: str) -> pl.LazyFrame:
    ms_lo = to_ms(start_dt)
    ms_hi = to_ms(end_dt + timedelta(days=1))
    snap_glob = _snap_glob(curated_root, sport)

    # load minimal numeric snapshot columns
    base = (
        pl.scan_parquet(snap_glob)
        .select([
            pl.col("sport"),
            pl.col("marketId"),
            pl.col("selectionId"),
            pl.col("publishTimeMs").cast(pl.Int64),
            pl.col("ltp").cast(pl.Float64),
            pl.col("tradedVolume").cast(pl.Float64),
            pl.col("spreadTicks").cast(pl.Float64),
            pl.col("imbalanceBest1").cast(pl.Float64),
        ])
        .filter(pl.col("sport") == sport)
        .filter((pl.col("publishTimeMs") >= ms_lo) & (pl.col("publishTimeMs") < ms_hi))
    )

    defs = read_defs(curated_root, start_dt, end_dt, sport)

    # left join to bring marketStartMs, then compute mins_to_off
    lf = (
        base.join(defs.select(["marketId", "marketStartMs"]), on="marketId", how="left")
        .with_columns([
            time_to_off_minutes_expr(),
            implied_prob_from_ltp_expr("ltp"),
        ])
    )
    return lf

def filter_preoff(df: pl.DataFrame, preoff_max_minutes: int) -> pl.DataFrame:
    # keep rows where we have marketStartMs (mins_to_off not null) and within window
    return df.filter(
        pl.col("mins_to_off").is_not_null()
        & (pl.col("mins_to_off") >= 0.0)
        & (pl.col("mins_to_off") <= float(preoff_max_minutes))
    )

# ---------- misc ----------
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# Kelly fractions with commission (exchange take on net win)
def kelly_fraction_back(p: float, odds: float, comm: float) -> float:
    # net payoff per £1 back bet after commission
    b = (odds - 1.0) * (1.0 - comm)
    q = 1.0 - p
    denom = b
    return max(0.0, (b * p - q) / denom) if denom > 0 else 0.0

def kelly_fraction_lay(p: float, odds: float, comm: float) -> float:
    # laying at odds -> we win £1 from backer if selection loses; commission may apply on net win
    q = 1.0 - p
    b = 1.0 * (1.0 - comm)  # win £1 (approx) after comm when selection loses
    denom = odds - 1.0      # exposure per £1 lay is (odds-1)
    return max(0.0, (b * q - p) / denom) if denom > 0 else 0.0
