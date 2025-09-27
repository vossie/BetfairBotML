#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone, date
import json
import os
import tempfile

import polars as pl

UTC = timezone.utc

# --------------------------------------------------------------------
# Dates & IO
# --------------------------------------------------------------------

def parse_date(s: str) -> date:
    """
    Parse 'YYYY-MM-DD' -> datetime.date (naive, calendar date).
    """
    return datetime.strptime(s, "%Y-%m-%d").date()

def write_json(path: Path | str, payload: dict) -> None:
    """
    Atomic-ish JSON write to avoid partial files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


# --------------------------------------------------------------------
# Market time helpers
# --------------------------------------------------------------------

def time_to_off_minutes_expr() -> pl.Expr:
    """
    Polars expression computing minutes to scheduled off:
    (marketStartMs - publishTimeMs) / 60_000, clipped to >= -1e6 if missing.
    Returns Float64 column named 'mins_to_off'.
    """
    return (
        ((pl.col("marketStartMs").cast(pl.Int64) - pl.col("publishTimeMs").cast(pl.Int64)) / 60000.0)
        .alias("mins_to_off")
    )

def filter_preoff(df: pl.DataFrame, preoff_max_minutes: int) -> pl.DataFrame:
    """
    Keep rows in the 0..preoff_max_minutes window (i.e., before the off).
    Drops rows with missing mins_to_off.
    """
    return df.filter(
        pl.col("mins_to_off").is_not_null() &
        (pl.col("mins_to_off") >= 0.0) &
        (pl.col("mins_to_off") <= float(preoff_max_minutes))
    )


# --------------------------------------------------------------------
# Snapshot reader (column-pruned, joins definitions, optional country filter)
# --------------------------------------------------------------------

def _date_range(start_d: date, end_d: date) -> list[str]:
    days = []
    d = start_d
    while d <= end_d:
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days

def read_snapshots(
    curated_root: Path | str,
    start_dt: date,
    end_dt: date,
    sport: str,
    columns: list[str] | None = None,
    country_code: str | None = None,
) -> pl.LazyFrame:
    """
    Returns a LazyFrame with snapshots joined to market definitions.

    - Reads per-day partitions from:
        orderbook_snapshots_5s/sport=<sport>/date=<YYYY-MM-DD>/*.parquet
        market_definitions/sport=<sport>/date=<YYYY-MM-DD>/*.parquet
    - Joins on marketId to bring marketStartMs (+ countryCode).
    - If 'columns' is provided, prunes snapshot columns at scan time to reduce IO.
      IMPORTANT: If columns include any of the book arrays (backTicks/backSizes/layTicks/laySizes),
                 they will be projected when present. This is required for liquidity enforcement.
    - If 'country_code' is provided (e.g., 'GB'), rows are filtered to that country.
      Use None or '' to disable country filtering.

    Output schema includes at least:
        marketId (Utf8), selectionId (Int64), publishTimeMs (Int64), ltp (Float),
        tradedVolume, spreadTicks, imbalanceBest1, marketStartMs (Int64),
        optionally the book arrays if requested & present.
    """
    curated_root = Path(curated_root)
    days = _date_range(start_dt, end_dt)

    snap_lfs: list[pl.LazyFrame] = []
    want_book = False
    if columns:
        need = {"backTicks", "backSizes", "layTicks", "laySizes"}
        want_book = any(c in need for c in columns)

    for day in days:
        # Snapshots
        snaps = pl.scan_parquet(
            str(curated_root / f"orderbook_snapshots_5s/sport={sport}/date={day}/*.parquet"),
            low_memory=True,
        )

        # Project only requested columns if provided; else keep as-is
        if columns:
            # Only keep columns that are actually present in the file schema
            schema_names = set(snaps.collect_schema().names())
            keep_cols = [c for c in columns if c in schema_names]
            if keep_cols:
                snaps = snaps.select(keep_cols)

        # Ensure core columns are present/cast
        snaps = snaps.with_columns([
            pl.col("marketId").cast(pl.Utf8),
            pl.col("selectionId").cast(pl.Int64),
            pl.col("publishTimeMs").cast(pl.Int64),
        ])

        # Market definitions (bring marketStartMs + countryCode)
        defs = (
            pl.scan_parquet(
                str(curated_root / f"market_definitions/sport={sport}/date={day}/*.parquet"),
                low_memory=True,
            )
            .select([
                pl.col("marketId").cast(pl.Utf8),
                pl.col("marketStartMs").cast(pl.Int64),
                pl.col("countryCode").cast(pl.Utf8),
            ])
        )

        lf = snaps.join(defs, on="marketId", how="inner")

        # Filter by country if requested
        if country_code and len(country_code) > 0:
            lf = lf.filter(pl.col("countryCode") == country_code)

        snap_lfs.append(lf)

    if not snap_lfs:
        # Empty LF with expected columns so downstream ops don't break.
        # Note: we do not include book arrays by default here.
        empty_schema = {
            "sport": pl.Utf8,
            "marketId": pl.Utf8,
            "selectionId": pl.Int64,
            "publishTimeMs": pl.Int64,
            "ltp": pl.Float64,
            "tradedVolume": pl.Float64,
            "spreadTicks": pl.Int64,
            "imbalanceBest1": pl.Float64,
            "marketStartMs": pl.Int64,
            "countryCode": pl.Utf8,
        }
        if want_book:
            empty_schema.update({
                "backTicks": pl.List(pl.Int32),
                "backSizes": pl.List(pl.Float64),
                "layTicks": pl.List(pl.Int32),
                "laySizes": pl.List(pl.Float64),
            })
        return pl.LazyFrame(schema=empty_schema)

    # Vertical concat is still lazy; caller decides select/filter/collect
    return pl.concat(snap_lfs, how="vertical_relaxed")


# --------------------------------------------------------------------
# Kelly sizing helpers (approximate exchange commission handling)
# --------------------------------------------------------------------

def _net_back_odds(odds: float, commission: float) -> float:
    """
    Net odds for a BACK bet after exchange commission on winnings.
    """
    return max(0.0, (odds - 1.0) * (1.0 - commission))

def kelly_fraction_back(p_win: float, odds: float, commission: float) -> float:
    """
    Kelly fraction for BACK bets:
      f* = (b*p - q) / b,  with b = net_back_odds, q = 1 - p
    Clamps to [0, +inf) (caller should cap).
    """
    p = max(0.0, min(1.0, float(p_win)))
    q = 1.0 - p
    b = _net_back_odds(float(odds), float(commission))
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    return max(0.0, f)

def kelly_fraction_lay(p_win: float, odds: float, commission: float) -> float:
    """
    Kelly fraction for LAY bets – in terms of fraction of bankroll staked (not liability).
    For a LAY at odds o:
      stake S, liability L = (o-1)*S.
    Expected return per £ staked (ignoring commission) is:
      E = (1 - p) - p*(o - 1)
    Kelly fraction f (per £ stake) is (E / (o - 1)) approximately,
    then we apply a simple commission adjustment on the 'win' outcome (when favourite loses).
    We clamp negative results to 0.
    """
    p = max(0.0, min(1.0, float(p_win)))
    o = max(1.01, float(odds))
    b = o - 1.0
    # Approximate commission: when we "win" a lay (runner loses),
    # we keep the backer's stake minus commission; approximate scaling:
    keep_factor = (1.0 - float(commission))
    e = keep_factor * (1.0 - p) - p * b
    if b <= 0:
        return 0.0
    f = e / b
    return max(0.0, f)
