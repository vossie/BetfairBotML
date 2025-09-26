#!/usr/bin/env python3
import os, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl

UTC = timezone.utc

def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def implied_prob_from_ltp_expr(col: str = "ltp") -> pl.Expr:
    # p = 1 / max(ltp, 1e-12)
    return (1.0 / pl.col(col).clip(1e-12, None)).alias("__p_now")

def time_to_off_minutes_expr() -> pl.Expr:
    # requires marketStartMs and publishTimeMs in frame
    return ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off")

def filter_preoff(df: pl.DataFrame, max_mins: int) -> pl.DataFrame:
    return df.filter((pl.col("mins_to_off") >= 0.0) & (pl.col("mins_to_off") <= float(max_mins)))

def _scan_cols(glob_pat: str) -> list[str]:
    return pl.scan_parquet(glob_pat).collect_schema().names()

def read_snapshots(curated: Path, start: datetime, end: datetime, sport: str) -> pl.LazyFrame:
    need_cols = [
        "sport","marketId","selectionId","publishTimeMs",
        "ltp","tradedVolume","spreadTicks","imbalanceBest1",
        "marketStartMs",
    ]
    snap_glob = str(curated / "orderbook_snapshots_5s" / f"sport={sport}" / "date=*" / "*.parquet")
    snap_cols = set(_scan_cols(snap_glob))
    sel_cols = [c for c in need_cols if c in snap_cols]

    lf_snap = pl.scan_parquet(snap_glob).select(sel_cols)

    # Join in marketStartMs from defs if missing in snapshots
    if "marketStartMs" not in sel_cols:
        defs_glob = str(curated / "market_definitions" / f"sport={sport}" / "date=*" / "*.parquet")
        try:
            defs_cols = set(_scan_cols(defs_glob))
            if {"sport","marketId","marketStartMs"} <= defs_cols:
                lf_defs = (
                    pl.scan_parquet(defs_glob)
                    .select(["sport","marketId","marketStartMs"])
                    .unique(stable=True, maintain_order=True)
                )
                lf_snap = lf_snap.join(lf_defs, on=["sport","marketId"], how="left")
        except Exception:
            pass

    # Time filter by publishTimeMs
    start_ms, end_ms = to_ms(start), to_ms(end + timedelta(days=1))
    lf = lf_snap.filter((pl.col("publishTimeMs") >= start_ms) & (pl.col("publishTimeMs") < end_ms))

    # Cast numerics consistently
    have = set(lf.collect_schema().names())
    proj: list[pl.Expr] = []
    for c in ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","marketStartMs"]:
        if c not in have:
            continue
        if c in ("sport","marketId"):
            proj.append(pl.col(c))
        elif c == "selectionId":
            proj.append(pl.col(c).cast(pl.Int64))
        else:
            proj.append(pl.col(c).cast(pl.Float64))
    lf = lf.select(proj)
    return lf

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def kelly_fraction_back(p_true: float, price: float, commission: float) -> float:
    # Fraction of bankroll to stake on back (approx)
    b = price - 1.0
    if b <= 0.0:
        return 0.0
    q = 1.0 - p_true
    # net win prob after commission on profit
    return max(0.0, (p_true * (1.0 - commission) - q / b))

def kelly_fraction_lay(p_true: float, price: float, commission: float) -> float:
    # Fraction of liability to risk when laying (heuristic)
    b = price - 1.0
    if b <= 0.0:
        return 0.0
    q = 1.0 - p_true
    return max(0.0, (q - p_true * b) / b)
