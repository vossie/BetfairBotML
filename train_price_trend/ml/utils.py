#!/usr/bin/env python3
import os, json, math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl

UTC = timezone.utc

def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def implied_prob_from_ltp_expr(col="ltp"):
    # p = 1 / ltp ; guard zero
    return (1.0 / pl.col(col).clip_min(1e-12)).alias("__p_now")

def time_to_off_minutes_expr():
    # expects marketStartMs column
    return ((pl.col("marketStartMs") - pl.col("publishTimeMs")) / 60000.0).alias("mins_to_off")

def filter_preoff(df: pl.DataFrame, max_mins: int) -> pl.DataFrame:
    return df.filter((pl.col("mins_to_off") >= 0.0) & (pl.col("mins_to_off") <= float(max_mins)))

def read_snapshots(curated: Path, start: datetime, end: datetime, sport: str) -> pl.LazyFrame:
    # Only the columns we actually use
    cols = [
        "sport","marketId","selectionId","publishTimeMs",
        "ltp","tradedVolume","spreadTicks","imbalanceBest1",
        "marketStartMs"
    ]
    # scan both snapshots and defs (defs will provide marketStartMs if absent in snapshots parquet)
    snap_glob = str(curated / "orderbook_snapshots_5s" / f"sport={sport}" / "date=*" / "*.parquet")
    lf = pl.scan_parquet(snap_glob).select([c for c in cols if c in pl.scan_parquet(snap_glob).columns])

    # Some curated snapshots may miss marketStartMs; inject from market defs when available
    defs_glob = str(curated / "market_definitions" / f"sport={sport}" / "date=*" / "*.parquet")
    if Path(defs_glob.split("date=*")[0]).exists() or True:
        try:
            lf_defs = pl.scan_parquet(defs_glob).select(
                "sport","marketId","marketStartMs"
            ).unique(stable=True, maintain_order=True)
            lf = lf.join(lf_defs, on=["sport","marketId"], how="left")
        except Exception:
            pass

    # file-date partition filter
    # Expect path includes date=YYYY-MM-DD; we also time-filter on publishTimeMs
    start_ms, end_ms = to_ms(start), to_ms(end + timedelta(days=1))
    lf = lf.filter((pl.col("publishTimeMs") >= start_ms) & (pl.col("publishTimeMs") < end_ms))

    # keep numerics
    proj = []
    for c in ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","marketStartMs"]:
        if c in lf.columns:
            if c in ("sport","marketId"):
                proj.append(pl.col(c))
            else:
                proj.append(pl.col(c).cast(pl.Float64) if c not in ("selectionId",) else pl.col(c).cast(pl.Int64))
    lf = lf.select(proj)
    return lf

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def kelly_fraction_back(p_true: float, price: float, commission: float) -> float:
    # EV on £1: p*(o-1)*(1-c) - (1-p)*0 ; edge over (o-1)
    b = price - 1.0
    if b <= 0: return 0.0
    q = 1.0 - p_true
    return max(0.0, (p_true*(1.0-commission) - q/b))

def kelly_fraction_lay(p_true: float, price: float, commission: float) -> float:
    # Layer receives 1 if lose, pays (o-1) if win (ignoring tick fill nuances).
    # EV per £(backer stake) for layer: (1-p) * 1 - p*(o-1) ; fraction over liability (o-1)
    b = price - 1.0
    if b <= 0: return 0.0
    q = 1.0 - p_true
    # Convert to Kelly-like on liability:
    return max(0.0, (q - p_true*b) / b)  # heuristic

def clip_float(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else (lo if x < lo else x)
