# ml/features.py
from __future__ import annotations
import polars as pl
from typing import Tuple
from ml import dataio

def build_features_for_date(
    curated_root: str,
    sport: str,
    date: str,
    decision_secs_before_off: int = 5,
    commission: float = 0.05,
    maker_horizon_secs: int = 10
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    snap_t = dataio.read_table(dataio.ds_orderbook(curated_root, sport, date))
    defs_t = dataio.read_table(dataio.ds_market_defs(curated_root, sport, date))
    res_t  = dataio.read_table(dataio.ds_results(curated_root, sport, date))

    if snap_t.num_rows == 0 or defs_t.num_rows == 0 or res_t.num_rows == 0:
        return pl.DataFrame(), pl.DataFrame()

    df_snap = pl.from_arrow(snap_t)
    df_defs = pl.from_arrow(defs_t)
    df_res  = pl.from_arrow(res_t)

    df_snap = df_snap.with_columns([
        (1.0 / pl.col("ltp").cast(pl.Float64)).alias("ltp_odds"),
        (1.0 / pl.col("backTicks").list.first().cast(pl.Float64)).alias("best_back_odds"),
        (1.0 / pl.col("layTicks").list.first().cast(pl.Float64)).alias("best_lay_odds"),
        pl.col("backSizes").list.first().cast(pl.Float64).alias("best_back_size"),
        pl.col("laySizes").list.first().cast(pl.Float64).alias("best_lay_size"),
    ])
    df_snap = df_snap.with_columns([
        (1.0 / pl.col("best_back_odds")).alias("back_overround"),
        (1.0 / pl.col("best_lay_odds")).alias("lay_overround"),
    ])
    field_sizes = df_snap.group_by("marketId").agg(pl.count("selectionId").alias("field_size"))
    df_snap = df_snap.join(field_sizes, on="marketId")
    df_lbl = df_res.select(["marketId","selectionId","winLabel"])
    return df_snap, df_lbl
