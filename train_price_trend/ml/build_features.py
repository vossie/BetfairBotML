#!/usr/bin/env python3
import polars as pl

# assuming ~5s cadence snapshots; adjust if you downsample differently
LAG_30S = 6
LAG_60S = 12
LAG_120S = 24

def add_core_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["marketId","selectionId","publishTimeMs"])

    def lag_set(col: str):
        return [
            pl.col(col).shift(LAG_30S).alias(f"{col}_lag30s"),
            pl.col(col).shift(LAG_60S).alias(f"{col}_lag60s"),
            pl.col(col).shift(LAG_120S).alias(f"{col}_lag120s"),
        ]

    df = df.with_columns([
        (pl.col("ltp") - pl.col("ltp").shift(1)).alias("ltp_diff_5s"),
        (pl.col("tradedVolume") - pl.col("tradedVolume").shift(1)).alias("vol_diff_5s"),
        *lag_set("ltp"),
        *lag_set("tradedVolume"),
    ])

    df = df.with_columns([
        (pl.col("ltp") - pl.col("ltp_lag30s")).alias("ltp_mom_30s"),
        (pl.col("ltp") - pl.col("ltp_lag60s")).alias("ltp_mom_60s"),
        (pl.col("ltp") - pl.col("ltp_lag120s")).alias("ltp_mom_120s"),
        ((pl.col("ltp") - pl.col("ltp_lag30s")) / pl.col("ltp_lag30s")).alias("ltp_ret_30s"),
        ((pl.col("ltp") - pl.col("ltp_lag60s")) / pl.col("ltp_lag60s")).alias("ltp_ret_60s"),
        ((pl.col("ltp") - pl.col("ltp_lag120s")) / pl.col("ltp_lag120s")).alias("ltp_ret_120s"),
        (pl.col("tradedVolume") - pl.col("tradedVolume_lag30s")).alias("vol_mom_30s"),
        (pl.col("tradedVolume") - pl.col("tradedVolume_lag60s")).alias("vol_mom_60s"),
        (pl.col("tradedVolume") - pl.col("tradedVolume_lag120s")).alias("vol_mom_120s"),
    ])

    # sanitize non-finite to nulls
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in df.columns
    ])
    return df

def add_label_delta_prob(df: pl.DataFrame, horizon_secs: int) -> pl.DataFrame:
    step = max(1, int(round(horizon_secs / 5.0)))
    df = df.sort(["marketId","selectionId","publishTimeMs"])
    df = df.with_columns([
        (1.0 / pl.col("ltp").clip(1e-12, None)).alias("__p_now"),
        (1.0 / pl.col("ltp").shift(-step).clip(1e-12, None)).alias("__p_future"),
    ])
    return df.with_columns((pl.col("__p_future") - pl.col("__p_now")).alias("delta_p"))
