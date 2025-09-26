#!/usr/bin/env python3
import polars as pl

# We assume ~5s sampling; lags below approximate 30s/60s/120s windows.
# If your downsampling differs, adjust LAG_* accordingly.
LAG_30S = 6
LAG_60S = 12
LAG_120S = 24

def add_core_features(df: pl.DataFrame) -> pl.DataFrame:
    # Group-sort within runner stream
    df = df.sort(["marketId","selectionId","publishTimeMs"])

    # Lag joins via group context
    def with_lags(expr_name: str):
        return [
            pl.col(expr_name).shift(LAG_30S).alias(f"{expr_name}_lag30s"),
            pl.col(expr_name).shift(LAG_60S).alias(f"{expr_name}_lag60s"),
            pl.col(expr_name).shift(LAG_120S).alias(f"{expr_name}_lag120s"),
        ]

    feats = [
        # Deltas on LTP
        (pl.col("ltp") - pl.col("ltp").shift(1)).alias("ltp_diff_5s"),
        (pl.col("tradedVolume") - pl.col("tradedVolume").shift(1)).alias("vol_diff_5s"),
    ] + with_lags("ltp") + with_lags("tradedVolume")

    df = df.with_columns(feats) \
           .with_columns([
               # Momentum over windows
               (pl.col("ltp") - pl.col("ltp_lag30s")).alias("ltp_mom_30s"),
               (pl.col("ltp") - pl.col("ltp_lag60s")).alias("ltp_mom_60s"),
               (pl.col("ltp") - pl.col("ltp_lag120s")).alias("ltp_mom_120s"),
               # % change
               ((pl.col("ltp") - pl.col("ltp_lag30s")) / pl.col("ltp_lag30s")).alias("ltp_ret_30s"),
               ((pl.col("ltp") - pl.col("ltp_lag60s")) / pl.col("ltp_lag60s")).alias("ltp_ret_60s"),
               ((pl.col("ltp") - pl.col("ltp_lag120s")) / pl.col("ltp_lag120s")).alias("ltp_ret_120s"),
               # Volume momentum
               (pl.col("tradedVolume") - pl.col("tradedVolume_lag30s")).alias("vol_mom_30s"),
               (pl.col("tradedVolume") - pl.col("tradedVolume_lag60s")).alias("vol_mom_60s"),
               (pl.col("tradedVolume") - pl.col("tradedVolume_lag120s")).alias("vol_mom_120s"),
           ])

    # Replace inf/nan with nulls, to be later filled or dropped
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in df.columns
    ])
    return df

def add_label_delta_prob(df: pl.DataFrame, horizon_secs: int) -> pl.DataFrame:
    # Label is future change in implied prob over horizon.
    # Build future implied prob via shift with steps ≈ horizon/5s
    step = max(1, int(round(horizon_secs / 5)))
    df = df.sort(["marketId","selectionId","publishTimeMs"])
    df = df.with_columns([
        (1.0 / pl.col("ltp").clip_min(1e-12)).alias("__p_now"),
        (1.0 / pl.col("ltp").shift(-step).clip_min(1e-12)).alias("__p_future")
    ])
    df = df.with_columns((pl.col("__p_future") - pl.col("__p_now")).alias("delta_p"))
    return df
