import argparse
import polars as pl
import xgboost as xgb
import cupy as cp
import numpy as np
import features
import os

# ---------------- stake cap ----------------
def _cap_stakes(bets: pl.DataFrame, stake_cap_market: float, stake_cap_day: float) -> pl.DataFrame:
    """Cap stake exposure per market and per day"""
    if bets.is_empty():
        return bets

    # Cap per market
    bets = (
        bets
        .with_columns(
            pl.col("stake_unit")
            .cum_sum().over("marketId")
            .alias("cum_stake_mkt")
        )
        .with_columns(
            pl.when(pl.col("cum_stake_mkt") > stake_cap_market)
            .then(pl.lit(0.0))
            .otherwise(pl.col("stake_unit"))
            .alias("stake_unit_capped_mkt")
        )
        .drop("cum_stake_mkt")
        .with_columns(pl.col("stake_unit_capped_mkt").alias("stake_unit"))
        .drop("stake_unit_capped_mkt")
    )

    # Cap per day
    bets = (
        bets
        .with_columns(
            pl.col("stake_unit")
            .cum_sum().over("event_date")
            .alias("cum_stake_day")
        )
        .with_columns(
            pl.when(pl.col("cum_stake_day") > stake_cap_day)
            .then(pl.lit(0.0))
            .otherwise(pl.col("stake_unit"))
            .alias("stake_unit_capped_day")
        )
        .drop("cum_stake_day")
        .with_columns(pl.col("stake_unit_capped_day").alias("stake_unit"))
        .drop("stake_unit_capped_day")
    )

    return bets


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--side", choices=["auto", "back", "lay"], default="auto")
    ap.add_argument("--bets-out", required=True)
    ap.add_argument("--stake-cap-market", type=float, default=5.0)
    ap.add_argument("--stake-cap-day", type=float, default=20.0)
    args = ap.parse_args()

    # Load features
    print(f"Building features for sim from {args.curated} {args.sport} {args.date} ± {args.days} days")
    df_feat, _ = features.build_features_streaming(
        args.curated, args.sport,
        [args.date], args.preoff_mins,
        batch_markets=100,
        downsample_secs=0,
    )

    # Load model
    booster = xgb.Booster()
    booster.load_model(args.model)

    # Features
    feature_cols = [c for c in df_feat.columns if c not in ("winLabel", "marketId", "selectionId")]
    X = df_feat.select(feature_cols).fill_null(strategy="mean").to_numpy()
    X = cp.asarray(X, dtype=cp.float32)

    dmat = xgb.DMatrix(X)
    probs = booster.predict(dmat)

    df = df_feat.with_columns(pl.Series("pred", probs.get()))

    # Expected value vs odds
    df = df.with_columns([
        (pl.col("pred") * (1 - args.commission)).alias("p_win"),
        (1 / pl.col("ltp")).alias("implied_prob"),
    ])
    df = df.with_columns([
        (pl.col("p_win") - pl.col("implied_prob")).alias("edge")
    ])

    # Filter bets
    bets = (
        df.filter(pl.col("edge") >= args.min_edge)
        .with_columns(
            (args.kelly * pl.col("edge") / pl.col("implied_prob")).alias("stake_unit")
        )
    )

    # Add event_date from marketStartMs
    if "marketStartMs" in bets.columns:
        bets = bets.with_columns(
            (pl.col("marketStartMs") // 86400000).alias("event_date")
        )

    # Cap stakes
    bets = _cap_stakes(bets, args.stake_cap_market, args.stake_cap_day)

    # Save bets
    bets.write_csv(args.bets_out)
    print(f"💾 Saved {bets.height} bets to {args.bets_out}")


if __name__ == "__main__":
    main()
