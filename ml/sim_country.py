# ml/sim_country.py
from __future__ import annotations

import argparse
import numpy as np
import polars as pl
import xgboost as xgb
from pathlib import Path


# ----------------------------
# Utilities
# ----------------------------

def _to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return (
        df.select(cols)
          .fill_null(strategy="mean")
          .to_numpy()
          .astype(np.float32, copy=False)
    )

def _load_booster(path: str) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(path)
    return bst

def _implied_prob() -> pl.Expr:
    return 1.0 / pl.col("ltp")

def _kelly_stake(edge: np.ndarray, prob: np.ndarray, kelly: float) -> np.ndarray:
    """Kelly fraction stake sizing"""
    return np.clip(kelly * (edge / (1 - prob)), 0, 1)

def _add_country_feat(df: pl.DataFrame) -> pl.DataFrame:
    if "country_feat" in df.columns:
        return df
    if "countryCode" in df.columns:
        return df.with_columns(
            pl.col("countryCode")
              .fill_null("UNK")
              .cast(pl.Categorical)
              .to_physical()
              .alias("country_feat")
        )
    return df


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser("Simulate bets with country feature")
    ap.add_argument("--model", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--days-before", type=int, default=7)
    ap.add_argument("--preoff-mins", type=int, default=180)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=None)
    ap.add_argument("--stake-cap-day", type=float, default=None)
    ap.add_argument("--country-filter", type=str, default=None)
    ap.add_argument("--pnl-by-country-out", type=str, default=None)
    ap.add_argument("--bets-out", type=str, default=None)
    args = ap.parse_args()

    # load model + features
    bst = _load_booster(args.model)
    feats = bst.feature_names

    # --- build features
    from . import features
    dates = [args.date]  # simplified; replace with date range if needed
    df, _ = features.build_features_streaming(
        curated_root=args.curated,
        sport=args.sport,
        dates=dates,
        preoff_minutes=args.preoff_mins,
        batch_markets=args.batch_markets,
        downsample_secs=(args.downsample_secs or None),
    )

    if args.country_filter:
        df = df.filter(pl.col("countryCode") == args.country_filter)

    # add numeric country feature
    df = _add_country_feat(df)

    # ensure features exist
    X = _to_numpy(df, feats)
    p = bst.predict(xgb.DMatrix(X, feature_names=feats))

    # implied prob and edge
    df = df.with_columns([
        _implied_prob().alias("implied_prob"),
        pl.Series("model_pred", p),
    ])
    df = df.with_columns((pl.col("model_pred") - pl.col("implied_prob")).alias("edge"))

    # filter edge
    df = df.filter(pl.col("edge") >= args.min_edge)

    # --- Polars 1.33 safe: rank per marketId
    df = (
        df.sort(["marketId", "edge"], descending=[False, True])
          .with_columns(pl.arange(0, pl.count()).over("marketId").alias("_rn"))
          .filter(pl.col("_rn") < args.top_n_per_market)
          .drop("_rn")
    )

    # stake via Kelly
    kelly_frac = _kelly_stake(
        edge=df["edge"].to_numpy(),
        prob=df["model_pred"].to_numpy(),
        kelly=args.kelly,
    )
    stake = np.where(kelly_frac > 0, 1.0 * kelly_frac, 0.0)  # base stake=1
    df = df.with_columns(pl.Series("stake", stake))

    # pnl (gross – commission)
    df = df.with_columns(
        (pl.col("stake") * (pl.col("model_pred") - args.commission)).alias("pnl")
    )

    # save bets
    if args.bets_out:
        df.write_csv(args.bets_out)
        print(f"Saved bets -> {args.bets_out}")

    # aggregate PnL by country
    if args.pnl_by_country_out:
        pnl_by_country = (
            df.group_by("countryCode")
              .agg([
                  pl.count().alias("n_bets"),
                  pl.sum("stake").alias("stake"),
                  pl.sum("pnl").alias("pnl"),
                  (pl.sum("pnl") / pl.sum("stake")).alias("roi"),
              ])
        )
        pnl_by_country.write_csv(args.pnl_by_country_out)
        print(f"Saved PnL by country -> {args.pnl_by_country_out}")

    # summary
    n_bets = df.height
    stake_total = float(df["stake"].sum())
    pnl_total = float(df["pnl"].sum())
    roi_total = pnl_total / stake_total if stake_total > 0 else 0
    print(
        f"Summary: n_bets={n_bets} stake={stake_total:.2f} "
        f"pnl={pnl_total:.2f} ROI={roi_total*100:.3f}%"
    )


if __name__ == "__main__":
    main()
