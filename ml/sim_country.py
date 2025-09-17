# ml/sim_country.py
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import xgboost as xgb

from . import features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers: IO / dates / numpy
# ----------------------------
def _load_booster(path: str) -> xgb.Booster:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Model not found: {p}")
    bst = xgb.Booster()
    bst.load_model(str(p))
    return bst


def _load_feature_list_for_model(model_path: Path) -> List[str]:
    feats_txt = model_path.with_suffix(".features.txt")
    if feats_txt.exists():
        return [ln.strip() for ln in feats_txt.read_text().splitlines() if ln.strip()]
    # Fallback to embedded names if present
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    if getattr(bst, "feature_names", None):
        return list(bst.feature_names)
    raise SystemExit(f"Unable to determine feature list for model: {model_path}")


def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


def _daterange(end_date_str: str, days_before: int) -> List[str]:
    """Inclusive range: [end - days_before, ..., end]."""
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days_before)
    d = start
    out: List[str] = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


# ----------------------------
# Feature engineering
# ----------------------------
def _ensure_country_feat(df: pl.DataFrame) -> pl.DataFrame:
    """Mirror trainer: derive numeric country_feat from a country code column."""
    for cand in ("countryCode", "eventCountryCode", "country"):
        if cand in df.columns:
            try:
                return df.with_columns(
                    pl.col(cand).fill_null("UNK").cast(pl.Categorical).to_physical().alias("country_feat")
                )
            except Exception:
                return df.with_columns(
                    pl.col(cand).fill_null("UNK").cast(pl.Utf8).hash().alias("country_feat")
                )
    print("WARN: no country column found; continuing without 'country_feat'.")
    return df


def _build_features_chunked(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int,
    batch_markets: int,
    downsample_secs: int | None,
    chunk_days: int,
) -> pl.DataFrame:
    parts: List[pl.DataFrame] = []
    total_rows = 0
    for i in range(0, len(dates), chunk_days):
        chunk = dates[i : i + chunk_days]
        label = f"{chunk[0]}..{chunk[-1]}" if len(chunk) > 1 else chunk[0]
        print(f"  • building features for {label}")
        df_c, raw_rows = features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=chunk,
            preoff_minutes=preoff_minutes,
            batch_markets=batch_markets,
            downsample_secs=downsample_secs,
        )
        # raw_rows may be None depending on implementation
        raw_rows = int(raw_rows or 0)
        total_rows += raw_rows if raw_rows else (df_c.height if isinstance(df_c, pl.DataFrame) else 0)
        print(f"    rows this chunk: df={df_c.height} raw={raw_rows}")
        if not df_c.is_empty():
            parts.append(df_c)
    if not parts:
        raise SystemExit("No features built.")
    df = pl.concat(parts, how="vertical", rechunk=True)
    print(f"  • total feature rows: {df.height} (raw: {total_rows})")
    return df


# ----------------------------
# Kelly staking + caps
# ----------------------------
def _kelly_fraction(prob: pl.Expr, price: pl.Expr) -> pl.Expr:
    """
    f* = ((b * p) - (1 - p)) / b  where b = price - 1.
    If b <= 0 or edge <= 0 -> 0
    """
    b = price - 1.0
    edge = (b * prob) - (1.0 - prob)
    return pl.when((b > 0) & (edge > 0)).then(edge / b).otherwise(0.0)


def _apply_caps_renorm(
    df: pl.DataFrame,
    stake_col: str,
    market_cap: float,
    day_cap: float,
    market_key: str = "marketId",
) -> pl.DataFrame:
    s = stake_col
    # Per-market cap: scale stakes inside each market to sum <= market_cap
    mkt_sums = df.group_by(market_key, maintain_order=True).agg(pl.sum(s).alias("_sum_mkt"))
    df = (
        df.join(mkt_sums, on=market_key, how="left")
          .with_columns(
              pl.when(pl.col("_sum_mkt") > market_cap)
                .then(pl.col(s) * (market_cap / pl.col("_sum_mkt")))
                .otherwise(pl.col(s))
                .alias(s)
          )
          .drop("_sum_mkt")
    )
    # Per-day cap: scale all stakes to sum <= day_cap
    sum_day = float(df.select(pl.sum(s).alias("_sum_day"))["_sum_day"][0] or 0.0)
    if day_cap > 0 and sum_day > day_cap:
        scale = day_cap / sum_day
        df = df.with_columns((pl.col(s) * scale).alias(s))
    return df


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Country-aware simulator with Kelly staking (chunked & logged).")
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

    # Bankroll and caps
    ap.add_argument("--bankroll-day", type=float, default=500.0, help="Daily bankroll used for Kelly sizing")
    ap.add_argument("--stake-cap-market", type=float, default=50.0)
    ap.add_argument("--stake-cap-day", type=float, default=500.0)

    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)

    ap.add_argument("--country-filter", default=None)
    ap.add_argument("--pnl-by-country-out", default="./output/pnl_by_country.csv")
    ap.add_argument("--bets-out", default="./output/bets_country.csv")
    args = ap.parse_args()

    # Load model + features
    model_path = Path(args.model)
    bst = _load_booster(str(model_path))
    feat_order = _load_feature_list_for_model(model_path)

    # Build date range & features (chunked)
    dates = _daterange(args.date, int(args.days_before))
    print(f"▶ Simulating for date={args.date} covering {len(dates)} days from {dates[0]}")
    df = _build_features_chunked(
        curated_root=args.curated,
        sport=args.sport,
        dates=dates,
        preoff_minutes=args.preoff_mins,
        batch_markets=args.batch_markets,
        downsample_secs=(args.downsample_secs or None),
        chunk_days=args.chunk_days,
    )

    # Optional filter by country
    if args.country_filter and "countryCode" in df.columns:
        df = df.filter(pl.col("countryCode") == args.country_filter)

    # Ensure derived country feature exists to match training
    df = _ensure_country_feat(df)

    # Predict probabilities with feature names matching training order
    X = _to_numpy(df, feat_order)
    dX = xgb.DMatrix(X, feature_names=feat_order)
    p = bst.predict(dX)
    df = df.with_columns(pl.lit(p).alias("p_hat"))

    # Edge using observed price (ltp). Keep bets above min edge.
    df = df.with_columns(
        ((pl.col("p_hat") * pl.col("ltp")) - (1.0 - pl.col("p_hat"))).alias("edge")
    ).filter(pl.col("edge") >= args.min_edge)

    # Side (auto): back when p_hat > implied_prob, else skip
    if "implied_prob" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("p_hat") > pl.col("implied_prob"))
              .then(pl.lit("back"))
              .otherwise(pl.lit("skip"))
              .alias("side")
        ).filter(pl.col("side") == "back")
    else:
        df = df.with_columns(pl.lit("back").alias("side"))

    # Kelly stake before caps (stake_raw)
    df = df.with_columns(
        (args.bankroll_day * args.kelly * _kelly_fraction(pl.col("p_hat"), pl.col("ltp"))).alias("stake_raw")
    )

    # Keep top-N per market by edge (Polars 1.33-safe ranking)
    df = (
        df.sort(["marketId", "edge"], descending=[False, True])
          .with_columns(pl.arange(0, pl.count()).over("marketId").alias("_rn"))
          .filter(pl.col("_rn") < args.top_n_per_market)  # 0..top_n-1
          .drop("_rn")
    )

    # Start from stake_raw
    df = df.with_columns(pl.col("stake_raw").alias("stake")).drop("stake_raw")

    # Caps + proportional re-normalization
    df = _apply_caps_renorm(
        df, stake_col="stake",
        market_cap=args.stake_cap_market,
        day_cap=args.stake_cap_day,
    )

    # Enforce Betfair £1 minimum on non-zero stakes
    df = df.with_columns(
        pl.when(pl.col("stake") > 0.0)
          .then(pl.max_horizontal(pl.col("stake"), pl.lit(1.0)))
          .otherwise(0.0)
          .alias("stake")
    )

    # Realized PnL if label exists, else expected PnL
    win_payout = ((pl.col("ltp") - 1.0) * pl.col("stake") * (1.0 - args.commission)).alias("_win_payout")
    df = df.with_columns(win_payout)
    if args.label_col in df.columns:
        df = df.with_columns(
            pl.when(pl.col(args.label_col) == 1)
              .then(pl.col("_win_payout"))
              .otherwise(-pl.col("stake"))
              .alias("pnl")
        )
    else:
        df = df.with_columns(
            (pl.col("p_hat") * (pl.col("ltp") - 1.0) * pl.col("stake") * (1.0 - args.commission)
             - (1.0 - pl.col("p_hat")) * pl.col("stake")).alias("pnl")
        )
    df = df.drop("_win_payout")

    # Persist outputs
    bets_cols = [c for c in ("marketId","selectionId","countryCode","tto_minutes","ltp","p_hat","edge","side","stake","pnl") if c in df.columns]
    bets_out = str(Path(args.bets_out))
    df.select(bets_cols).write_csv(bets_out)

    if "countryCode" in df.columns:
        pnl_by_country = (
            df.group_by("countryCode")
              .agg([
                  pl.len().alias("n_bets"),
                  pl.sum("stake").alias("stake"),
                  pl.sum("pnl").alias("pnl"),
              ])
              .with_columns((pl.col("pnl") / pl.col("stake")).alias("roi"))
              .sort("roi", descending=True)
        )
    else:
        pnl_by_country = pl.DataFrame({"countryCode": [], "n_bets": [], "stake": [], "pnl": [], "roi": []})

    pnl_out = str(Path(args.pnl_by_country_out))
    pnl_by_country.write_csv(pnl_out)

    # Console summary
    n_bets = df.height
    stake_sum = float(df["stake"].sum()) if n_bets else 0.0
    pnl_sum = float(df["pnl"].sum()) if n_bets else 0.0
    roi = (pnl_sum / stake_sum) if stake_sum > 0 else 0.0
    print(f"Saved bets -> {bets_out}")
    print(f"Saved PnL by country -> {pnl_out}")
    print(f"Summary: n_bets={n_bets} stake={stake_sum:.2f} pnl={pnl_sum:.2f} ROI={roi:.3%}")


if __name__ == "__main__":
    main()
