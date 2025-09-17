# ml/sim_country.py
from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

from . import features


def _load_booster(path: str) -> xgb.Booster:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Model not found: {p}")
    bst = xgb.Booster()
    bst.load_model(str(p))
    return bst


def _load_features(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Feature file not found: {p}")
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


def _build_features(
    curated_root: str,
    sport: str,
    dates: list[str],
    preoff_minutes: int,
    batch_markets: int,
    downsample_secs: int | None,
    chunk_days: int,
) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for i in range(0, len(dates), chunk_days):
        chunk = dates[i:i + chunk_days]
        print(f"  • building features for {chunk[0]}..{chunk[-1]}")
        df_c, _ = features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=chunk,
            preoff_minutes=preoff_minutes,
            batch_markets=batch_markets,
            downsample_secs=downsample_secs,
        )
        if not df_c.is_empty():
            parts.append(df_c)
    if not parts:
        raise SystemExit("No features built.")
    return pl.concat(parts, how="vertical", rechunk=True)


def _add_country_feature(df: pl.DataFrame) -> pl.DataFrame:
    if "countryCode" in df.columns:
        return df.with_columns(
            pl.col("countryCode").fill_null("UNK").cast(pl.Categorical).to_physical().alias("country_feat")
        )
    print("WARN: no countryCode column; continuing without country_feat")
    return df


def main():
    ap = argparse.ArgumentParser("Simulate bets with trained XGB (country-aware).")
    ap.add_argument("--model", required=True, help="Path to model JSON")
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
    ap.add_argument("--stake-cap-market", type=float, default=9999)
    ap.add_argument("--stake-cap-day", type=float, default=99999)
    ap.add_argument("--country-filter", type=str, default=None)
    ap.add_argument("--pnl-by-country-out", default="./output/pnl_by_country.csv")
    ap.add_argument("--bets-out", default="./output/bets_country.csv")
    args = ap.parse_args()

    # Load model + feature order
    bst = _load_booster(args.model)
    feats = _load_features(Path(args.model).with_suffix(".features.txt"))

    # Build feature set
    end = datetime.strptime(args.date, "%Y-%m-%d").date()
    dates = [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in reversed(range(args.days_before + 1))]

    df = _build_features(
        curated_root=args.curated,
        sport=args.sport,
        dates=dates,
        preoff_minutes=args.preoff_mins,
        batch_markets=args.batch_markets,
        downsample_secs=(args.downsample_secs or None),
        chunk_days=args.chunk_days,
    )

    # Add derived country_feat so features align
    df = _add_country_feature(df)

    # Only keep rows with labels if present
    if args.label_col in df.columns:
        df = df.filter(pl.col(args.label_col).is_not_null())

    if df.is_empty():
        raise SystemExit("No usable rows after feature build.")

    # Convert to numpy with correct feature order
    X = _to_numpy(df, feats)
    dX = xgb.DMatrix(X, feature_names=feats)

    # Predict
    p = bst.predict(dX)
    print(f"Simulated {len(p)} predictions, example={p[:5]}")


if __name__ == "__main__":
    main()
