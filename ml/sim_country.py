# ml/sim_country.py
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import xgboost as xgb

from . import features
from .sim import (
    _build_bets_table,
    _pick_topn_per_market,
    _cap_stakes,
    _pnl_columns,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_booster(path: str) -> xgb.Booster:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Model not found: {p}")
    bst = xgb.Booster()
    bst.load_model(str(p))
    return bst


def _load_feature_list_for_model(model_path: Path) -> List[str]:
    feats_txt = model_path.with_suffix(".features.txt")
    if not feats_txt.exists():
        raise SystemExit(f"Feature list not found: {feats_txt}")
    return [ln.strip() for ln in feats_txt.read_text().splitlines() if ln.strip()]


def _dates_back_from(end_date: str, days_before: int) -> List[str]:
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    return [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in reversed(range(days_before + 1))]


def _build_features_window(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int,
    batch_markets: int,
    downsample_secs: int | None,
    chunk_days: int,
) -> pl.DataFrame:
    parts: List[pl.DataFrame] = []
    for i in range(0, len(dates), chunk_days):
        chunk = dates[i : i + chunk_days]
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


def _ensure_country_feat(df: pl.DataFrame) -> pl.DataFrame:
    # Mirror trainer: derive numeric category code for country
    for cand in ("countryCode", "eventCountryCode", "country"):
        if cand in df.columns:
            try:
                return df.with_columns(
                    pl.col(cand).fill_null("UNK").cast(pl.Categorical).to_physical().alias("country_feat")
                )
            except Exception:
                # Older Polars fallback
                return df.with_columns(
                    pl.col(cand).fill_null("UNK").cast(pl.Utf8).hash().alias("country_feat")
                )
    print("WARN: no country column found; continuing without 'country_feat'.")
    return df


def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    # Keep order as in features.txt; fill NaNs conservatively
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser("Country-aware simulator (aligned with sim.py logic).")
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
    ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=10.0)
    ap.add_argument("--stake-cap-day", type=float, default=100.0)

    ap.add_argument("--country-filter", default=None)
    ap.add_argument("--pnl-by-country-out", default="./output/pnl_by_country.csv")
    ap.add_argument("--bets-out", default="./output/bets_country.csv")
    args = ap.parse_args()

    model_path = Path(args.model)
    bst = _load_booster(str(model_path))
    feat_order = _load_feature_list_for_model(model_path)

    # Build feature window
    dates = _dates_back_from(args.date, int(args.days_before))
    df = _build_features_window(
        curated_root=args.curated,
        sport=args.sport,
        dates=dates,
        preoff_minutes=args.preoff_mins,
        batch_markets=args.batch_markets,
        downsample_secs=(args.downsample_secs or None),
        chunk_days=args.chunk_days,
    )

    # Optional filter by country code if present/requested
    if args.country_filter and "countryCode" in df.columns:
        df = df.filter(pl.col("countryCode") == args.country_filter)

    # Ensure derived country feature exists to match training
    df = _ensure_country_feat(df)

    # Keep only rows with label when available (harmless in sim)
    if args.label_col in df.columns:
        df = df.filter(pl.col(args.label_col).is_not_null())

    if df.is_empty():
        raise SystemExit("No usable rows after feature build.")

    # Predict with feature names (must match training)
    X = _to_numpy(df, feat_order)
    dX = xgb.DMatrix(X, feature_names=feat_order)
    p = bst.predict(dX)
    df_scored = df.with_columns(pl.lit(p).alias("p_hat"))

    # Build bets using the same utilities as sim.py
    # Assumes df contains 'implied_prob' (from features builder); if not, consider adding it there.
    bets = _build_bets_table(
        df_scored,
        label_col=args.label_col,
        min_edge=args.min_edge,
        kelly_frac=args.kelly,
        side_mode=args.side,
    )

    # Pick top-N per market (default 1) to avoid fanning out stakes
    bets = _pick_topn_per_market(bets, args.top_n_per_market)

    # Cap at per-market/day budgets and re-normalize (sim.py semantics)
    bets = _cap_stakes(bets, cap_market=args.stake_cap_market, cap_day=args.stake_cap_day)

    # Enforce Betfair £1 minimum stake after caps (final guard)
    if "stake" in bets.columns:
        bets = bets.with_columns(
            pl.when(pl.col("stake") < 1.0).then(1.0).otherwise(pl.col("stake")).alias("stake")
        )

    # Compute PnL with commission, consistent with sim.py (_pnl_columns applies commission to winners only)
    bets = _pnl_columns(bets, commission=args.commission)

    # Write outputs
    bets_out = str(Path(args.bets_out))
    bets.write_csv(bets_out)

    if "countryCode" in bets.columns:
        pnl_by_country = (
            bets.group_by("countryCode")
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
    total_bets = bets.height
    stake_sum = float(bets["stake"].sum()) if total_bets else 0.0
    pnl_sum = float(bets["pnl"].sum()) if total_bets else 0.0
    roi = (pnl_sum / stake_sum) if stake_sum > 0 else 0.0
    print(f"Saved bets -> {bets_out}")
    print(f"Saved PnL by country -> {pnl_out}")
    print(f"Summary: n_bets={total_bets} stake={stake_sum:.2f} pnl={pnl_sum:.2f} ROI={roi:.3%}")


if __name__ == "__main__":
    main()
