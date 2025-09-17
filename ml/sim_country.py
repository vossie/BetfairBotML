# ml/sim_country.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import xgboost as xgb

from . import features
from .sim import (
    _edge_columns, _kelly_stake_units, _build_bets_table,
    _pick_topn_per_market, _cap_stakes, _pnl_columns
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _outpath(p: str) -> str:
    pp = Path(p)
    return str(pp if pp.is_absolute() else OUTPUT_DIR / pp.name)


def _daterange(end_date_str: str, days: int) -> List[str]:
    from datetime import datetime, timedelta
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _select_feature_cols(df: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {
        "sport","marketId","selectionId","ts","ts_ms","publishTimeMs",
        label_col,"runnerStatus","p_hat","countryCode"
    }
    cols: List[str] = []
    for name, dtype in df.schema.items():
        if name in exclude:
            continue
        lname = name.lower()
        if "label" in lname or "target" in lname:
            continue
        if dtype.is_numeric():
            cols.append(name)
    if not cols:
        raise RuntimeError("No numeric feature columns found.")
    return cols


def _to_numpy(df: pl.DataFrame, cols: List[str]):
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


def _load_booster(path: str) -> xgb.Booster:
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        alt = OUTPUT_DIR / p.name
        if alt.exists():
            p = alt
    bst = xgb.Booster()
    bst.load_model(str(p))
    return bst


def main():
    ap = argparse.ArgumentParser(description="Simulate wagering with country-aware model and report by country.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--days-before", type=int, default=0)

    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--side", choices=["back","lay","auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=10.0)
    ap.add_argument("--stake-cap-day", type=float, default=100.0)

    ap.add_argument("--country-filter", default=None)
    ap.add_argument("--pnl-by-country-out", default="pnl_by_country.csv")
    ap.add_argument("--bets-out", default="bets_country.csv")

    args = ap.parse_args()

    bst = _load_booster(args.model)

    # Dates
    days_total = int(args.days_before) + 1
    dates = _daterange(args.date, days_total)

    # Build features
    parts = []
    for i in range(0, len(dates), args.chunk_days):
        dchunk = dates[i:i+args.chunk_days]
        df_c, _ = features.build_features_streaming(
            curated_root=args.curated, sport=args.sport, dates=dchunk,
            preoff_minutes=args.preoff_mins, batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
        )
        if not df_c.is_empty():
            parts.append(df_c)
    if not parts:
        raise SystemExit("No features produced.")
    df = pl.concat(parts, how="vertical", rechunk=True)

    if args.country_filter:
        df = df.filter(pl.col("countryCode") == args.country_filter)

    # Score with booster
    fcols = _select_feature_cols(df, args.label_col)
    X = _to_numpy(df, fcols)
    p = bst.predict(xgb.DMatrix(X))
    df_scored = df.with_columns(pl.lit(p).alias("p_hat"))

    # Build bets + pnl
    bets = _build_bets_table(
        df_scored,
        label_col=args.label_col,
        min_edge=args.min_edge,
        kelly_frac=args.kelly,
        side_mode=args.side,
    )
    bets = _pick_topn_per_market(bets, args.top_n_per_market)
    bets = _cap_stakes(bets, cap_market=args.stake_cap_market, cap_day=args.stake_cap_day)
    bets = _pnl_columns(bets, commission=args.commission)

    bets_out = _outpath(args.bets_out)
    bets.write_csv(bets_out)

    # PnL by countryCode
    if "countryCode" in bets.columns:
        pnl_by_country = (
            bets.group_by("countryCode")
                .agg([
                    pl.len().alias("n_bets"),
                    pl.sum("stake").alias("stake"),
                    pl.sum("pnl").alias("pnl")
                ])
                .with_columns((pl.col("pnl") / pl.col("stake")).alias("roi"))
                .sort("roi", descending=True)
        )
    else:
        pnl_by_country = pl.DataFrame({"countryCode": [], "n_bets": [], "stake": [], "pnl": [], "roi": []})

    pnl_out = _outpath(args.pnl_by_country_out)
    pnl_by_country.write_csv(pnl_out)

    total_bets = bets.height
    stake_sum = float(bets["stake"].sum()) if total_bets else 0.0
    pnl_sum = float(bets["pnl"].sum()) if total_bets else 0.0
    roi = (pnl_sum / stake_sum) if stake_sum > 0 else 0.0

    print(f"Saved bets to {bets_out}")
    print(f"Saved PnL by country to {pnl_out}")
    print(f"Summary: n_bets={total_bets} stake={stake_sum:.2f} pnl={pnl_sum:.2f} ROI={roi:.3%}")


if __name__ == "__main__":
    main()
