# ml/run_train.py
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, UTC
from typing import List
import logging

import polars as pl

from . import dataio, features


def _daterange(end_date_str: str, days: int) -> List[str]:
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    d = start
    out = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True, help="s3://bucket[/prefix] or /local/path")
    ap.add_argument("--sport", required=True, help="e.g. horse_racing")
    ap.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    ap.add_argument("--days", type=int, default=1, help="how many days back inclusive")

    # streaming builder knobs
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)

    # Kelly / bankroll knobs
    ap.add_argument("--kelly", type=float, default=0.125)
    ap.add_argument("--cap-ratio", type=float, default=0.02)
    ap.add_argument("--market-cap", type=float, default=50.0)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    args = ap.parse_args()

    # show available dates
    available = {
        "orderbook": dataio.list_dates(args.curated, args.sport, "orderbook_snapshots_5s"),
        "defs": dataio.list_dates(args.curated, args.sport, "market_definitions"),
        "results": dataio.list_dates(args.curated, args.sport, "results"),
    }
    print("Available dates:", available)

    dates = _daterange(args.date, args.days)
    print(f"Loading sport={args.sport}, dates={dates[0]}..{dates[-1]}")

    # memory-safe batch feature builder
    df_feat, total_raw = features.build_features_streaming(
        args.curated,
        args.sport,
        dates,
        preoff_minutes=args.preoff_mins,
        batch_markets=args.batch_markets,
        downsample_secs=(args.downsample_secs or None),
    )
    if df_feat.is_empty():
        logging.error(
            "No features generated. Verify data availability for sport=%s dates %s..%s",
            args.sport,
            dates[0],
            dates[-1],
        )
        return

    print(f"Feature rows: {df_feat.height:,} (from ~{total_raw:,} raw snapshot rows scanned)")
    print("Training model...")
    artifacts = features.train_model(df_feat)
    print("Metrics:", artifacts["metrics"])

    print("Generating recommendations (fractional Kelly)...")
    recs = features.recommend(
        df_feat,
        artifacts,
        bankroll=args.bankroll,
        kelly=args.kelly,
        cap_ratio=args.cap_ratio,
        market_cap=args.market_cap,
    )
    print(recs.head(25))


if __name__ == "__main__":
    main()
