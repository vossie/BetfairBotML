#!/usr/bin/env python3
import argparse, numpy as np
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta
from typing import List

@dataclass
class SplitPlan:
    train_dates: List[str]
    valid_dates: List[str]

def _fmt(d: _date) -> str:
    return d.strftime("%Y-%m-%d")

def _daterange_inclusive(start: _date, end: _date) -> List[str]:
    d, out = start, []
    while d <= end:
        out.append(_fmt(d))
        d += timedelta(days=1)
    return out

def build_split(asof: _date, train_days: int, valid_days: int) -> SplitPlan:
    val_start = asof - timedelta(days=valid_days - 1)
    val_end   = asof
    train_end = val_start - timedelta(days=1)
    train_start = train_end - timedelta(days=train_days - 1)
    return SplitPlan(
        train_dates=_daterange_inclusive(train_start, train_end),
        valid_dates=_daterange_inclusive(val_start, val_end),
    )

def _parse_date(s: str) -> _date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", type=str, required=True)
    ap.add_argument("--sport", type=str, required=True)
    ap.add_argument("--asof", type=str, required=True)
    ap.add_argument("--train-days", type=int, required=True)
    ap.add_argument("--valid-days", type=int, default=2,
                    help="Validation window length in days (ending at --asof)")
    args = ap.parse_args()

    plan = build_split(_parse_date(args.asof), args.train_days, args.valid_days)

    print("=== Temporal split ===")
    print(f"  Train: {plan.train_dates[0]} .. {plan.train_dates[-1]}  ({len(plan.train_dates)} days)")
    print(f"  Valid: {plan.valid_dates[0]} .. {plan.valid_dates[-1]}  ({len(plan.valid_dates)} days)\n")

    # TODO: plug in your existing trainer pipeline here

if __name__ == "__main__":
    main()
