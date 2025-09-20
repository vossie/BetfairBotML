#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, date as _date
from dataclasses import dataclass
from typing import List

# ----------------------------
# Helpers for date math
# ----------------------------
@dataclass
class SplitPlan:
    train_dates: List[str]
    valid_dates: List[str]

def _parse_date(s: str) -> _date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _fmt(d: _date) -> str:
    return d.strftime("%Y-%m-%d")

def _daterange_inclusive(start: _date, end: _date) -> List[str]:
    out, d = [], start
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

# ----------------------------
# Stub trainer (replace with your actual training code)
# ----------------------------
def train_temporal(
    curated_root,
    sport,
    asof_date,
    train_days,
    preoff_minutes,
    label_col,
    downsample_secs,
    device,
    n_estimators,
    learning_rate,
    early_stopping_rounds,
    commission,
    edge_thresh,
    calibrate,
    pm_horizon_secs,
    pm_tick_threshold,
    pm_slack_secs,
    pm_cutoff,
    market_prob,
    per_market_topk,
    side,
    ltp_min,
    ltp_max,
    sum_to_one,
    stake_mode,
    kelly_cap,
    kelly_floor,
    bankroll_nom,
    valid_days,
):
    print(">>> [DEBUG] Training run with arguments:")
    print(f"curated_root={curated_root}")
    print(f"sport={sport}")
    print(f"asof_date={asof_date}")
    print(f"train_days={train_days}, valid_days={valid_days}")
    print(f"preoff_minutes={preoff_minutes}, downsample_secs={downsample_secs}")
    print(f"commission={commission}, edge_thresh={edge_thresh}, calibrate={calibrate}")
    print(f"pm_horizon_secs={pm_horizon_secs}, pm_tick_threshold={pm_tick_threshold}, pm_cutoff={pm_cutoff}")
    print(f"stake_mode={stake_mode}, kelly_cap={kelly_cap}, kelly_floor={kelly_floor}, bankroll_nom={bankroll_nom}")
    print(f"side={side}, per_market_topk={per_market_topk}, ltp=[{ltp_min}, {ltp_max}]")
    print("... replace this stub with actual training logic ...")

# ----------------------------
# Main entrypoint
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Temporal trainer with flexible validation and staking options.")

    # --- Data / windowing ---
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--asof", required=True, help="Validation end date (YYYY-MM-DD)")
    ap.add_argument("--train-days", type=int, required=True)
    ap.add_argument("--valid-days", type=int, default=2, help="Validation window length (days)")
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--downsample-secs", type=int, default=5)

    # --- Device / model ---
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    # --- Market prob + backtest ---
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--edge-thresh", type=float, default=0.015)
    ap.add_argument("--edge-prob", choices=["raw", "cal"], default="cal")
    ap.add_argument("--no-sum-to-one", action="store_true")
    ap.add_argument("--market-prob", choices=["ltp", "overround"], default="overround")
    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--ltp-min", type=float, default=1.5)
    ap.add_argument("--ltp-max", type=float, default=5.0)
    ap.add_argument("--side", choices=["back", "lay", "both"], default="back")

    # --- Price-move head / gating ---
    ap.add_argument("--pm-horizon-secs", type=int, default=300)
    ap.add_argument("--pm-tick-threshold", type=int, default=1)
    ap.add_argument("--pm-slack-secs", type=int, default=3)
    ap.add_argument("--pm-cutoff", type=float, default=0.0)

    # --- Staking ---
    ap.add_argument("--stake", choices=["flat", "kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.0)
    ap.add_argument("--bankroll-nom", type=float, default=1000.0)

    args = ap.parse_args()

    # --- Split plan info ---
    asof = _parse_date(args.asof)
    plan = build_split(asof, args.train_days, args.valid_days)
    print("=== Temporal split ===")
    print(f"  Train: {plan.train_dates[0]} .. {plan.train_dates[-1]} ({len(plan.train_dates)} days)")
    print(f"  Valid: {plan.valid_dates[0]} .. {plan.valid_dates[-1]} ({len(plan.valid_dates)} days)\n")

    # --- Call trainer ---
    train_temporal(
        curated_root=args.curated,
        sport=args.sport,
        asof_date=args.asof,
        train_days=args.train_days,
        preoff_minutes=args.preoff_mins,
        label_col=args.label_col,
        downsample_secs=args.downsample_secs,
        device=args.device,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        commission=args.commission,
        edge_thresh=args.edge_thresh,
        calibrate=(args.edge_prob == "cal"),
        pm_horizon_secs=args.pm_horizon_secs,
        pm_tick_threshold=args.pm_tick_threshold,
        pm_slack_secs=args.pm_slack_secs,
        pm_cutoff=args.pm_cutoff,
        market_prob=args.market_prob,
        per_market_topk=args.per_market_topk,
        side=args.side,
        ltp_min=args.ltp_min,
        ltp_max=args.ltp_max,
        sum_to_one=(not args.no_sum_to_one),
        stake_mode=args.stake,
        kelly_cap=args.kelly_cap,
        kelly_floor=args.kelly_floor,
        bankroll_nom=args.bankroll_nom,
        valid_days=args.valid_days,
    )

if __name__ == "__main__":
    main()
