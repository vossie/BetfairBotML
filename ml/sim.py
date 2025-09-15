# ml/sim.py
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import numpy as np
import polars as pl
import xgboost as xgb

from . import features


# --------------------- helpers ---------------------

def _daterange(end_date_str: str, days: int) -> List[str]:
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    out: List[str] = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _load_booster(path: str) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(path)
    return bst


def _predict_proba(
    df: pl.DataFrame,
    booster: xgb.Booster,
    feature_cols: List[str],
) -> np.ndarray:
    X = df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X)
    p = booster.predict(dm)
    if p.ndim == 2:  # multi/prob
        if p.shape[1] == 1:
            p = p.ravel()
        else:
            p = p[:, 1]  # positive class
    return p


def _select_feature_cols(df: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {"marketId", "selectionId", "ts", "publishTimeMs", label_col, "runnerStatus"}
    cols: List[str] = []
    schema = df.collect_schema()
    for c, dt in zip(schema.names(), schema.dtypes()):
        if c in exclude:
            continue
        if "label" in c.lower() or "target" in c.lower():
            continue
        if dt.is_numeric():
            cols.append(c)
    if not cols:
        raise RuntimeError("No numeric features found.")
    return cols


def _kelly_back_fraction(p: float, price: float, commission: float) -> float:
    # b = net odds (after commission)
    b = max(price - 1.0, 0.0) * (1.0 - commission)
    q = 1.0 - p
    denom = b
    if denom <= 0.0:
        return 0.0
    f = (b * p - q) / denom
    return max(f, 0.0)


def _kelly_lay_fraction(p: float, price: float, commission: float) -> float:
    # Lay profit if horse loses is stake*(1-commission), loss if wins is liability=(price-1)*stake
    # Kelly in terms of "liability budget" can be derived similarly; here a conservative proxy:
    # b_lay = 1.0 - commission  (unit profit per unit stake when lose)
    b = (1.0 - commission)
    q = p
    denom = (price - 1.0)  # risk per unit stake
    if denom <= 0.0:
        return 0.0
    # Fraction of liability; we return fraction of stake instead (approx) with same bound
    f = (b * (1.0 - p) - q * denom) / (denom * (1.0 - commission) + 1e-12)
    return max(f, 0.0)


def _ev_back(p: float, price: float, commission: float, stake: float) -> float:
    win_net = (price - 1.0) * (1.0 - commission)
    return stake * (p * win_net - (1.0 - p) * 1.0)


def _ev_lay(p: float, price: float, commission: float, stake: float) -> float:
    # Profit if loses: +stake*(1-comm). Loss if wins: -liability=(price-1)*stake
    lose_profit = stake * (1.0 - commission)
    win_loss = (price - 1.0) * stake
    return (1.0 - p) * lose_profit - p * win_loss


def _realized_pnl_back(win_label: int, price: float, commission: float, stake: float) -> float:
    if win_label == 1:
        return stake * (price - 1.0) * (1.0 - commission)
    else:
        return -stake


def _realized_pnl_lay(win_label: int, price: float, commission: float, stake: float) -> float:
    if win_label == 1:
        return -(price - 1.0) * stake  # pay liability
    else:
        return stake * (1.0 - commission)  # keep backer's stake minus comm


def _choose_side_auto(p: float, price: float, commission: float) -> str:
    # Compare EV of back vs lay with unit stake
    ev_b = _ev_back(p, price, commission, 1.0)
    ev_l = _ev_lay(p, price, commission, 1.0)
    return "back" if ev_b >= ev_l else "lay"


# --------------------- simulation ---------------------

def simulate(
    df: pl.DataFrame,
    booster_30: Optional[xgb.Booster],
    booster_180: Optional[xgb.Booster],
    gate_minutes: float,
    label_col: str,
    min_edge: float,
    kelly_fraction: float,
    commission: float,
    side_mode: str,
    top_n_per_market: int,
) -> Tuple[pl.DataFrame, Dict]:
    # Clean labels to compute realized PnL
    df = df.filter(pl.col(label_col).is_not_null()).with_columns(pl.col(label_col).cast(pl.Int32))

    # Split for dual horizon
    if booster_30 is not None and booster_180 is not None:
        df_early = df.filter(pl.col("tto_minutes") > gate_minutes)
        df_late = df.filter(pl.col("tto_minutes") <= gate_minutes)
        parts = []
        for part, bst in ((df_early, booster_180), (df_late, booster_30)):
            if part.is_empty():
                continue
            fcols = _select_feature_cols(part, label_col)
            p = _predict_proba(part, bst, fcols)
            parts.append(part.with_columns(pl.lit(p).alias("p_hat")))
        df_scored = pl.concat(parts, how="vertical")
    else:
        fcols = _select_feature_cols(df, label_col)
        bst = booster_30 or booster_180
        p = _predict_proba(df, bst, fcols)
        df_scored = df.with_columns(pl.lit(p).alias("p_hat"))

    # We need ltp (price) and implied_prob (1/ltp) for edge comparisons.
    if "ltp" not in df_scored.columns:
        raise SystemExit("features must include 'ltp'.")
    if "implied_prob" not in df_scored.columns:
        # fall back to compute
        df_scored = df_scored.with_columns(
            pl.when(pl.col("ltp") > 0).then(1.0 / pl.col("ltp")).otherwise(None).alias("implied_prob")
        )

    # Compute EV and stake per side and then pick selections
    def _row_calc(p: float, price: float) -> Tuple[str, float, float, float]:
        # pick side
        if side_mode == "auto":
            chosen = _choose_side_auto(p, price, commission)
        else:
            chosen = side_mode  # 'back' or 'lay'
        # kelly sizing
        if chosen == "back":
            f = _kelly_back_fraction(p, price, commission)
            stake = max(kelly_fraction * f, 0.0)
            ev = _ev_back(p, price, commission, stake)
        else:
            f = _kelly_lay_fraction(p, price, commission)
            stake = max(kelly_fraction * f, 0.0)
            ev = _ev_lay(p, price, commission, stake)
        return chosen, stake, ev, f

    # Edge definition: difference between model prob and market implied prob (or EV threshold)
    # We'll keep both and use EV>0 plus edge>=min_edge as gates.
    df_scored = df_scored.with_columns(
        (pl.col("p_hat") - pl.col("implied_prob")).alias("edge")
    )

    # Per-row calc (vectorized via apply on small subset)
    arr = df_scored.select(["p_hat", "ltp"]).to_numpy()
    sides = []
    stakes = []
    evs = []
    kelly_fs = []
    for p, price in arr:
        s, st, ev, kf = _row_calc(float(p), float(price))
        sides.append(s)
        stakes.append(st)
        evs.append(ev)
        kelly_fs.append(kf)

    df_scored = df_scored.with_columns([
        pl.Series("side", sides),
        pl.Series("stake_unit", stakes),   # stake as fraction of bankroll (unit=1.0)
        pl.Series("ev_unit", evs),
        pl.Series("kelly_raw", kelly_fs),
    ])

    # Filter for positive EV and minimum edge
    df_scored = df_scored.filter((pl.col("ev_unit") > 0.0) & (pl.col("edge") >= min_edge))

    if df_scored.is_empty():
        return df_scored, {
            "n_bets": 0, "roi": 0.0, "bankroll_ret": 0.0,
            "avg_price": None, "hit_rate": None
        }

    # Pick top-N per market by EV
    df_scored = (
        df_scored
        .with_columns(pl.col("ev_unit").rank(method="ordinal", descending=True).over("marketId").alias("ev_rank_mkt"))
        .filter(pl.col("ev_rank_mkt") <= top_n_per_market)
        .drop("ev_rank_mkt")
    )

    # Realized PnL (unit bankroll)
    def _row_pnl(win: int, price: float, side: str, stake: float) -> float:
        if stake <= 0:
            return 0.0
        if side == "back":
            return _realized_pnl_back(win, price, commission, stake)
        else:
            return _realized_pnl_lay(win, price, commission, stake)

    pnl_list = []
    wins = df_scored.select("winLabel").to_numpy().ravel().astype(int)
    prices = df_scored.select("ltp").to_numpy().ravel().astype(float)
    stakes_vec = df_scored.select("stake_unit").to_numpy().ravel().astype(float)
    sides_vec = df_scored.select("side").to_series().to_list()

    for w, pr, sd, st in zip(wins, prices, sides_vec, stakes_vec):
        pnl_list.append(_row_pnl(w, pr, sd, st))

    df_scored = df_scored.with_columns(pl.Series("pnl_unit", pnl_list))

    # Summary
    n_bets = df_scored.height
    bankroll_ret = float(np.sum(pnl_list))       # return per unit bankroll
    stake_sum = float(np.sum(stakes_vec))
    roi = bankroll_ret / stake_sum if stake_sum > 0 else 0.0
    hit_rate = float(np.mean(wins == 1)) if n_bets > 0 else None
    avg_price = float(np.mean(prices)) if n_bets > 0 else None

    summary = {
        "n_bets": n_bets,
        "bankroll_return_units": bankroll_ret,
        "total_stake_units": stake_sum,
        "roi": roi,
        "avg_price": avg_price,
        "hit_rate": hit_rate,
        "commission": commission,
        "kelly_fraction": kelly_fraction,
        "min_edge": min_edge,
        "side_mode": side_mode,
        "gate_minutes": gate_minutes if (booster_30 and booster_180) else None,
    }
    return df_scored, summary


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Bet selection & PnL simulation using trained XGBoost models.")
    # models
    ap.add_argument("--model", help="Path to single model (JSON).")
    ap.add_argument("--model-30", help="Path to short-horizon model (<= gate-mins).")
    ap.add_argument("--model-180", help="Path to long-horizon model (> gate-mins).")
    ap.add_argument("--gate-mins", type=float, default=45.0, help="Gate for dual model switching (tto<=gate→short).")

    # data
    ap.add_argument("--curated", required=True, help="s3://bucket or /local/path")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)

    # selection
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02, help="Require model_p - implied_prob >= min_edge")
    ap.add_argument("--kelly", type=float, default=0.25, help="Fractional Kelly (0..1) applied to raw Kelly fraction")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)

    # outputs
    ap.add_argument("--bets-out", default="bets.csv")

    args = ap.parse_args()

    if args.model:
        if args.model_30 or args.model_180:
            raise SystemExit("Provide either --model OR (--model-30 and --model-180), not both.")
        booster_30 = _load_booster(args.model)
        booster_180 = None
    else:
        if not (args.model_30 and args.model_180):
            raise SystemExit("Dual mode requires --model-30 and --model-180.")
        booster_30 = _load_booster(args.model_30)
        booster_180 = _load_booster(args.model_180)

    dates = _daterange(args.date, args.days)

    # Build features once
    df_parts: List[pl.DataFrame] = []
    total_raw = 0
    for dchunk in [dates[i : i + 2] for i in range(0, len(dates), 2)]:  # fixed 2-day chunks for memory balance
        df_c, raw_c = features.build_features_streaming(
            curated_root=args.curated,
            sport=args.sport,
            dates=dchunk,
            preoff_minutes=args.preoff_mins,
            batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
        )
        total_raw += raw_c
        if not df_c.is_empty():
            df_parts.append(df_c)

    if not df_parts:
        raise SystemExit("No features produced for given date range.")
    df_feat = pl.concat(df_parts, how="vertical", rechunk=True)

    # Sort by time for sanity
    if "ts" in df_feat.columns:
        df_feat = df_feat.sort("ts")
    elif "publishTimeMs" in df_feat.columns:
        df_feat = df_feat.sort("publishTimeMs")

    # Simulate
    bets_df, summary = simulate(
        df=df_feat,
        booster_30=booster_30,
        booster_180=booster_180,
        gate_minutes=args.gate_mins,
        label_col=args.label_col,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        commission=args.commission,
        side_mode=args.side,
        top_n_per_market=args.top_n_per_market,
    )

    # Save bets
    if not bets_df.is_empty():
        bets_df.write_csv(args.bets_out)
        print(f"Saved bets to {args.bets_out}")

    # Print summary
    print("\n=== Simulation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
