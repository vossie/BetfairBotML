# ml/sim.py
from __future__ import annotations

import argparse
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


def _select_feature_cols(df: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {
        "marketId", "selectionId", "ts", "ts_ms",
        "publishTimeMs", label_col, "runnerStatus"
    }
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


def _predict_proba(df: pl.DataFrame, booster: xgb.Booster, feature_cols: List[str]) -> np.ndarray:
    X = df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X)
    p = booster.predict(dm)
    if p.ndim == 2:
        if p.shape[1] == 1:
            p = p.ravel()
        else:
            p = p[:, 1]
    return p


def _kelly_back_fraction(p: float, price: float, commission: float) -> float:
    b = max(price - 1.0, 0.0) * (1.0 - commission)
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    return max(f, 0.0)


def _kelly_lay_fraction(p: float, price: float, commission: float) -> float:
    # Conservative approximation for lay Kelly on Betfair:
    # profit if lose: +(1-comm)*stake ; loss if win: -(price-1)*stake
    b = (1.0 - commission)
    denom = (price - 1.0)
    if denom <= 0:
        return 0.0
    # liability-scaled approximation; then map back to stake fraction conservatively
    f = (b * (1.0 - p) - p * denom) / (denom + 1e-12)
    return max(f, 0.0)


def _ev_back(p: float, price: float, commission: float, stake: float) -> float:
    win_net = (price - 1.0) * (1.0 - commission)
    return stake * (p * win_net - (1.0 - p))


def _ev_lay(p: float, price: float, commission: float, stake: float) -> float:
    lose_profit = stake * (1.0 - commission)
    win_loss = (price - 1.0) * stake
    return (1.0 - p) * lose_profit - p * win_loss


def _realized_pnl_back(win_label: int, price: float, commission: float, stake: float) -> float:
    return stake * (price - 1.0) * (1.0 - commission) if win_label == 1 else -stake


def _realized_pnl_lay(win_label: int, price: float, commission: float, stake: float) -> float:
    return -(price - 1.0) * stake if win_label == 1 else stake * (1.0 - commission)


def _choose_side_auto(p: float, price: float, commission: float) -> str:
    ev_b = _ev_back(p, price, commission, 1.0)
    ev_l = _ev_lay(p, price, commission, 1.0)
    return "back" if ev_b >= ev_l else "lay"


def _ensure_time_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "ts" in df.columns:
        return df.with_columns(
            pl.col("ts").dt.date().alias("event_date")
        )
    if "publishTimeMs" in df.columns:
        return df.with_columns(
            pl.from_epoch((pl.col("publishTimeMs") / 1000).cast(pl.Int64)).alias("ts")
        ).with_columns(
            pl.col("ts").dt.date().alias("event_date")
        )
    return df


# --------------------- simulation core ---------------------

def _score_with_gate(
    df: pl.DataFrame,
    booster_30: Optional[xgb.Booster],
    booster_180: Optional[xgb.Booster],
    gate_minutes: float,
    label_col: str,
) -> pl.DataFrame:
    if booster_30 is not None and booster_180 is not None:
        early = df.filter(pl.col("tto_minutes") > gate_minutes)
        late = df.filter(pl.col("tto_minutes") <= gate_minutes)
        out_parts = []
        if not early.is_empty():
            fcols = _select_feature_cols(early, label_col)
            p = _predict_proba(early, booster_180, fcols)
            out_parts.append(early.with_columns(pl.lit(p).alias("p_hat")))
        if not late.is_empty():
            fcols = _select_feature_cols(late, label_col)
            p = _predict_proba(late, booster_30, fcols)
            out_parts.append(late.with_columns(pl.lit(p).alias("p_hat")))
        return pl.concat(out_parts, how="vertical") if out_parts else df.with_columns(pl.lit(None).alias("p_hat"))
    # single model path
    fcols = _select_feature_cols(df, label_col)
    bst = booster_30 or booster_180
    p = _predict_proba(df, bst, fcols)
    return df.with_columns(pl.lit(p).alias("p_hat"))


def _build_bets_table(
    df_scored: pl.DataFrame,
    min_edge: float,
    kelly_fraction: float,
    commission: float,
    side_mode: str,
) -> pl.DataFrame:
    # price sanity + implied prob if missing
    df_scored = df_scored.filter(pl.col("ltp").is_not_null() & (pl.col("ltp") > 1.01))
    if "implied_prob" not in df_scored.columns:
        df_scored = df_scored.with_columns(
            pl.when(pl.col("ltp") > 0).then(1.0 / pl.col("ltp")).otherwise(None).alias("implied_prob")
        )
    # edge
    df_scored = df_scored.with_columns((pl.col("p_hat") - pl.col("implied_prob")).alias("edge"))

    # per-row calc
    p_arr = df_scored["p_hat"].to_numpy()
    price_arr = df_scored["ltp"].to_numpy()

    sides, stake_units, ev_units, kelly_raw = [], [], [], []
    for p, price in zip(p_arr, price_arr):
        if side_mode == "auto":
            side = _choose_side_auto(float(p), float(price), commission)
        else:
            side = side_mode
        if side == "back":
            f = _kelly_back_fraction(float(p), float(price), commission)
            stake = max(kelly_fraction * f, 0.0)
            ev = _ev_back(float(p), float(price), commission, stake)
        else:
            f = _kelly_lay_fraction(float(p), float(price), commission)
            stake = max(kelly_fraction * f, 0.0)
            ev = _ev_lay(float(p), float(price), commission, stake)
        sides.append(side)
        stake_units.append(stake)
        ev_units.append(ev)
        kelly_raw.append(f)

    bets = df_scored.with_columns([
        pl.Series("side", sides),
        pl.Series("stake_unit", stake_units),
        pl.Series("ev_unit", ev_units),
        pl.Series("kelly_raw", kelly_raw),
    ])

    # Filter for positive EV and minimum edge
    bets = bets.filter((pl.col("ev_unit") > 0.0) & (pl.col("edge") >= min_edge))
    return bets


def _cap_stakes(
    bets: pl.DataFrame,
    stake_cap_market: float,
    stake_cap_day: float,
) -> pl.DataFrame:
    if bets.is_empty():
        return bets

    bets = _ensure_time_columns(bets)

    # Cap per market
    if stake_cap_market is not None:
        bets = (
            bets
            .with_columns(pl.col("stake_unit").cumsum().over("marketId").alias("cum_stake_mkt"))
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
    if stake_cap_day is not None:
        bets = (
            bets
            .with_columns(pl.col("stake_unit").cumsum().over("event_date").alias("cum_stake_day"))
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

    # Drop zeroed stakes
    bets = bets.filter(pl.col("stake_unit") > 0)
    return bets


def _pick_topn_per_market(bets: pl.DataFrame, top_n: int) -> pl.DataFrame:
    if top_n is None or top_n <= 0:
        return bets
    return (
        bets
        .with_columns(pl.col("ev_unit").rank(method="ordinal", descending=True).over("marketId").alias("ev_rank_mkt"))
        .filter(pl.col("ev_rank_mkt") <= top_n)
        .drop("ev_rank_mkt")
    )


def _aggregate_per_market(bets: pl.DataFrame) -> pl.DataFrame:
    # One row per market with summed stake and pnl, plus best selection (by stake or EV)
    if bets.is_empty():
        return bets
    agg = (
        bets.group_by("marketId")
        .agg([
            pl.first("sport").alias("sport"),
            pl.max("tto_minutes").alias("tto_max"),
            pl.count().alias("n_bets"),
            pl.sum("stake_unit").alias("stake_unit_sum"),
            pl.sum("pnl_unit").alias("pnl_unit_sum"),
            pl.mean("edge").alias("edge_mean"),
            pl.max("ev_unit").alias("ev_unit_max"),
            pl.first("event_date").alias("event_date"),
        ])
        .sort(["event_date", "marketId"])
    )
    return agg


def _pnl_columns(bets: pl.DataFrame, commission: float) -> pl.DataFrame:
    if bets.is_empty():
        return bets
    wins = bets["winLabel"].to_numpy().astype(int)
    prices = bets["ltp"].to_numpy().astype(float)
    sides = bets["side"].to_list()
    stakes = bets["stake_unit"].to_numpy().astype(float)

    pnls = []
    for w, pr, sd, st in zip(wins, prices, sides, stakes):
        pnls.append(
            _realized_pnl_back(w, pr, commission, st) if sd == "back" else _realized_pnl_lay(w, pr, commission, st)
        )
    return bets.with_columns(pl.Series("pnl_unit", pnls))


def _binwise_pnl(bets: pl.DataFrame, bins: List[int]) -> pl.DataFrame:
    if bets.is_empty() or "tto_minutes" not in bets.columns:
        return pl.DataFrame([])
    # assign bin labels
    edges = bins
    labels = [f"{edges[i]:>3}-{edges[i+1]:>3}" for i in range(len(edges)-1)]
    dfb = bets.with_columns(
        pl.cut(pl.col("tto_minutes"), bins=edges, labels=labels, include_breaks=False).alias("tto_bin")
    )
    out = (
        dfb.group_by("tto_bin")
        .agg([
            pl.count().alias("n"),
            pl.sum("stake_unit").alias("stake_sum"),
            pl.sum("pnl_unit").alias("pnl_sum"),
        ])
        .with_columns(
            (pl.col("pnl_sum") / pl.col("stake_sum").replace(0, None)).alias("roi")
        )
        .sort("tto_bin")
    )
    return out


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Bet selection & PnL simulation using trained XGBoost models.")

    # models (single or dual)
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
    ap.add_argument("--chunk-days", type=int, default=2)

    # selection & staking
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02, help="Require model_p - implied_prob >= min_edge")
    ap.add_argument("--kelly", type=float, default=0.25, help="Fraction of Kelly fraction (0..1)")
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1, help="Select top N by EV in each market (prevents double-bets unless raised)")
    ap.add_argument("--stake-cap-market", type=float, default=1.0, help="Max total stake units per market (after EV filtering).")
    ap.add_argument("--stake-cap-day", type=float, default=10.0, help="Max total stake units per day (after EV filtering).")

    # outputs
    ap.add_argument("--bets-out", default="bets.csv")
    ap.add_argument("--agg-out", default="bets_by_market.csv")
    ap.add_argument("--bin-out", default="pnl_by_tto_bin.csv")

    args = ap.parse_args()

    # load models
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

    # Build features in chunks
    df_parts: List[pl.DataFrame] = []
    total_raw = 0
    for dchunk in [dates[i:i + args.chunk_days] for i in range(0, len(dates), args.chunk_days)]:
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
    df_feat = _ensure_time_columns(df_feat)

    # Score with single or dual model
    df_scored = _score_with_gate(
        df=df_feat,
        booster_30=booster_30,
        booster_180=booster_180,
        gate_minutes=args.gate_mins,
        label_col=args.label_col,
    )

    # Build candidate bets
    bets = _build_bets_table(
        df_scored=df_scored,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        commission=args.commission,
        side_mode=args.side,
    )

    # Respect top-N per market (avoid double-betting unless you choose to)
    bets = _pick_topn_per_market(bets, args.top_n_per_market)

    # Apply stake caps (market/day) then drop zeroed stakes
    bets = _cap_stakes(
        bets=bets,
        stake_cap_market=args.stake_cap_market,
        stake_cap_day=args.stake_cap_day,
    )

    # Realized PnL with commission
    bets = _pnl_columns(bets, args.commission)

    # Save per-bet CSV
    if not bets.is_empty():
        bets.write_csv(args.bets_out)
        print(f"Saved bets to {args.bets_out}")
    else:
        print("No bets selected after filters and caps.")
        # still produce empty CSVs for downstream stability
        pl.DataFrame([]).write_csv(args.bets_out)
        pl.DataFrame([]).write_csv(args.agg_out)
        pl.DataFrame([]).write_csv(args.bin_out)
        return

    # Per-market aggregation (so you can confirm no accidental double-bets)
    agg = _aggregate_per_market(bets)
    if not agg.is_empty():
        agg.write_csv(args.agg_out)
        print(f"Saved per-market aggregation to {args.agg_out}")

    # Per-bin (horizon) PnL breakdown
    bins = [0, 30, 60, 90, 120, 180]
    bin_pnl = _binwise_pnl(bets, bins)
    if not bin_pnl.is_empty():
        bin_pnl.write_csv(args.bin_out)
        print(f"Saved per-bin PnL breakdown to {args.bin_out}")

    # Print a simple summary
    total_stake = float(bets["stake_unit"].sum())
    total_pnl = float(bets["pnl_unit"].sum())
    roi = (total_pnl / total_stake) if total_stake > 0 else 0.0
    n_bets = bets.height
    hit_rate = float((bets["winLabel"].to_numpy() == 1).mean()) if n_bets > 0 else 0.0
    print("\n=== Simulation Summary ===")
    print(f"n_bets                : {n_bets}")
    print(f"total_stake_units     : {total_stake:.4f}")
    print(f"total_pnl_units       : {total_pnl:.4f}")
    print(f"roi                   : {roi:.4f}")
    print(f"hit_rate              : {hit_rate:.4f}")
    print(f"commission            : {args.commission}")
    print(f"kelly_fraction        : {args.kelly}")
    print(f"min_edge              : {args.min_edge}")
    print(f"gate_minutes          : {args.gate_mins if (booster_30 and booster_180) else 'n/a (single model)'}")


if __name__ == "__main__":
    main()
