# ml/sim.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import numpy as np
import polars as pl
import xgboost as xgb

from . import features

# ----------------------------- constants & IO helpers -----------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _outpath(p: str) -> str:
    """Ensure relative outputs land in ./output/; pass absolute paths through."""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str(OUTPUT_DIR / pp.name)


# ----------------------------- date & features utils -----------------------------

def _daterange(end_date_str: str, days: int) -> List[str]:
    """
    Build a list of YYYY-MM-DD strings covering [end - (days-1) ... end], inclusive.
    """
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    out: List[str] = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _select_feature_cols(df: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {
        "marketId", "selectionId", "ts", "ts_ms",
        "publishTimeMs", label_col, "runnerStatus", "p_hat",
    }
    cols: List[str] = []
    schema = df.collect_schema()
    for name, dtype in zip(schema.names(), schema.dtypes()):
        if name in exclude:
            continue
        if "label" in name.lower() or "target" in name.lower():
            continue
        if dtype.is_numeric():
            cols.append(name)
    if not cols:
        raise RuntimeError("No numeric feature columns found.")
    return cols


# ----------------------------- model loading & prediction -----------------------------

def _load_booster(path: str) -> xgb.Booster:
    """Load xgboost model; if relative and not found, also try ./output/<name>."""
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        alt = OUTPUT_DIR / p.name
        if alt.exists():
            p = alt
    bst = xgb.Booster()
    bst.load_model(str(p))
    return bst


def _predict_proba(df: pl.DataFrame, booster: xgb.Booster, feature_cols: List[str]) -> np.ndarray:
    X = df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)
    dm = xgb.DMatrix(X)
    p = booster.predict(dm)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] > 1 else p.ravel()
    return p


# ----------------------------- betting helpers -----------------------------

def _edge_columns(df: pl.DataFrame, side: str = "auto") -> pl.DataFrame:
    """
    Compute fair odds from p_hat; implied from ltp; edge for back/lay; pick side if auto.
    Assumptions:
      - 'ltp' is the available back price proxy (odds). If missing/<=1, we drop the row.
      - edge_back   = p_hat - implied_prob
      - edge_lay    = (1 - p_hat) - (1 - implied_prob) = implied_prob - p_hat  (mirror)
    """
    have_cols = set(df.columns)
    if "p_hat" not in have_cols:
        raise RuntimeError("p_hat missing. Score the dataframe first.")
    if "ltp" not in have_cols:
        raise RuntimeError("ltp missing. Needed for implied_prob/odds.")

    out = df.with_columns([
        # guard against bad odds
        pl.when(pl.col("ltp") > 1.0).then(pl.col("ltp")).otherwise(None).alias("odds"),
    ]).drop_nulls(["odds"])

    out = out.with_columns([
        pl.when(pl.col("p_hat") > 0).then(1.0 / pl.col("p_hat")).otherwise(None).alias("fair_odds"),
        pl.when(pl.col("odds") > 0).then(1.0 / pl.col("odds")).otherwise(None).alias("implied_prob"),
    ])

    out = out.with_columns([
        (pl.col("p_hat") - pl.col("implied_prob")).alias("edge_back"),
        (pl.col("implied_prob") - pl.col("p_hat")).alias("edge_lay"),
    ])

    if side == "back":
        out = out.with_columns([
            pl.lit("back").alias("side"),
            pl.col("edge_back").alias("edge"),
        ])
    elif side == "lay":
        out = out.with_columns([
            pl.lit("lay").alias("side"),
            pl.col("edge_lay").alias("edge"),
        ])
    else:
        # auto: choose the larger positive edge; if both negative, take the less negative (we'll filter later)
        out = out.with_columns([
            pl.when(pl.col("edge_back") >= pl.col("edge_lay")).then(pl.lit("back")).otherwise(pl.lit("lay")).alias("side"),
            pl.max_horizontal(["edge_back", "edge_lay"]).alias("edge"),
        ])
    return out


def _kelly_stake_units(p: np.ndarray, odds: np.ndarray, frac: float) -> np.ndarray:
    """
    Kelly fraction for BACK bets: f* = frac * (b p - q) / b, b = odds-1.
    For lay, we reuse this stake as 'exposure units'; we handle pnl separately.
    """
    b = np.maximum(odds - 1.0, 1e-9)
    q = 1.0 - p
    f = frac * (b * p - q) / b
    f = np.clip(f, 0.0, 1.0)  # no negative stakes
    return f


def _build_bets_table(
    df: pl.DataFrame,
    label_col: str,
    min_edge: float,
    kelly_frac: float,
    side_mode: str,
) -> pl.DataFrame:
    # compute edges & choose side
    df2 = _edge_columns(df, side=side_mode)

    # filter by edge threshold
    df2 = df2.filter(pl.col("edge") >= min_edge)

    # Kelly stakes in "units" (relative bankroll); we'll treat 1.0 as one unit
    p = df2["p_hat"].to_numpy()
    odds = df2["odds"].to_numpy()
    stake_units = _kelly_stake_units(p, odds, kelly_frac)

    df2 = df2.with_columns([
        pl.Series(name="stake_unit", values=stake_units),
        pl.col(label_col).alias("winLabel"),
    ])

    # keep minimal useful columns
    keep = [
        "sport", "marketId", "selectionId", "publishTimeMs", "tto_minutes",
        "odds", "implied_prob", "p_hat", "edge", "side", "stake_unit", "winLabel",
    ]
    have = [c for c in keep if c in df2.columns]
    return df2.select(have)


def _pick_topn_per_market(bets: pl.DataFrame, top_n: int) -> pl.DataFrame:
    """
    Pick top-N by edge within each market; works across old/new Polars (no row_number/cum_count assumptions).
    """
    if bets.is_empty():
        return bets
    return (
        bets.sort(["marketId", "edge"], descending=[False, True])
        .with_columns(
            (getattr(pl, "len", None) or pl.count)().over("marketId").alias("n_in_market"),
            # version-agnostic per-group rank: 0..N-1
            pl.arange(0, (getattr(pl, "len", None) or pl.count)()).over("marketId").alias("rank_in_market"),
        )
        .filter(pl.col("rank_in_market") < top_n)
        .drop(["n_in_market", "rank_in_market"])
    )


def _cap_stakes(bets: pl.DataFrame, cap_market: float, cap_day: float) -> pl.DataFrame:
    """
    Apply simple caps on cumulative stake units per market and per day.
    Scale stakes if cumulative would exceed the cap.
    Enforce Betfair's £1 minimum bet size by rounding up small positive stakes.
    """
    if bets.is_empty():
        return bets

    # market cap
    b1 = (
        bets.with_columns([
            pl.col("stake_unit").cum_sum().over("marketId").alias("cum_mkt"),
        ])
        .with_columns([
            pl.when(pl.col("cum_mkt") <= cap_market)
            .then(pl.col("stake_unit"))
            .otherwise(pl.col("stake_unit") * (cap_market / pl.col("cum_mkt")))
            .alias("stake_unit_capped_mkt")
        ])
        .drop(["cum_mkt"])
    )

    # day cap
    b1 = b1.with_columns([
        (pl.col("publishTimeMs") // (1000 * 60 * 60 * 24)).alias("day_bucket")
    ])
    b2 = (
        b1.with_columns([
            pl.col("stake_unit_capped_mkt").cum_sum().over("day_bucket").alias("cum_day"),
        ])
        .with_columns([
            pl.when(pl.col("cum_day") <= cap_day)
            .then(pl.col("stake_unit_capped_mkt"))
            .otherwise(pl.col("stake_unit_capped_mkt") * (cap_day / pl.col("cum_day")))
            .alias("stake_unit_final")
        ])
        .drop(["cum_day", "day_bucket"])
    )

    # enforce £1 minimum: round up all positive stakes below 1.0
    b2 = b2.with_columns([
        pl.when(pl.col("stake_unit_final") > 0)
        .then(pl.col("stake_unit_final").clip_lower(1.0))
        .otherwise(0.0)
        .alias("stake")
    ])

    return b2



def _pnl_columns(bets: pl.DataFrame, commission: float) -> pl.DataFrame:
    """
    Compute PnL per bet including commission.
      - BACK bet:
          win: (odds - 1) * stake
          lose: - stake
        After commission: multiply * (1 - commission) on profits only.
      - LAY bet (we're laying at 'odds'):
          If selection loses (winLabel==0): we win stake (counterparty's loss / odds?) Simplify:
            profit = stake * (1 - commission)
          If selection wins (winLabel==1): we lose (odds - 1) * stake
    """
    if bets.is_empty():
        return bets

    back_win = (pl.col("odds") - 1.0) * pl.col("stake")
    back_lose = -pl.col("stake")
    lay_win = pl.col("stake")                 # when selection loses
    lay_lose = -(pl.col("odds") - 1.0) * pl.col("stake")

    pnl_back = pl.when(pl.col("winLabel") == 1).then(back_win).otherwise(back_lose)
    pnl_lay = pl.when(pl.col("winLabel") == 0).then(lay_win).otherwise(lay_lose)

    pnl_raw = pl.when(pl.col("side") == "back").then(pnl_back).otherwise(pnl_lay)
    pnl_net = pl.when(pnl_raw > 0).then(pnl_raw * (1.0 - commission)).otherwise(pnl_raw)

    return bets.with_columns([
        pnl_raw.alias("pnl_gross"),
        pnl_net.alias("pnl"),
    ])


def _binwise_pnl(bets: pl.DataFrame, edges: List[int]) -> pl.DataFrame:
    """Aggregate PnL by horizon bins using manual label construction (polars version agnostic)."""
    if bets.is_empty() or "tto_minutes" not in bets.columns:
        return pl.DataFrame({"tto_bin": [], "n_bets": [], "stake": [], "pnl": []})

    labels = [f"{edges[i]:02d}-{edges[i+1]:02d}" for i in range(len(edges) - 1)]
    expr = (
        pl.when((pl.col("tto_minutes") > edges[0]) & (pl.col("tto_minutes") <= edges[1]))
        .then(pl.lit(labels[0]))
    )
    for i in range(1, len(labels)):
        lo, hi = edges[i], edges[i + 1]
        expr = expr.when((pl.col("tto_minutes") > lo) & (pl.col("tto_minutes") <= hi)).then(pl.lit(labels[i]))
    expr = expr.otherwise(pl.lit(None)).alias("tto_bin")

    tmp = bets.with_columns(expr)
    agg = (
        tmp.group_by("tto_bin", maintain_order=True)
        .agg([
            pl.len().alias("n_bets"),
            pl.sum("stake").alias("stake"),
            pl.sum("pnl").alias("pnl"),
        ])
        .sort("tto_bin")
    )
    return agg


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulate wagering from trained model(s).")
    # Mutually exclusive modes, defaults resolved at runtime:
    ap.add_argument("--model", help="Single model path (JSON).", default=None)
    ap.add_argument("--model-30", help="Short-horizon model path.", default=None)
    ap.add_argument("--model-180", help="Long-horizon model path.", default=None)
    ap.add_argument("--gate-mins", type=float, default=45.0, help="Boundary: <=gate uses model-30; >gate uses model-180.")

    # Data
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)

    # New range selection: --date is the end date; --days-before N runs end and N days before
    ap.add_argument("--date", required=True, help="End date (YYYY-MM-DD) to run up to (inclusive).")
    ap.add_argument("--days-before", type=int, default=0, help="How many days before --date to include (0 = just the date).")

    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    # Selection & staking
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=1.0)
    ap.add_argument("--stake-cap-day", type=float, default=10.0)

    # Outputs (will be redirected to ./output/)
    ap.add_argument("--bets-out", default="bets.csv")
    ap.add_argument("--agg-out", default="bets_by_market.csv")
    ap.add_argument("--bin-out", default="pnl_by_tto_bin.csv")

    args = ap.parse_args()

    # --- Mode resolution (mutually exclusive) ---
    single = args.model is not None
    dual = (args.model_30 is not None) or (args.model_180 is not None)

    if single and dual:
        raise SystemExit("Provide either --model OR (--model-30 and --model-180), not both.")

    if not single and not dual:
        # Try dual defaults in ./output, else single default
        m30 = OUTPUT_DIR / "model_30.json"
        m180 = OUTPUT_DIR / "model_180.json"
        if m30.exists() and m180.exists():
            args.model_30 = str(m30)
            args.model_180 = str(m180)
            dual = True
        else:
            m1 = OUTPUT_DIR / "xgb_model.json"
            if not m1.exists():
                raise SystemExit("No model paths provided and no default models found in ./output/")
            args.model = str(m1)
            single = True

    # Load boosters
    booster_single = booster_30 = booster_180 = None
    if single:
        booster_single = _load_booster(args.model)
    else:
        if args.model_30 is None or args.model_180 is None:
            raise SystemExit("Dual mode requires both --model-30 and --model-180.")
        booster_30 = _load_booster(args.model_30)
        booster_180 = _load_booster(args.model_180)

    # Dates
    days_total = int(args.days_before) + 1  # include the end date itself
    dates = _daterange(args.date, days_total)

    # Build features chunked
    df_parts: List[pl.DataFrame] = []
    for i in range(0, len(dates), args.chunk_days):
        dchunk = dates[i:i + args.chunk_days]
        df_c, _ = features.build_features_streaming(
            curated_root=args.curated,
            sport=args.sport,
            dates=dchunk,
            preoff_minutes=args.preoff_mins,
            batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
        )
        if not df_c.is_empty():
            df_parts.append(df_c)

    if not df_parts:
        raise SystemExit("No features produced for given date range.")

    df_feat = pl.concat(df_parts, how="vertical", rechunk=True)

    # Score
    def _score(df: pl.DataFrame) -> pl.DataFrame:
        if dual:
            early = df.filter(pl.col("tto_minutes") > args.gate_mins)
            late = df.filter(pl.col("tto_minutes") <= args.gate_mins)
            parts = []
            if not early.is_empty():
                fcols = _select_feature_cols(early, args.label_col)
                p = _predict_proba(early, booster_180, fcols)
                parts.append(early.with_columns(pl.lit(p).alias("p_hat")))
            if not late.is_empty():
                fcols = _select_feature_cols(late, args.label_col)
                p = _predict_proba(late, booster_30, fcols)
                parts.append(late.with_columns(pl.lit(p).alias("p_hat")))
            return pl.concat(parts, how="vertical") if parts else df.with_columns(pl.lit(None).alias("p_hat"))
        else:
            fcols = _select_feature_cols(df, args.label_col)
            p = _predict_proba(df, booster_single, fcols)
            return df.with_columns(pl.lit(p).alias("p_hat"))

    df_scored = _score(df_feat)

    # Build bets → pick topN → cap → pnl
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

    # Per-market aggregation
    agg = (
        bets.group_by("marketId")
        .agg([
            pl.len().alias("n_bets"),
            pl.sum("stake").alias("stake_total"),
            pl.sum("pnl").alias("pnl_total"),
        ])
        .sort("marketId")
    )

    # Binwise pnl
    bin_pnl = _binwise_pnl(bets, edges=[0, 30, 60, 90, 120, 180])

    # Write outputs to ./output/
    bets_out = _outpath(args.bets_out)
    agg_out = _outpath(args.agg_out)
    bin_out = _outpath(args.bin_out)

    bets.write_csv(bets_out)
    agg.write_csv(agg_out)
    bin_pnl.write_csv(bin_out)

    # Console summary
    total_bets = bets.height
    stake_sum = float(bets["stake"].sum()) if total_bets else 0.0
    pnl_sum = float(bets["pnl"].sum()) if total_bets else 0.0
    roi = (pnl_sum / stake_sum) if stake_sum > 0 else 0.0

    print(f"Saved bets to {bets_out}")
    print(f"Saved per-market aggregation to {agg_out}")
    print(f"Saved binwise pnl to {bin_out}")
    print(f"Summary: n_bets={total_bets} stake={stake_sum:.2f} pnl={pnl_sum:.2f} ROI={roi:.3%}")


if __name__ == "__main__":
    main()
