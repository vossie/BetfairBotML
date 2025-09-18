# ml/sim.py — streaming-style simulation
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

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


# ----------------------------- date & feature utils -----------------------------

def _daterange(end_date_str: str, days: int) -> List[str]:
    """
    Build a list of YYYY-MM-DD strings covering [end - (days-1) ... end], inclusive.
    Example: end=2025-09-15, days=2 -> ["2025-09-14","2025-09-15"]
    """
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    out: List[str] = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _is_numeric_dtype(dt) -> bool:
    try:
        return dt.is_numeric()
    except AttributeError:
        return dt in {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }


def _select_feature_cols(df: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {
        "sport", "marketId", "selectionId", "ts", "ts_ms",
        "publishTimeMs", label_col, "runnerStatus", "p_hat",
    }
    cols: List[str] = []
    schema = df.schema  # dict[name -> dtype]
    for name, dtype in schema.items():
        if name in exclude:
            continue
        lname = name.lower()
        if "label" in lname or "target" in lname:
            continue
        if _is_numeric_dtype(dtype):
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
# --- Betfair tick ladder helpers ---

# odds bands and tick sizes per Betfair ladder
_TICK_BANDS = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.1),
    (6.0, 10.0, 0.2),
    (10.0, 20.0, 0.5),
    (20.0, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0),
    (100.0, 1000.0, 10.0),
]


def _tick_size(odds: float) -> float:
    for lo, hi, step in _TICK_BANDS:
        if lo <= odds < hi:
            return step
    return 10.0


def _snap_to_tick(odds: float) -> float:
    odds = max(1.01, min(1000.0, float(odds)))
    step = _tick_size(odds)
    lo = 1.01
    # compute number of steps from floor of band start; approximate by rounding to nearest multiple
    return round((odds - lo) / step) * step + lo


def _step_ticks(odds: float, n: int) -> float:
    # move n ticks across the ladder (positive increases odds)
    if n == 0:
        return _snap_to_tick(odds)
    curr = _snap_to_tick(odds)
    direction = 1 if n > 0 else -1
    for _ in range(abs(n)):
        step = _tick_size(curr)
        curr += direction * step
        curr = max(1.01, min(1000.0, curr))
    return curr


def _edge_columns(df: pl.DataFrame, side: str = "auto") -> pl.DataFrame:
    """
    Compute fair odds from p_hat; implied from ltp; edge for back/lay; pick side if auto.
    Assumptions:
      - 'ltp' is the available back price proxy (odds). If missing/<=1, drop the row.
      - edge_back = p_hat - implied_prob
      - edge_lay  = implied_prob - p_hat (mirror)
    """
    have_cols = set(df.columns)
    if "p_hat" not in have_cols:
        raise RuntimeError("p_hat missing. Score the dataframe first.")
    if "ltp" not in have_cols:
        raise RuntimeError("ltp missing. Needed for implied_prob/odds.")

    out = df.with_columns([
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
        out = out.with_columns([
            pl.when(pl.col("edge_back") >= pl.col("edge_lay")).then(pl.lit("back")).otherwise(pl.lit("lay")).alias("side"),
            pl.max_horizontal(["edge_back", "edge_lay"]).alias("edge"),
        ])
    return out


def _kelly_stake_units(p: np.ndarray, odds: np.ndarray, frac: float) -> np.ndarray:
    """
    Kelly fraction for BACK bets: f* = frac * (b p - q) / b, where b = odds-1.
    For lay, we reuse this stake as 'exposure units'; PnL handled separately.
    """
    b = np.maximum(odds - 1.0, 1e-9)
    q = 1.0 - p
    f = frac * (b * p - q) / b
    f = np.clip(f, 0.0, 1.0)
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

    # Kelly stakes in "units" (relative bankroll)
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
    """Pick top-N by edge within each market; Polars-version-agnostic using per-group arange."""
    if bets.is_empty():
        return bets
    length_expr = getattr(pl, "len", None) or pl.count
    return (
        bets.sort(["marketId", "edge"], descending=[False, True])
        .with_columns(
            length_expr().over("marketId").alias("n_in_market"),
            pl.arange(0, length_expr()).over("marketId").alias("rank_in_market"),
        )
        .filter(pl.col("rank_in_market") < top_n)
        .drop(["n_in_market", "rank_in_market"])
    )


def _cap_stakes(bets: pl.DataFrame, cap_market: float, cap_day: float) -> pl.DataFrame:
    """
    Cap per market and per day with a £1 minimum stake and re-normalization.
    Works on older Polars (no GroupBy.apply / no Expr.clip_lower).
    """
    if bets.is_empty():
        return bets

    # Priority columns used when caps are too tight
    with_priority = bets.with_columns([
        (pl.col("edge") if "edge" in bets.columns else pl.lit(0.0)).alias("_priority_edge"),
        pl.col("stake_unit").alias("_priority_stake_unit"),
    ])

    # Step 1: proportional market cap on stake_unit
    mkt_sum = with_priority.group_by("marketId").agg(pl.sum("stake_unit").alias("_sum_mkt"))
    b1 = (
        with_priority.join(mkt_sum, on="marketId", how="left")
        .with_columns([
            pl.when(pl.col("_sum_mkt") > 0)
            .then(pl.min_horizontal([pl.lit(1.0), pl.lit(cap_market) / pl.col("_sum_mkt")]))
            .otherwise(pl.lit(0.0))
            .alias("_mkt_scale")
        ])
        .with_columns([
            (pl.col("stake_unit") * pl.col("_mkt_scale")).alias("_stake_mkt")
        ])
        .drop("_sum_mkt", "_mkt_scale")
    )

    # Step 2: apply £1 floor to any positive stake
    b2 = b1.with_columns([
        pl.when(pl.col("_stake_mkt") <= 0.0).then(0.0)
        .when(pl.col("_stake_mkt") < 1.0).then(1.0)
        .otherwise(pl.col("_stake_mkt"))
        .alias("_stake_floor")
    ])

    # ---- helper: re-normalize within a group under a cap with £1 floor ----
    def _renorm_group(df: pl.DataFrame, cap: float) -> pl.DataFrame:
        df = df.sort(by=["_priority_edge", "_priority_stake_unit"], descending=[True, True])

        stake = df["_stake_floor"].to_numpy().astype(float)
        pos = stake > 0
        n_pos = int(pos.sum())

        if n_pos == 0 or cap <= 0:
            stake[:] = 0.0
            return df.with_columns(pl.Series("_stake_grouped", stake))

        if cap < n_pos:
            # keep top k positive bets at exactly £1; drop the rest
            k = int(cap)
            stake_new = np.zeros_like(stake)
            if k > 0:
                idx_pos = np.flatnonzero(pos)
                keep_idx = idx_pos[:k]
                stake_new[keep_idx] = 1.0
            return df.with_columns(pl.Series("_stake_grouped", stake_new))

        # give everyone £1, then scale down excess to fit remaining budget
        base = float(n_pos)
        budget = cap - base

        excess = np.where(pos, np.maximum(stake - 1.0, 0.0), 0.0)
        sum_excess = float(excess.sum())

        if sum_excess <= budget + 1e-12:
            return df.with_columns(pl.Series("_stake_grouped", stake))

        scale = budget / sum_excess if sum_excess > 0 else 0.0
        stake_new = np.where(pos, 1.0 + excess * scale, 0.0)
        return df.with_columns(pl.Series("_stake_grouped", stake_new))

    # ---- Step 3: re-normalize per market (map_groups if available; else Python loop) ----
    if hasattr(b2.group_by("marketId"), "map_groups"):
        b3 = b2.group_by("marketId", maintain_order=True).map_groups(lambda g: _renorm_group(g, cap_market))
    else:
        pieces = []
        for mkt in b2.select("marketId").unique().to_series().to_list():
            g = b2.filter(pl.col("marketId") == mkt)
            pieces.append(_renorm_group(g, cap_market))
        b3 = pl.concat(pieces, how="vertical")

    # ---- Step 4: re-normalize per day similarly ----
    b4 = b3.with_columns([
        (pl.col("publishTimeMs") // (1000 * 60 * 60 * 24)).alias("_day_bucket")
    ])

    if hasattr(b4.group_by("_day_bucket"), "map_groups"):
        b5 = b4.group_by("_day_bucket", maintain_order=True).map_groups(lambda g: _renorm_group(g, cap_day))
    else:
        pieces = []
        for day in b4.select("_day_bucket").unique().to_series().to_list():
            g = b4.filter(pl.col("_day_bucket") == day)
            pieces.append(_renorm_group(g, cap_day))
        b5 = pl.concat(pieces, how="vertical")

    out = b5.rename({"_stake_grouped": "stake"}).drop(
        "_priority_edge", "_priority_stake_unit", "_stake_mkt", "_stake_floor", "_day_bucket"
    )
    return out


def _pnl_columns(bets: pl.DataFrame, commission: float) -> pl.DataFrame:
    """
    Compute PnL per bet including Betfair commission on winnings.
      - BACK:
          win: (odds_exec - 1) * stake * (1 - commission)
          lose: -stake
      - LAY:
          selection loses: stake * (1 - commission)
          selection wins : -(odds_exec - 1) * stake
    Falls back to 'odds' if 'odds_exec' missing.
    """
    if bets.is_empty():
        return bets

    odds_col = pl.when(pl.col("odds_exec").is_not_null()).then(pl.col("odds_exec")).otherwise(pl.col("odds")).alias("_od")

    back_win = (pl.col("_od") - 1.0) * pl.col("stake") * (1.0 - commission)
    back_lose = -pl.col("stake")

    lay_win = pl.col("stake") * (1.0 - commission)              # selection loses
    lay_lose = -(pl.col("_od") - 1.0) * pl.col("stake")        # selection wins

    pnl_back = pl.when(pl.col("winLabel") == 1).then(back_win).otherwise(back_lose)
    pnl_lay = pl.when(pl.col("winLabel") == 0).then(lay_win).otherwise(lay_lose)

    pnl = pl.when(pl.col("side") == "back").then(pnl_back).otherwise(pnl_lay)
    out = bets.with_columns([odds_col, pnl.alias("pnl")]).drop("_od")
    return out


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


# ----------------------------- streaming sim core -----------------------------

class StreamState:
    """Holds intraday state resembling a live trading process."""
    def __init__(self, cooldown_secs: int, max_open_per_market: int, max_exposure_day: float, latency_ms: int):
        self.cooldown_secs = cooldown_secs
        self.max_open_per_market = max_open_per_market
        self.max_exposure_day = max_exposure_day
        self.latency_ms = latency_ms
        # runtime state
        self.last_bet_ts: Dict[Tuple[str, int], int] = {}  # (marketId, selectionId) -> publishTimeMs
        self.open_count_by_market: Dict[str, int] = {}
        self.exposure_day: float = 0.0

    def can_place(self, mkt: str, sel: int, now_ms: int, stake: float) -> bool:
        # cooldown per (market, selection)
        key = (mkt, sel)
        last = self.last_bet_ts.get(key)
        if last is not None:
            if (now_ms - last) < self.cooldown_secs * 1000:
                return False
        # max open per market
        if self.open_count_by_market.get(mkt, 0) >= self.max_open_per_market:
            return False
        # daily exposure cap
        if self.exposure_day + stake > self.max_exposure_day:
            return False
        return True

    def register_bet(self, mkt: str, sel: int, now_ms: int, stake: float):
        self.last_bet_ts[(mkt, sel)] = now_ms
        self.open_count_by_market[mkt] = self.open_count_by_market.get(mkt, 0) + 1
        self.exposure_day += stake

    def settle_market(self, mkt: str):
        # when a market finishes, reset open count; cooldowns remain
        self.open_count_by_market[mkt] = 0


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Streaming-style wagering sim from trained model(s).").")

    # Mutually exclusive modes, defaults resolved at runtime:
    ap.add_argument("--model", help="Single model path (JSON).", default=None)
    ap.add_argument("--model-30", help="Short-horizon model path.", default=None)
    ap.add_argument("--model-180", help="Long-horizon model path.", default=None)
    ap.add_argument("--gate-mins", type=float, default=45.0, help="Boundary: <=gate uses model-30; >gate uses model-180.")

    # Data
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)

    # Range selection: --date is the end date; --days-before N runs end and N days before
    ap.add_argument("--date", required=True, help="End date (YYYY-MM-DD) to run up to (inclusive).")
    ap.add_argument("--days-before", type=int, default=0, help="How many days before --date to include (0 = just the date).")

    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    # Streaming realism
    ap.add_argument("--stream-bucket-secs", type=int, default=5, help="Process ticks in N-second buckets (simulated stream clock).")
    ap.add_argument("--latency-ms", type=int, default=500, help="Decision latency added to timestamps before placement.")
    ap.add_argument("--cooldown-secs", type=int, default=60, help="Min seconds between bets on the same (marketId,selectionId).")
    ap.add_argument("--max-open-per-market", type=int, default=1, help="Throttle concurrent bets per market.")
    ap.add_argument("--max-exposure-day", type=float, default=1000.0, help="Cap total daily exposure (sum of stakes).")
    ap.add_argument("--place-until-mins", type=float, default=1.0, help="Stop placing when tto_minutes <= this (final minute freeze).")

    # Execution realism
    ap.add_argument("--min-stake", type=float, default=2.0, help="Minimum tradable stake; orders below are rejected.")
    ap.add_argument("--tick-snap", action="store_true", help="Snap execution odds to Betfair tick ladder.")
    ap.add_argument("--slip-ticks", type=int, default=1, help="Adverse slippage in ticks applied due to latency.").")

    # Selection & staking
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.05)  # Betfair 5% default
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=10.0)
    ap.add_argument("--stake-cap-day", type=float, default=100.0)

    # Outputs (will be redirected to ./output/)
    ap.add_argument("--bets-out", default="bets.csv")
    ap.add_argument("--agg-out", default="bets_by_market.csv")
    ap.add_argument("--bin-out", default="pnl_by_tto_bin.csv")
    ap.add_argument("--stream-log-out", default="stream_log.csv")

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

    # Build features chunked (raw stream source)
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

    # Sort by time and bucket to simulate a clock
    bucket_ms = max(1, int(args.stream_bucket_secs)) * 1000
    df_feat = df_feat.sort(["publishTimeMs", "marketId", "selectionId"]) \
                     .with_columns((pl.col("publishTimeMs") // bucket_ms).alias("_bucket"))

    # Score-on-arrival helper (avoids any future leakage)
    def _score_arrivals(arrivals: pl.DataFrame) -> pl.DataFrame:
        if arrivals.is_empty():
            return arrivals
        if dual:
            early = arrivals.filter(pl.col("tto_minutes") > args.gate_mins)
            late  = arrivals.filter(pl.col("tto_minutes") <= args.gate_mins)
            parts = []
            if not early.is_empty():
                fcols = _select_feature_cols(early, args.label_col)
                p = _predict_proba(early, booster_180, fcols)
                parts.append(early.with_columns(pl.lit(p).alias("p_hat")))
            if not late.is_empty():
                fcols = _select_feature_cols(late, args.label_col)
                p = _predict_proba(late, booster_30, fcols)
                parts.append(late.with_columns(pl.lit(p).alias("p_hat")))
            return pl.concat(parts, how="vertical") if parts else arrivals.with_columns(pl.lit(None).alias("p_hat"))
        else:
            fcols = _select_feature_cols(arrivals, args.label_col)
            p = _predict_proba(arrivals, booster_single, fcols)
            return arrivals.with_columns(pl.lit(p).alias("p_hat"))

    # Stream loop
    state = StreamState(
        cooldown_secs=args.cooldown_secs,
        max_open_per_market=args.max_open_per_market,
        max_exposure_day=args.max_exposure_day,
        latency_ms=args.latency_ms,
    )

    placed_rows: List[pl.DataFrame] = []
    stream_log_rows: List[Dict] = []

    for bucket_id, arrivals in df_feat.group_by("_bucket", maintain_order=True):
        arrivals = arrivals.drop("_bucket")
        now_ms = int(arrivals["publishTimeMs"].max()) + args.latency_ms

        # stop placing very late to mimic exchange cutoff / in-running avoidance
        arrivals = arrivals.filter(pl.col("tto_minutes") > args.place_until_mins)
        if arrivals.is_empty():
            # We still may settle markets that crossed tto<=0 in this bucket
            pass
        else:
            scored = _score_arrivals(arrivals)

            # Build candidate bets and apply per-market top-N + static caps
            cands = _build_bets_table(
                scored,
                label_col=args.label_col,
                min_edge=args.min_edge,
                kelly_frac=args.kelly,
                side_mode=args.side,
            )
            cands = _pick_topn_per_market(cands, args.top_n_per_market)
            cands = _cap_stakes(cands, cap_market=args.stake_cap_market, cap_day=args.stake_cap_day)

            # Enforce streaming constraints + execution realism (fills at L1 with slippage/tick-snap)
            if not cands.is_empty():
                executed_rows = []

                # choose columns for best quotes if present
                cols = set(cands.columns)
                have_back = {"bestBackOdds", "bestBackSize"}.issubset(cols)
                have_lay = {"bestLayOdds", "bestLaySize"}.issubset(cols)
                have_lists = {"backSizes", "laySizes"}.issubset(cols)

                def _extract_best(row, side: str):
                    # returns (price, size) at best level for the CONTRARY side
                    if side == "back":
                        # match against lays
                        if have_lay:
                            return float(row.get("bestLayOdds", row.get("odds", np.nan))), float(row.get("bestLaySize", np.inf))
                        if have_lists:
                            sizes = row.get("laySizes") or []
                            size = float(sizes[0]) if sizes else float("inf")
                            return float(row.get("odds", np.nan)), size
                        return float(row.get("odds", np.nan)), float("inf")
                    else:  # lay → match against backs
                        if have_back:
                            return float(row.get("bestBackOdds", row.get("odds", np.nan))), float(row.get("bestBackSize", np.inf))
                        if have_lists:
                            sizes = row.get("backSizes") or []
                            size = float(sizes[0]) if sizes else float("inf")
                            return float(row.get("odds", np.nan)), size
                        return float(row.get("odds", np.nan)), float("inf")

                for r in cands.iter_rows(named=True):
                    mkt = r["marketId"]
                    sel = int(r["selectionId"])
                    stake_req = float(r.get("stake", 0.0))
                    side = r.get("side", "back")

                    # throttle checks first
                    if stake_req <= 0.0:
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": "zero_stake"})
                        continue
                    if not state.can_place(mkt, sel, now_ms, stake_req):
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": "throttle_or_cap"})
                        continue

                    # level-1 quote
                    q_price, q_size = _extract_best(r, side)
                    if not np.isfinite(q_price) or q_price <= 1.0:
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": "no_quote"})
                        continue

                    # optional tick snap
                    px_exec = _snap_to_tick(q_price) if args.tick_snap else float(q_price)

                    # adverse slippage due to latency: move by +/− slip_ticks against us
                    if args.slip_ticks != 0 and args.latency_ms > 0:
                        if side == "back":
                            # worse for backer is LOWER odds
                            px_exec = _step_ticks(px_exec, -abs(args.slip_ticks))
                        else:  # lay
                            # worse for layer is HIGHER odds
                            px_exec = _step_ticks(px_exec, +abs(args.slip_ticks))

                    # enforce min stake and available size
                    fill = min(stake_req, float(q_size))
                    if fill < args.min_stake:
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": "insufficient_size_or_below_min"})
                        continue

                    # accept and record
                    state.register_bet(mkt, sel, now_ms, fill)
                    row_out = {
                        **{k: r.get(k) for k in r.keys()},
                        "stake": fill,
                        "odds_exec": px_exec,
                    }
                    executed_rows.append(row_out)
                    stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "place", "stake": fill, "odds_exec": px_exec})

                if executed_rows:
                    placed_rows.append(pl.DataFrame(executed_rows))

        # settle markets that have effectively started (tto<=0) in this bucket
        finishers = arrivals.filter(pl.col("tto_minutes") <= 0)
        if not finishers.is_empty():
            for mkt in finishers.select("marketId").unique().to_series().to_list():
                state.settle_market(mkt)

    bets = pl.concat(placed_rows, how="vertical") if placed_rows else pl.DataFrame([])

    # Compute PnL on settled labels. In a real stream you know the label later; here we already have winLabel
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
    log_out = _outpath(args.stream_log_out)

    bets.write_csv(bets_out)
    agg.write_csv(agg_out)
    bin_pnl.write_csv(bin_out)
    if stream_log_rows:
        pl.DataFrame(stream_log_rows).write_csv(log_out)

    # Console summary
    total_bets = bets.height
    stake_sum = float(bets["stake"].sum()) if total_bets else 0.0
    pnl_sum = float(bets["pnl"].sum()) if total_bets else 0.0
    roi = (pnl_sum / stake_sum) if stake_sum > 0 else 0.0

    print(f"Saved bets to {bets_out}")
    print(f"Saved per-market aggregation to {agg_out}")
    print(f"Saved binwise pnl to {bin_out}")
    if stream_log_rows:
        print(f"Saved stream log to {log_out}")
    print(f"Summary: n_bets={total_bets} stake={stake_sum:.2f} pnl={pnl_sum:.2f} ROI={roi:.3%}")


if __name__ == "__main__":
    main()
