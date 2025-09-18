# ml/sim2.py — streaming simulator with execution realism, back/lay guardrails, partial fills, and simple persistence
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import polars as pl
import xgboost as xgb

from . import features


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _outpath(p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str(OUTPUT_DIR / pp.name)


def _daterange(end_date_str: str, days: int) -> List[str]:
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


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
    for name, dtype in df.schema.items():
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


def _load_booster(path: str) -> xgb.Booster:
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


# --- Betfair tick ladder helpers ---
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
    steps = round((odds - 1.01) / step)
    return round(1.01 + steps * step, 2)


def _step_ticks(odds: float, n: int) -> float:
    curr = _snap_to_tick(odds)
    if n == 0:
        return curr
    direction = 1 if n > 0 else -1
    for _ in range(abs(n)):
        s = _tick_size(curr)
        curr = max(1.01, min(1000.0, curr + direction * s))
    return round(curr, 2)


def _opposing_best(row: dict, side: str, have_back: bool, have_lay: bool, have_lists: bool) -> tuple[float, float]:
    """Return (price, size) at best opposing quote for the given row/side."""
    if side == "back":
        # match against lays
        if have_lay:
            return float(row.get("bestLayOdds", row.get("odds", np.nan))), float(row.get("bestLaySize", np.inf))
        if have_lists:
            sizes = row.get("laySizes") or []
            size = float(sizes[0]) if sizes else float("inf")
            return float(row.get("odds", np.nan)), size
        return float(row.get("odds", np.nan)), float("inf")
    else:
        # lay → match against backs
        if have_back:
            return float(row.get("bestBackOdds", row.get("odds", np.nan))), float(row.get("bestBackSize", np.inf))
        if have_lists:
            sizes = row.get("backSizes") or []
            size = float(sizes[0]) if sizes else float("inf")
            return float(row.get("odds", np.nan)), size
        return float(row.get("odds", np.nan)), float("inf")


def _edge_columns(df: pl.DataFrame, side: str = "auto") -> pl.DataFrame:
    if "p_hat" not in df.columns:
        raise RuntimeError("p_hat missing. Score the dataframe first.")
    src = "ltp" if "ltp" in df.columns else ("odds" if "odds" in df.columns else None)
    if not src:
        raise RuntimeError("ltp/odds missing for implied prob.")
    out = df.with_columns([
        pl.when(pl.col(src) > 1.0).then(pl.col(src)).otherwise(None).alias("odds"),
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
        out = out.with_columns([pl.lit("back").alias("side"), pl.col("edge_back").alias("edge")])
    elif side == "lay":
        out = out.with_columns([pl.lit("lay").alias("side"), pl.col("edge_lay").alias("edge")])
    else:
        out = out.with_columns([
            pl.when(pl.col("edge_back") >= pl.col("edge_lay")).then(pl.lit("back")).otherwise(pl.lit("lay")).alias("side"),
            pl.max_horizontal(["edge_back", "edge_lay"]).alias("edge"),
        ])
    return out


def _kelly_stake_units(p: np.ndarray, odds: np.ndarray, frac: float) -> np.ndarray:
    b = np.maximum(odds - 1.0, 1e-9)
    q = 1.0 - p
    f = frac * (b * p - q) / b
    return np.clip(f, 0.0, 1.0)


def _build_bets_table(df: pl.DataFrame, label_col: str, min_edge: float, kelly_frac: float, side_mode: str) -> pl.DataFrame:
    df2 = _edge_columns(df, side=side_mode).filter(pl.col("edge") >= min_edge)
    p = df2["p_hat"].to_numpy()
    odds = df2["odds"].to_numpy()
    stake_units = _kelly_stake_units(p, odds, kelly_frac)
    df2 = df2.with_columns([
        pl.Series(name="stake_unit", values=stake_units),
        pl.col(label_col).alias("winLabel"),
    ])
    keep = [
        "sport", "marketId", "selectionId", "publishTimeMs", "tto_minutes",
        "odds", "implied_prob", "p_hat", "edge", "side", "stake_unit", "winLabel",
        "bestBackOdds", "bestBackSize", "bestLayOdds", "bestLaySize", "backSizes", "laySizes",
    ]
    have = [c for c in keep if c in df2.columns]
    return df2.select(have)


def _pick_topn_per_market(bets: pl.DataFrame, top_n: int) -> pl.DataFrame:
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


def _cap_stakes(bets: pl.DataFrame, cap_market: float, cap_day: float, min_floor: float) -> pl.DataFrame:
    if bets.is_empty():
        return bets

    with_priority = bets.with_columns([
        (pl.col("edge") if "edge" in bets.columns else pl.lit(0.0)).alias("_priority_edge"),
        pl.col("stake_unit").alias("_priority_stake_unit"),
    ])

    mkt_sum = with_priority.group_by("marketId").agg(pl.sum("stake_unit").alias("_sum_mkt"))
    b1 = (
        with_priority.join(mkt_sum, on="marketId", how="left")
        .with_columns([
            pl.when(pl.col("_sum_mkt") > 0)
             .then((pl.lit(cap_market) / pl.col("_sum_mkt")))
             .otherwise(pl.lit(0.0))
             .alias("_mkt_scale")
        ])
        .with_columns([(pl.col("stake_unit") * pl.col("_mkt_scale")).alias("_stake_mkt")])
        .drop("_sum_mkt", "_mkt_scale")
    )

    b2 = b1.with_columns([
        pl.when(pl.col("_stake_mkt") <= 0.0).then(0.0)
        .when(pl.col("_stake_mkt") < min_floor).then(min_floor)
        .otherwise(pl.col("_stake_mkt")).alias("_stake_floor")
    ])

    def _renorm_group(df: pl.DataFrame, cap: float) -> pl.DataFrame:
        df = df.sort(by=["_priority_edge", "_priority_stake_unit"], descending=[True, True])
        stake = df["_stake_floor"].to_numpy().astype(float)
        pos = stake > 0
        n_pos = int(pos.sum())
        if n_pos == 0 or cap <= 0:
            stake[:] = 0.0
            return df.with_columns(pl.Series("_stake_grouped", stake))
        # if cap < base, select top k by priority at min_floor
        base = float(n_pos) * min_floor
        if cap < base:
            k = max(0, int(cap // min_floor))
            stake_new = np.zeros_like(stake)
            if k > 0:
                idx_pos = np.flatnonzero(pos)[:k]
                stake_new[idx_pos] = min_floor
            return df.with_columns(pl.Series("_stake_grouped", stake_new))
        budget = cap - base
        excess = np.where(pos, np.maximum(stake - min_floor, 0.0), 0.0)
        sum_excess = float(excess.sum())
        if sum_excess <= budget + 1e-12:
            return df.with_columns(pl.Series("_stake_grouped", stake))
        scale = budget / sum_excess if sum_excess > 0 else 0.0
        stake_new = np.where(pos, min_floor + excess * scale, 0.0)
        return df.with_columns(pl.Series("_stake_grouped", stake_new))

    if hasattr(b2.group_by("marketId"), "map_groups"):
        b3 = b2.group_by("marketId", maintain_order=True).map_groups(lambda g: _renorm_group(g, cap_market))
    else:
        parts = []
        for mkt in b2.select("marketId").unique().to_series().to_list():
            parts.append(_renorm_group(b2.filter(pl.col("marketId") == mkt), cap_market))
        b3 = pl.concat(parts, how="vertical")

    b4 = b3.with_columns([(pl.col("publishTimeMs") // (1000 * 60 * 60 * 24)).alias("_day_bucket")])
    if hasattr(b4.group_by("_day_bucket"), "map_groups"):
        b5 = b4.group_by("_day_bucket", maintain_order=True).map_groups(lambda g: _renorm_group(g, cap_day))
    else:
        parts = []
        for day in b4.select("_day_bucket").unique().to_series().to_list():
            parts.append(_renorm_group(b4.filter(pl.col("_day_bucket") == day), cap_day))
        b5 = pl.concat(parts, how="vertical")

    out = b5.rename({"_stake_grouped": "stake"}).drop(
        "_priority_edge", "_priority_stake_unit", "_stake_mkt", "_stake_floor", "_day_bucket"
    )
    return out


def _pnl_columns(bets: pl.DataFrame, commission: float) -> pl.DataFrame:
    if bets.is_empty():
        return bets
    odds_col = pl.col("odds_exec") if "odds_exec" in bets.columns else pl.col("odds")
    back_win = (odds_col - 1.0) * pl.col("stake") * (1.0 - commission)
    back_lose = -pl.col("stake")
    lay_win = pl.col("stake") * (1.0 - commission)
    lay_lose = -(odds_col - 1.0) * pl.col("stake")
    pnl_back = pl.when(pl.col("winLabel") == 1).then(back_win).otherwise(back_lose)
    pnl_lay = pl.when(pl.col("winLabel") == 0).then(lay_win).otherwise(lay_lose)
    pnl = pl.when(pl.col("side") == "back").then(pnl_back).otherwise(pnl_lay)
    return bets.with_columns(pnl.alias("pnl"))


class StreamState:
    def __init__(self, cooldown_secs: int, max_open_per_market: int, max_exposure_day: float, latency_ms: int):
        self.cooldown_secs = cooldown_secs
        self.max_open_per_market = max_open_per_market
        self.max_exposure_day = max_exposure_day
        self.latency_ms = latency_ms
        self.last_bet_ts: Dict[Tuple[str, int], int] = {}
        self.open_count_by_market: Dict[str, int] = {}
        self.exposure_day: float = 0.0
        self.unmatched: List[Dict] = []

    def can_place(self, mkt: str, sel: int, now_ms: int, stake: float) -> Tuple[bool, str]:
        key = (mkt, sel)
        last = self.last_bet_ts.get(key)
        if last is not None and (now_ms - last) < self.cooldown_secs * 1000:
            return False, "cooldown"
        if self.open_count_by_market.get(mkt, 0) >= self.max_open_per_market:
            return False, "max_open_per_market"
        if self.exposure_day + stake > self.max_exposure_day:
            return False, "max_exposure_day"
        return True, "ok"

    def register_bet(self, mkt: str, sel: int, now_ms: int, stake: float):
        self.last_bet_ts[(mkt, sel)] = now_ms
        self.open_count_by_market[mkt] = self.open_count_by_market.get(mkt, 0) + 1
        self.exposure_day += stake

    def add_exposure(self, stake: float):
        self.exposure_day += stake

    def settle_market(self, mkt: str):
        self.open_count_by_market[mkt] = 0


def main():
    ap = argparse.ArgumentParser(description="Streaming-style wagering sim from trained model(s).")
    # Mode
    ap.add_argument("--model", help="Single model path (JSON).", default=None)
    ap.add_argument("--model-30", help="Short-horizon model path.", default=None)
    ap.add_argument("--model-180", help="Long-horizon model path.", default=None)
    ap.add_argument("--gate-mins", type=float, default=45.0, help="Boundary: <=gate uses model-30; >gate uses model-180.")
    # Data
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True, help="End date (YYYY-MM-DD) to run up to (inclusive).")
    ap.add_argument("--days-before", type=int, default=0, help="How many days before --date to include (0 = just the date).")
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)
    # Streaming realism
    ap.add_argument("--stream-bucket-secs", type=int, default=5, help="Process ticks in N-second buckets (simulated stream clock).")
    ap.add_argument("--latency-ms", type=int, default=300, help="Decision latency added to timestamps before placement.")
    ap.add_argument("--cooldown-secs", type=int, default=60, help="Min seconds between bets on same selection.")
    ap.add_argument("--max-open-per-market", type=int, default=1, help="Throttle concurrent bets per market.")
    ap.add_argument("--max-exposure-day", type=float, default=5000.0, help="Cap total daily exposure (sum of stakes).")
    ap.add_argument("--place-until-mins", type=float, default=1.0, help="Stop placing when tto_minutes <= this (final minute freeze).")
    # Persistence (simple queue model)
    ap.add_argument("--persistence", choices=["lapse", "keep"], default="keep",
                    help="What to do with unmatched remainder: 'keep' tries to fill across subsequent buckets.")
    ap.add_argument("--rest-secs", type=int, default=15, help="How long to keep trying to match remainder before lapsing.")
    # Execution realism
    ap.add_argument("--min-stake", type=float, default=1.0, help="Minimum tradable stake; orders below are rejected (Betfair API min is £1).")
    ap.add_argument("--tick-snap", action="store_true", help="Snap execution odds to Betfair tick ladder.")
    ap.add_argument("--slip-ticks", type=int, default=1, help="Adverse slippage in ticks applied due to latency.")
    # Selection & staking
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--side", choices=["back", "lay", "auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=50.0)
    ap.add_argument("--stake-cap-day", type=float, default=2000.0)
    # Guardrails / EV / odds
    ap.add_argument("--odds-min", type=float, default=1.6, help="Global lower odds bound at decision time.")
    ap.add_argument("--odds-max", type=float, default=6.0, help="Global upper odds bound at decision time.")
    ap.add_argument("--back-odds-min", type=float, default=None, help="Override lower bound for BACK; default=odds-min.")
    ap.add_argument("--back-odds-max", type=float, default=None, help="Override upper bound for BACK; default=odds-max.")
    ap.add_argument("--lay-odds-min", type=float, default=None, help="Override lower bound for LAY; default=odds-min.")
    ap.add_argument("--lay-odds-max", type=float, default=None, help="Override upper bound for LAY; default=odds-max.")
    ap.add_argument("--max-stake-per-bet", type=float, default=5.0, help="Hard cap per bet after all scaling.")
    ap.add_argument("--max-liability-per-bet", type=float, default=20.0, help="LAY: cap (odds-1)*stake to this amount.")
    ap.add_argument("--min-ev", dest="min_ev", type=float, default=0.02, help="Min EV per £1 at execution odds (after commission).")
    # Outputs
    ap.add_argument("--bets-out", default="bets.csv")
    ap.add_argument("--agg-out", default="bets_by_market.csv")
    ap.add_argument("--bin-out", default="pnl_by_tto_bin.csv")
    ap.add_argument("--stream-log-out", default="stream_log.csv")

    args = ap.parse_args()

    # Resolve mode
    single = args.model is not None
    dual = (args.model_30 is not None) or (args.model_180 is not None)
    if single and dual:
        raise SystemExit("Provide either --model OR (--model-30 and --model-180), not both.")
    if not single and not dual:
        m1 = OUTPUT_DIR / "xgb_model.json"
        if not m1.exists():
            raise SystemExit("No model path provided and default ./output/xgb_model.json not found")
        args.model = str(m1)
        single = True

    booster_single = booster_30 = booster_180 = None
    if single:
        booster_single = _load_booster(args.model)
    else:
        if args.model_30 is None or args.model_180 is None:
            raise SystemExit("Dual mode requires both --model-30 and --model-180.")
        booster_30 = _load_booster(args.model_30)
        booster_180 = _load_booster(args.model_180)

    # Build features
    dates = _daterange(args.date, int(args.days_before) + 1)
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
        # Graceful empty run
        print("No features produced for given date range.")
        pl.DataFrame({"marketId": [], "n_bets": [], "stake_total": [], "pnl_total": []}).write_csv(_outpath(args.agg_out))
        pl.DataFrame({"tto_bin": [], "n_bets": [], "stake": [], "pnl": []}).write_csv(_outpath(args.bin_out))
        pl.DataFrame([]).write_csv(_outpath(args.bets_out))
        return

    df_feat = pl.concat(df_parts, how="vertical", rechunk=True)

    # Stream clock buckets
    bucket_ms = max(1, int(args.stream_bucket_secs)) * 1000
    df_feat = df_feat.sort(["publishTimeMs", "marketId", "selectionId"]).with_columns(
        (pl.col("publishTimeMs") // bucket_ms).alias("_bucket")
    )

    def _score_arrivals(arrivals: pl.DataFrame) -> pl.DataFrame:
        if arrivals.is_empty():
            return arrivals
        if dual:
            early = arrivals.filter(pl.col("tto_minutes") > args.gate_mins)
            late = arrivals.filter(pl.col("tto_minutes") <= args.gate_mins)
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

    state = StreamState(
        cooldown_secs=args.cooldown_secs,
        max_open_per_market=args.max_open_per_market,
        max_exposure_day=args.max_exposure_day,
        latency_ms=args.latency_ms,
    )

    placed_rows: List[pl.DataFrame] = []
    stream_log_rows: List[Dict] = []

    # helper: try to fill unmatched remainders present in state
    def _try_fill_unmatched(arrivals_df: pl.DataFrame, now_ms: int):
        if not state.unmatched or arrivals_df.is_empty():
            return
        cols = set(arrivals_df.columns)
        have_back = {"bestBackOdds", "bestBackSize"}.issubset(cols)
        have_lay = {"bestLayOdds", "bestLaySize"}.issubset(cols)
        have_lists = {"backSizes", "laySizes"}.issubset(cols)
        snap = arrivals_df.sort("publishTimeMs").group_by(["marketId", "selectionId"]).tail(1)
        snap_map = {(r["marketId"], int(r["selectionId"])): r for r in snap.iter_rows(named=True)}

        still: List[Dict] = []
        exec_rows = []

        for od in state.unmatched:
            if now_ms - od["placed_ms"] > args.rest_secs * 1000 or args.persistence == "lapse":
                stream_log_rows.append({
                    "publishTimeMs": now_ms, "marketId": od["marketId"], "selectionId": od["selectionId"],
                    "event": "lapse_unfilled", "remaining": od["remaining"]
                })
                continue

            key = (od["marketId"], od["selectionId"])
            row = snap_map.get(key)
            if not row:
                still.append(od)
                continue

            best_px, best_sz = _opposing_best(row, od["side"], have_back, have_lay, have_lists)
            if not np.isfinite(best_px) or best_sz <= 0:
                still.append(od)
                continue

            # Fill only if the market "crosses" our limit
            limit_px = od["limit_px"]
            cross = (od["side"] == "back" and best_px <= limit_px) or (od["side"] == "lay" and best_px >= limit_px)
            if not cross:
                still.append(od)
                continue

            fill_amt = min(od["remaining"], float(best_sz))
            if fill_amt <= 0:
                still.append(od)
                continue

            # record execution at opposing price
            exec_rows.append({
                "marketId": od["marketId"],
                "selectionId": od["selectionId"],
                "publishTimeMs": now_ms,
                "tto_minutes": row.get("tto_minutes"),
                "side": od["side"],
                "stake": float(fill_amt),
                "odds": row.get("odds", best_px),
                "odds_exec": float(best_px),
                "winLabel": row.get("winLabel"),
                "p_hat": row.get("p_hat"),
                "edge": row.get("edge"),
            })
            stream_log_rows.append({
                "publishTimeMs": now_ms, "marketId": od["marketId"], "selectionId": od["selectionId"],
                "event": "place_partial", "stake": float(fill_amt), "odds_exec": float(best_px)
            })
            od["remaining"] -= float(fill_amt)
            state.add_exposure(float(fill_amt))

            if od["remaining"] > 1e-9:
                still.append(od)

        state.unmatched = still
        if exec_rows:
            placed_rows.append(pl.DataFrame(exec_rows))

    for _, arrivals in df_feat.group_by("_bucket", maintain_order=True):
        arrivals = arrivals.drop("_bucket")
        now_ms = int(arrivals["publishTimeMs"].max()) + args.latency_ms

        # settle markets that have started (based on this bucket)
        finishers = arrivals.filter(pl.col("tto_minutes") <= 0)
        if not finishers.is_empty():
            for mkt in finishers.select("marketId").unique().to_series().to_list():
                state.settle_market(mkt)

        # optional final-minute freeze for NEW placements
        arrivals_for_new = arrivals.filter(pl.col("tto_minutes") > args.place_until_mins)

        # first, try to fill any previously unmatched
        _try_fill_unmatched(arrivals, now_ms)

        if not arrivals_for_new.is_empty():
            scored = _score_arrivals(arrivals_for_new)

            # candidate bets
            cands = _build_bets_table(scored, label_col=args.label_col, min_edge=args.min_edge, kelly_frac=args.kelly, side_mode=args.side)
            cands = _pick_topn_per_market(cands, args.top_n_per_market)
            cands = _cap_stakes(cands, cap_market=args.stake_cap_market, cap_day=args.stake_cap_day, min_floor=args.min_stake)

            if not cands.is_empty():
                executed_rows = []
                cols = set(cands.columns)
                have_back = {"bestBackOdds", "bestBackSize"}.issubset(cols)
                have_lay = {"bestLayOdds", "bestLaySize"}.issubset(cols)
                have_lists = {"backSizes", "laySizes"}.issubset(cols)

                for r in cands.iter_rows(named=True):
                    mkt = r["marketId"]
                    sel = int(r["selectionId"])
                    side = r.get("side", "back")
                    stake_req = float(r.get("stake", 0.0))

                    # throttles for NEW placements
                    if stake_req < args.min_stake:
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": "below_min_stake"})
                        continue
                    ok, why = state.can_place(mkt, sel, now_ms, stake_req)
                    if not ok:
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": why})
                        continue

                    # level-1 quote
                    q_price, q_size = _opposing_best(r, side, have_back, have_lay, have_lists)
                    if not np.isfinite(q_price) or q_price <= 1.0 or q_size <= 0:
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": "no_quote"})
                        continue

                    # pick side-aware odds band (falls back to global bounds)
                    if side == "back":
                        omin = args.back_odds_min if args.back_odds_min is not None else args.odds_min
                        omax = args.back_odds_max if args.back_odds_max is not None else args.odds_max
                    else:
                        omin = args.lay_odds_min if args.lay_odds_min is not None else args.odds_min
                        omax = args.lay_odds_max if args.lay_odds_max is not None else args.odds_max
                    if (q_price < omin) or (q_price > omax):
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "reject", "reason": "odds_band"})
                        continue

                    # tick snap + adverse slippage on NEW placement
                    px_exec = _snap_to_tick(q_price) if args.tick_snap else float(q_price)
                    if args.slip_ticks != 0 and args.latency_ms > 0:
                        if side == "back":
                            px_exec = _step_ticks(px_exec, -abs(args.slip_ticks))  # worse for backer is lower odds
                        else:
                            px_exec = _step_ticks(px_exec, +abs(args.slip_ticks))  # worse for layer is higher odds

                    # EV-at-execution gate (after snap/slip, with commission)
                    p = float(r.get("p_hat", 0.0))
                    if side == "back":
                        ev_per_1 = p * (px_exec - 1.0) * (1.0 - args.commission) - (1.0 - p)
                    else:
                        ev_per_1 = (1.0 - p) * (1.0 - args.commission) - p * (px_exec - 1.0)
                    if ev_per_1 < args.min_ev:
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel,
                                                "event": "reject", "reason": "ev_below_threshold", "ev": ev_per_1, "px_exec": px_exec})
                        continue

                    # Per-bet caps
                    stake_req = min(stake_req, args.max_stake_per_bet)
                    if side == "lay":
                        mult = max(px_exec - 1.0, 1e-9)  # liability = mult * stake
                        max_stake_by_liab = args.max_liability_per_bet / mult
                        if max_stake_by_liab <= 0:
                            stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel,
                                                    "event": "reject", "reason": "liability_cap_zero"})
                            continue
                        stake_req = min(stake_req, max_stake_by_liab)

                    # immediate fill at L1 (can be < min-stake): accept partials since stake_req >= min-stake
                    fill = min(stake_req, float(q_size))
                    if fill > 0:
                        state.register_bet(mkt, sel, now_ms, float(fill))
                        executed_rows.append({
                            **{k: r.get(k) for k in r.keys()},
                            "stake": float(fill),
                            "odds_exec": float(px_exec),
                        })
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "place", "stake": float(fill), "odds_exec": float(px_exec)})

                    # persist unmatched remainder if requested
                    remaining = stake_req - float(fill)
                    if remaining > 1e-9 and args.persistence == "keep":
                        state.unmatched.append({
                            "marketId": mkt,
                            "selectionId": sel,
                            "side": side,
                            "limit_px": float(px_exec),  # our limit at submission after snap+slip
                            "remaining": float(remaining),
                            "placed_ms": now_ms,
                        })
                        stream_log_rows.append({"publishTimeMs": now_ms, "marketId": mkt, "selectionId": sel, "event": "queue_remainder", "remaining": float(remaining), "limit_px": float(px_exec)})

                if executed_rows:
                    placed_rows.append(pl.DataFrame(executed_rows))

    # Build bets df with known schema even if empty
    if placed_rows:
        bets = pl.concat(placed_rows, how="vertical")
    else:
        bets = pl.DataFrame(schema={
            "marketId": pl.Utf8,
            "selectionId": pl.Int64,
            "publishTimeMs": pl.Int64,
            "tto_minutes": pl.Float64,
            "side": pl.Utf8,
            "stake": pl.Float64,
            "odds": pl.Float64,
            "odds_exec": pl.Float64,
            "winLabel": pl.Int64,
            "pnl": pl.Float64,
        })

    if not bets.is_empty():
        bets = _pnl_columns(bets, commission=args.commission)
    else:
        if "pnl" not in bets.columns:
            bets = bets.with_columns(pl.lit(None).cast(pl.Float64).alias("pnl"))

    # Per-market aggregation (safe on empty)
    if bets.is_empty():
        agg = pl.DataFrame({"marketId": [], "n_bets": [], "stake_total": [], "pnl_total": []})
    else:
        agg = (
            bets.group_by("marketId")
            .agg([pl.len().alias("n_bets"), pl.sum("stake").alias("stake_total"), pl.sum("pnl").alias("pnl_total")])
            .sort("marketId")
        )

    # Binwise pnl (safe on empty)
    edges = [0, 30, 60, 90, 120, 180]
    if not bets.is_empty() and "tto_minutes" in bets.columns:
        labels = [f"{edges[i]:02d}-{edges[i+1]:02d}" for i in range(len(edges) - 1)]
        expr = pl.when((pl.col("tto_minutes") > edges[0]) & (pl.col("tto_minutes") <= edges[1])).then(pl.lit(labels[0]))
        for i in range(1, len(labels)):
            lo, hi = edges[i], edges[i + 1]
            expr = expr.when((pl.col("tto_minutes") > lo) & (pl.col("tto_minutes") <= hi)).then(pl.lit(labels[i]))
        expr = expr.otherwise(pl.lit(None)).alias("tto_bin")
        tmp = bets.with_columns(expr)
        bin_pnl = (
            tmp.group_by("tto_bin", maintain_order=True)
            .agg([pl.len().alias("n_bets"), pl.sum("stake").alias("stake"), pl.sum("pnl").alias("pnl")])
            .sort("tto_bin")
        )
    else:
        bin_pnl = pl.DataFrame({"tto_bin": [], "n_bets": [], "stake": [], "pnl": []})

    # Write outputs
    bets_out = _outpath(args.bets_out)
    agg_out = _outpath(args.agg_out)
    bin_out = _outpath(args.bin_out)
    log_out = _outpath(args.stream_log_out)

    bets.write_csv(bets_out)
    agg.write_csv(agg_out)
    bin_pnl.write_csv(bin_out)
    if stream_log_rows:
        pl.DataFrame(stream_log_rows).write_csv(log_out)

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
