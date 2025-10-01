#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import timedelta, timezone
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any

import numpy as np
import polars as pl
import xgboost as xgb
import pickle
import json

from utils import (
    parse_date, read_snapshots,
    time_to_off_minutes_expr, filter_preoff,
    write_json, kelly_fraction_back, kelly_fraction_lay
)

UTC = timezone.utc

LAG_30S = 6
LAG_60S = 12
LAG_120S = 24

BASE_FEATS = [
    "ltp","tradedVolume","spreadTicks","imbalanceBest1","mins_to_off","__p_now",
    "ltp_diff_5s","vol_diff_5s",
    "ltp_lag30s","ltp_lag60s","ltp_lag120s",
    "tradedVolume_lag30s","tradedVolume_lag60s","tradedVolume_lag120s",
    "ltp_mom_30s","ltp_mom_60s","ltp_mom_120s",
    "ltp_ret_30s","ltp_ret_60s","ltp_ret_120s",
]
FEATURES_WITH_COUNTRY = BASE_FEATS + ["is_gb","country_freq"]

def parse_args():
    p = argparse.ArgumentParser(
        description="Streaming backtest with portfolioing, liquidity VWAP, contra leg, MTM/settlement P&L, and daily ROI."
    )
    # Data/time
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")

    # Modeling horizon / window
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--preoff-max", type=int, default=30)

    # Trading economics
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--ev-cap", type=float, default=1.0)
    p.add_argument("--ev-scale", type=float, default=1.0)
    p.add_argument("--edge-thresh", type=float, default=0.002)

    # Staking / bankroll
    p.add_argument("--stake-mode", choices=["flat","kelly"], default="kelly")
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)

    # Odds filter
    p.add_argument("--odds-min", type=float, default=None)
    p.add_argument("--odds-max", type=float, default=None)

    # Liquidity enforcement
    p.add_argument("--liquidity-levels", type=int, default=1)
    p.add_argument("--enforce-liquidity", action="store_true")
    p.add_argument("--require-book", action="store_true")
    p.add_argument("--min-fill-frac", type=float, default=0.0)

    # Per-market portfolioing (price-move dutching)
    p.add_argument("--per-market-topk", type=int, default=2, help="Max runners per market/time (side-aware).")
    p.add_argument("--per-market-budget", type=float, default=50.0, help="£ cap across chosen runners in a market/time.")
    p.add_argument("--basket-sizing", choices=["equal_ev","inverse_ev","fractional_kelly"], default="equal_ev",
                   help="How to allocate the per-market budget across chosen runners.")
    p.add_argument("--exit-on-move-ticks", type=int, default=0,
                   help="If >0, mark if price moved N ticks in the profitable direction before horizon.")

    # Contra leg controls
    p.add_argument("--contra-mode", choices=["off","prob","always"], default="off",
                   help="off=no contra; prob=place contra only if swing_prob≥thresh; always=always place contra.")
    p.add_argument("--contra-frac", type=float, default=0.25, help="Stake fraction for contra vs primary filled stake.")
    p.add_argument("--contra-hold-secs", type=int, default=300, help="Extra holding time for contra leg.")
    p.add_argument("--contra-prob-thresh", type=float, default=0.65, help="Swing probability threshold for contra.")
    p.add_argument("--contra-beta", type=float, default=80.0, help="Sensitivity for swing prob: swing_prob=exp(-beta*|dp|).")

    # Runtime / model / output
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--calib-path", default="")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=75_000)
    p.add_argument("--write-trades", type=int, default=0)
    return p.parse_args()

def _has_book_columns(df: pl.DataFrame) -> bool:
    cols = set(df.columns)
    return {"backTicks","backSizes","layTicks","laySizes"}.issubset(cols)

def _make_features_row(bufs, last_row):
    if len(bufs["ltp"]) < LAG_120S or len(bufs["vol"]) < LAG_120S:
        return None
    ltp_now = last_row["ltp"]; vol_now = last_row["tradedVolume"]
    ltp_l30 = bufs["ltp"][-LAG_30S]; ltp_l60 = bufs["ltp"][-LAG_60S]; ltp_l120 = bufs["ltp"][-LAG_120S]
    vol_l30 = bufs["vol"][-LAG_30S]; vol_l60 = bufs["vol"][-LAG_60S]; vol_l120 = bufs["vol"][-LAG_120S]
    ltp_prev = bufs["ltp"][-1] if len(bufs["ltp"])>=1 else None
    vol_prev = bufs["vol"][-1] if len(bufs["vol"])>=1 else None
    ltp_diff_5s = (ltp_now - ltp_prev) if ltp_prev is not None else 0.0
    vol_diff_5s = (vol_now - vol_prev) if vol_prev is not None else 0.0
    def safe_div(a, b): return (a / b) if (b is not None and abs(b) > 1e-12) else None
    feats = {
        "ltp": ltp_now, "tradedVolume": vol_now,
        "spreadTicks": last_row.get("spreadTicks", 0.0) or 0.0,
        "imbalanceBest1": last_row.get("imbalanceBest1", 0.0) or 0.0,
        "mins_to_off": last_row["mins_to_off"],
        "__p_now": 1.0 / max(ltp_now, 1e-12),
        "ltp_diff_5s": ltp_diff_5s, "vol_diff_5s": vol_diff_5s,
        "ltp_lag30s": ltp_l30, "ltp_lag60s": ltp_l60, "ltp_lag120s": ltp_l120,
        "tradedVolume_lag30s": vol_l30, "tradedVolume_lag60s": vol_l60, "tradedVolume_lag120s": vol_l120,
        "ltp_mom_30s": ltp_now - ltp_l30, "ltp_mom_60s": ltp_now - ltp_l60, "ltp_mom_120s": ltp_now - ltp_l120,
        "ltp_ret_30s": safe_div(ltp_now - ltp_l30, ltp_l30),
        "ltp_ret_60s": safe_div(ltp_now - ltp_l60, ltp_l60),
        "ltp_ret_120s": safe_div(ltp_now - ltp_l120, ltp_l120),
    }
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = np.nan
    return feats

# Tick ladder helpers
_TICKS = [
    (1.01, 2.00, 0.01), (2.00, 3.00, 0.02), (3.00, 4.00, 0.05), (4.00, 6.00, 0.10),
    (6.00,10.00,0.20), (10.0,20.0,0.50), (20.0,30.0,1.00), (30.0,50.0,2.00),
    (50.0,100.0,5.00), (100.0,1000.0,10.0),
]
_cum_ticks = []; cum = 0
for lo, hi, step in _TICKS:
    n = int(round((hi - lo) / step)) + 1
    _cum_ticks.append((lo, hi, step, cum, cum + n - 1))
    cum += n
def _odds_from_tick_index(tick_index: int) -> float:
    if tick_index < 0: tick_index = 0
    for lo, hi, step, t0, t1 in _cum_ticks:
        if t0 <= tick_index <= t1:
            k = tick_index - t0
            return round(lo + k * step, 2)
    lo, hi, step, t0, t1 = _cum_ticks[-1]
    k = min(t1, max(t0, tick_index)) - t0
    return round(lo + k * step, 2)
def _tick_index_from_odds(odds: float) -> float:
    acc = 0
    for lo, hi, step, t0, t1 in _cum_ticks:
        if lo <= odds <= hi:
            k = int(round((odds - lo) / step))
            return t0 + k
        acc = t1 + 1
    return acc

def _vwap_fill(side: str, ticks: list[int], sizes: list[float], need: float, levels: int) -> tuple[float,float]:
    if not ticks or not sizes or need <= 0.0: return 0.0, None
    L = max(1, int(levels)); filled = 0.0; notional = 0.0
    for i in range(min(L, len(ticks))):
        px = _odds_from_tick_index(int(ticks[i])); size_i = float(sizes[i])
        if size_i <= 0: continue
        take = min(need - filled, size_i)
        if take <= 0: break
        filled += take; notional += take * px
        if filled >= need - 1e-12: break
    if filled <= 0.0: return 0.0, None
    return filled, float(notional / filled)

def _ev_mtm(p_now: float, p_pred: float, commission: float, side: str) -> float:
    eps = 1e-6
    p_now_c  = max(eps, min(1.0 - eps, p_now))
    p_pred_c = max(eps, min(1.0 - eps, p_pred))
    if side == "back":
        return ((p_pred_c / p_now_c) - 1.0) * (1.0 - commission)
    else:
        return (1.0 - (p_now_c / p_pred_c)) * (1.0 - commission)

def _ev_settlement(p_pred: float, ltp: float, commission: float, side: str) -> float:
    if side == "back":
        return p_pred * (ltp - 1.0) * (1.0 - commission) - (1.0 - p_pred)
    else:
        return (1.0 - p_pred) * (1.0 - commission) - p_pred * (ltp - 1.0)

def _load_results(curated_root: Path, sport: str, start_dt, end_dt) -> pl.DataFrame:
    days = []; d = start_dt
    while d <= end_dt:
        days.append(d.strftime("%Y-%m-%d")); d += timedelta(days=1)
    lfs = []
    for day in days:
        lfs.append(
            pl.scan_parquet(f"{curated_root}/results/sport={sport}/date={day}/*.parquet")
              .select([
                  pl.col("marketId").cast(pl.Utf8),
                  pl.col("selectionId").cast(pl.Int64),
                  pl.col("winLabel").cast(pl.Int32).fill_null(0),
              ])
        )
    if not lfs:
        return pl.DataFrame({"marketId": [], "selectionId": [], "winLabel": []})
    return pl.concat(lfs, how="vertical_relaxed").unique().collect()

def _prepare_exit_prices(df_snap: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        pl.col("marketId").cast(pl.Utf8),
        pl.col("selectionId").cast(pl.Int64),
        pl.from_epoch(pl.col("publishTimeMs").cast(pl.Int64), time_unit="ms")
          .dt.replace_time_zone("UTC").alias("ts_dt"),
        pl.col("ltp").cast(pl.Float64).alias("ltp_exit_proxy"),
    ]
    if _has_book_columns(df_snap):
        exprs += [pl.col("backTicks"), pl.col("backSizes"), pl.col("layTicks"), pl.col("laySizes")]
    return (
        df_snap
        .filter(pl.col("ltp").is_not_null() & (pl.col("ltp") > 1.01))
        .select(exprs)
        .sort(["marketId","selectionId","ts_dt"])
    )

def _basket_weights(cands: List[Dict[str, Any]], mode: str) -> List[float]:
    if not cands: return []
    if mode == "equal_ev":
        w = np.array([max(1e-9, c["ev_per_1"]) for c in cands], dtype=float)
    elif mode == "inverse_ev":
        w = np.array([1.0 / max(1e-9, c["ev_per_1"]) for c in cands], dtype=float)
    else:  # fractional_kelly
        w = np.array([max(1e-9, c.get("kelly_frac", 1e-3)) for c in cands], dtype=float)
    s = float(w.sum())
    if s <= 0: return [1.0 / len(cands)] * len(cands)
    return list(w / s)

def _swing_prob(dp: float, beta: float) -> float:
    # Heuristic: larger |dp| -> lower swing probability; small |dp| -> more likely to oscillate
    # swing_prob in (0,1): exp(-beta * |dp|)
    return float(np.exp(-float(beta) * abs(float(dp))))

def main():
    args = parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    curated = Path(args.curated)

    asof_dt = parse_date(args.asof)
    start_dt = parse_date(args.start_date)
    valid_end = asof_dt
    valid_start = asof_dt - timedelta(days=args.valid_days)

    # ---- Column-pruned snapshot scan ----
    cols = ["sport","marketId","selectionId","publishTimeMs","ltp","tradedVolume","spreadTicks","imbalanceBest1","marketStartMs","countryCode"]
    need_book = bool(args.enforce_liquidity)
    if need_book:
        cols += ["backTicks","backSizes","layTicks","laySizes"]

    lf = read_snapshots(curated, start_dt, valid_end, args.sport)
    schema_names = set(lf.collect_schema().names())
    present = [c for c in cols if c in schema_names]
    lf = lf.select(present).with_columns([ time_to_off_minutes_expr() ])

    df = lf.collect()
    df = filter_preoff(df, args.preoff_max)
    df = df.filter(pl.col("ltp").is_not_null() & (pl.col("ltp") > 1.01))
    if args.odds_min is not None: df = df.filter(pl.col("ltp") >= float(args.odds_min))
    if args.odds_max is not None: df = df.filter(pl.col("ltp") <= float(args.odds_max))

    # ---- Country facets ----
    df = df.with_columns(
        pl.from_epoch(pl.col("publishTimeMs"), time_unit="ms").dt.replace_time_zone("UTC").dt.date().alias("__date")
    )
    df_tr = df.filter(pl.col("__date") < valid_start)
    df_va = df.filter(pl.col("__date") >= valid_start)

    if "countryCode" in df.columns:
        freq = (
            df_tr.group_by("countryCode").agg(pl.len().alias("cnt"))
                 .with_columns((pl.col("cnt") / pl.col("cnt").sum()).alias("freq"))
                 .select(["countryCode","freq"])
        )
        def add_country_feats(dfi: pl.DataFrame) -> pl.DataFrame:
            dfi = dfi.with_columns((pl.col("countryCode") == "GB").cast(pl.Int8).alias("is_gb"))
            dfi = dfi.join(freq, on="countryCode", how="left").with_columns(pl.col("freq").fill_null(0.0).alias("country_freq"))
            return dfi
        df_tr = add_country_feats(df_tr)
        df_va = add_country_feats(df_va)
    else:
        df_tr = df_tr.with_columns([pl.lit(0).cast(pl.Int8).alias("is_gb"), pl.lit(0.0).alias("country_freq")])
        df_va = df_va.with_columns([pl.lit(0).cast(pl.Int8).alias("is_gb"), pl.lit(0.0).alias("country_freq")])

    df = pl.concat([df_tr, df_va], how="vertical_relaxed").sort(["marketId","selectionId","publishTimeMs"])

    # Liquidity availability & effective setting
    book_available = _has_book_columns(df)
    effective_enforce = bool(args.enforce_liquidity and book_available)
    if args.enforce_liquidity and not book_available:
        msg = "[simulate] NOTE: depth columns missing; "
        msg += "require-book=on → dropping ALL trades (no liquidity info)." if args.require_book else \
               "falling back to LTP exec/exit and disabling liquidity enforcement."
        print(msg)
    print(f"[simulate] Liquidity: requested={'on' if args.enforce_liquidity else 'off'}, "
          f"effective={'on' if effective_enforce else 'off'}; require_book={'on' if args.require_book else 'off'}")

    if args.enforce_liquidity and args.require_book and not book_available:
        _write_empty(outdir, args); return

    exit_df = _prepare_exit_prices(df)

    # ----- Model (+ optional calibrator) -----
    bst = xgb.Booster(); bst.load_model(args.model_path)
    try: bst.set_param({"device": args.device})
    except Exception: pass
    calibrator = None
    if args.calib_path:
        with open(args.calib_path, "rb") as f: calibrator = pickle.load(f)

    # rolling buffers per runner
    bufs_ltp = defaultdict(lambda: deque(maxlen=LAG_120S))
    bufs_vol = defaultdict(lambda: deque(maxlen=LAG_120S))

    # candidate buffer per (marketId, publishTimeMs) for portfolio selection
    cand_buf: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)

    feat_names = FEATURES_WITH_COUNTRY

    def _predict_dp(feat_row: Dict[str, float]) -> float:
        X = np.array([[feat_row.get(c, np.nan) for c in feat_names]], dtype=np.float32)
        try:
            return float(bst.inplace_predict(X, validate_features=False)[0])
        except Exception:
            return float(bst.predict(xgb.DMatrix(X, feature_names=feat_names))[0])

    def flush_candidates_for_time(market_id: str, ts_ms: int):
        key = (market_id, ts_ms)
        cands = cand_buf.get(key, [])
        if not cands:
            return []

        # Split by side and rank
        longs  = [c for c in cands if c["side"] == "back"]
        shorts = [c for c in cands if c["side"] == "lay"]

        longs.sort(key=lambda c: c["ev_per_1"], reverse=True)
        shorts.sort(key=lambda c: c["ev_per_1"], reverse=True)

        pick_longs  = longs[:max(0, args.per_market_topk)]
        pick_shorts = shorts[:max(0, args.per_market_topk)]
        picks = pick_longs + pick_shorts
        if not picks:
            cand_buf.pop(key, None)
            return []

        # Kelly fraction only if needed
        need_kelly = (args.stake_mode == "kelly") or (args.basket_sizing == "fractional_kelly")
        if need_kelly:
            for c in picks:
                kf = kelly_fraction_back(c["p_pred"], c["ltp"], args.commission) if c["side"] == "back" \
                     else kelly_fraction_lay (c["p_pred"], c["ltp"], args.commission)
                c["kelly_frac"] = max(args.kelly_floor, min(args.kelly_cap, kf))
        else:
            for c in picks: c["kelly_frac"] = 0.0

        # Basket sizing
        budget = float(args.per_market_budget)
        weights = _basket_weights(picks, args.basket_sizing)

        # Stakes requested
        stakes_req = [budget * w for w in weights]  # both flat and kelly use market budget distribution

        realized = []
        for c, stake_req in zip(picks, stakes_req):
            if stake_req <= 0:
                continue
            # Execute primary
            if effective_enforce and c["has_book"]:
                avail_ticks = c["backTicks"] if c["side"] == "back" else c["layTicks"]
                avail_sizes = c["backSizes"] if c["side"] == "back" else c["laySizes"]
                stake_filled, exec_vwap = _vwap_fill(c["side"], avail_ticks, avail_sizes, stake_req, args.liquidity_levels)
            elif args.enforce_liquidity and args.require_book:
                continue
            else:
                stake_filled = stake_req
                exec_vwap = float(c["ltp"])

            if stake_filled <= 0 or exec_vwap is None:
                continue

            fill_frac = stake_filled / stake_req if stake_req > 1e-9 else 1.0
            if fill_frac < float(args.min_fill_frac):
                continue

            exp_pnl_raw = c["ev_raw"] * stake_filled
            exp_pnl     = c["ev_per_1"] * stake_filled

            primary_rec = {
                "publishTimeMs": c["ts_ms"],
                "marketId": c["marketId"],
                "selectionId": c["selectionId"],
                "ltp": c["ltp"],
                "p_now": c["p_now"],
                "p_pred": c["p_pred"],
                "delta_pred": c["dp"],
                "side": c["side"],
                "is_contra": 0,
                "ev_per_1": c["ev_per_1"],
                "stake_req": float(stake_req),
                "stake_filled": float(stake_filled),
                "fill_frac": float(fill_frac),
                "exec_odds": float(exec_vwap),
                "exp_pnl": float(exp_pnl),
                "exp_pnl_raw": float(exp_pnl_raw),
                "stake_mode": args.stake_mode,
                # Primary exit at main horizon:
                "ts_exit_ms": int(c["ts_ms"] + args.horizon_secs * 1000),
            }
            realized.append(primary_rec)

            # Optionally place contra
            place_contra = False
            if args.contra_mode == "always":
                place_contra = True
            elif args.contra_mode == "prob":
                sp = _swing_prob(c["dp"], args.contra_beta)
                place_contra = (sp >= float(args.contra_prob_thresh))

            if place_contra:
                contra_side = "lay" if c["side"] == "back" else "back"
                contra_req  = float(args.contra_frac) * float(stake_filled)
                if contra_req > 0:
                    if effective_enforce and c["has_book"]:
                        avail_ticks = c["layTicks"] if contra_side == "lay" else c["backTicks"]
                        avail_sizes = c["laySizes"] if contra_side == "lay" else c["backSizes"]
                        contra_filled, contra_vwap = _vwap_fill(contra_side, avail_ticks, avail_sizes, contra_req, args.liquidity_levels)
                    elif args.enforce_liquidity and args.require_book:
                        contra_filled, contra_vwap = 0.0, None
                    else:
                        contra_filled, contra_vwap = contra_req, float(c["ltp"])

                    if contra_filled > 0 and contra_vwap is not None:
                        contra_rec = {
                            "publishTimeMs": c["ts_ms"],
                            "marketId": c["marketId"],
                            "selectionId": c["selectionId"],
                            "ltp": c["ltp"],
                            "p_now": c["p_now"],
                            "p_pred": c["p_pred"],
                            "delta_pred": c["dp"],
                            "side": contra_side,
                            "is_contra": 1,
                            "ev_per_1": c["ev_per_1"],  # reported; EV applies to primary signal context
                            "stake_req": float(contra_req),
                            "stake_filled": float(contra_filled),
                            "fill_frac": float(contra_filled/contra_req if contra_req>1e-9 else 1.0),
                            "exec_odds": float(contra_vwap),
                            "exp_pnl": 0.0,           # neutral: we don't add EV for contra (pure swing expression)
                            "exp_pnl_raw": 0.0,
                            "stake_mode": args.stake_mode,
                            # Contra exit at extended horizon:
                            "ts_exit_ms": int(c["ts_ms"] + (args.horizon_secs + args.contra_hold_secs) * 1000),
                        }
                        realized.append(contra_rec)

        cand_buf.pop(key, None)
        return realized

    # Stream rows and build features + candidates
    trades_records: List[pl.DataFrame] = []
    last_key = None

    for r in df.iter_rows(named=True):
        mkt = r["marketId"]; sel = int(r["selectionId"])
        ts_ms = int(r["publishTimeMs"])
        key = (mkt, ts_ms)

        # Maintain per-runner buffers
        runner_key = (mkt, sel)
        bufs_ltp[runner_key].append(float(r["ltp"]))
        bufs_vol[runner_key].append(float(r["tradedVolume"]))

        feats_base = _make_features_row({"ltp": bufs_ltp[runner_key], "vol": bufs_vol[runner_key]}, {
            "ltp": float(r["ltp"]),
            "tradedVolume": float(r["tradedVolume"]),
            "spreadTicks": float(r.get("spreadTicks") or 0.0),
            "imbalanceBest1": float(r.get("imbalanceBest1") or 0.0),
            "mins_to_off": float(r["mins_to_off"]),
        })
        if feats_base is None:
            continue

        feats_base["is_gb"] = float(1.0 if (r.get("countryCode") == "GB") else 0.0)
        feats_base["country_freq"] = float(r.get("country_freq") or 0.0)

        dp = _predict_dp(feats_base)
        p_now = 1.0 / max(1e-12, float(r["ltp"]))
        p_pred = p_now + dp
        p_pred = 0.0 if p_pred < 0.0 else (1.0 if p_pred > 1.0 else p_pred)

        side = "back" if dp > 0 else ("lay" if dp < 0 else "none")
        if side == "none":
            continue

        # EV computation (raw and scaled)
        if args.ev_mode == "mtm":
            ev_raw = _ev_mtm(p_now, p_pred, args.commission, side)
        else:
            ev_raw = _ev_settlement(p_pred, float(r["ltp"]), args.commission, side)
        ev_per_1 = ev_raw * float(args.ev_scale)
        if ev_per_1 < float(args.edge_thresh):
            continue

        # Add to candidate buffer
        has_book = book_available and (r.get("backTicks") is not None or r.get("layTicks") is not None)
        cand_buf[key].append({
            "ts_ms": ts_ms,
            "marketId": mkt,
            "selectionId": sel,
            "ltp": float(r["ltp"]),
            "p_now": float(p_now),
            "p_pred": float(p_pred),
            "dp": float(dp),
            "side": side,
            "ev_raw": float(ev_raw),
            "ev_per_1": float(ev_per_1),
            "stake_mode": args.stake_mode,
            "has_book": bool(has_book),
            "backTicks": r.get("backTicks") or [],
            "backSizes": r.get("backSizes") or [],
            "layTicks": r.get("layTicks") or [],
            "laySizes": r.get("laySizes") or [],
        })

        # Whenever market/time changes, flush previous key
        if last_key is None:
            last_key = key
        elif key != last_key:
            realized = flush_candidates_for_time(last_key[0], last_key[1])
            if realized:
                trades_records.append(_to_df(realized))
            last_key = key

    # Flush final batch
    if last_key is not None:
        realized = flush_candidates_for_time(last_key[0], last_key[1])
        if realized:
            trades_records.append(_to_df(realized))

    if not trades_records:
        print("[simulate] no trades generated.")
        _write_empty(outdir, args)
        return

    trades = pl.concat(trades_records, how="vertical_relaxed")

    # Exit price joining (use per-row ts_exit_ms if present, else default horizon)
    if "ts_exit_ms" not in trades.columns:
        trades = trades.with_columns(
            (pl.col("publishTimeMs").cast(pl.Int64) + args.horizon_secs * 1000).alias("ts_exit_ms")
        )

    trades = trades.with_columns([
        pl.from_epoch((pl.col("publishTimeMs").cast(pl.Int64)), time_unit="ms").dt.replace_time_zone("UTC").alias("ts_dt"),
        pl.from_epoch(pl.col("ts_exit_ms").cast(pl.Int64), time_unit="ms").dt.replace_time_zone("UTC").alias("ts_exit_dt"),
    ])

    exit_df = _prepare_exit_prices(df).sort(["marketId","selectionId","ts_dt"])
    trades_sorted = trades.sort(["marketId","selectionId","ts_exit_dt"])
    trades2 = trades_sorted.join_asof(
        exit_df,
        left_on="ts_exit_dt", right_on="ts_dt",
        by=["marketId","selectionId"],
        strategy="forward",
        tolerance=timedelta(minutes=10),
    ).with_columns(
        pl.col("ltp_exit_proxy").alias("exit_odds")
    ).drop(["ts_dt","ts_exit_dt","ltp_exit_proxy"]).filter(
        pl.col("exit_odds").is_not_null() & (pl.col("exit_odds") > 1.01)
    )

    # Optional tick-target annotation on primary only
    if int(args.exit_on_move_ticks) > 0:
        n_ticks = int(args.exit_on_move_ticks)
        trades2 = trades2.with_columns([
            pl.col("exec_odds").map_elements(_tick_index_from_odds, return_dtype=pl.Int64).alias("tick_entry"),
            pl.col("exit_odds").map_elements(_tick_index_from_odds, return_dtype=pl.Int64).alias("tick_exit"),
        ]).with_columns([
            pl.when(pl.col("side") == "back")
              .then((pl.col("tick_entry") - pl.col("tick_exit")) >= n_ticks)
              .otherwise((pl.col("tick_exit") - pl.col("tick_entry")) >= n_ticks)
              .alias("__hit_tick_target")
        ])

    if args.min_fill_frac > 0:
        trades2 = trades2.filter(pl.col("fill_frac") >= float(args.min_fill_frac))

    EV_CAP = float(args.ev_cap)
    trades2 = trades2.with_columns(pl.col("ev_per_1").clip(-EV_CAP, EV_CAP))

    # MTM P&L
    trades2 = trades2.with_columns([
        pl.when(pl.col("side") == "back")
          .then(((pl.col("exec_odds") / pl.col("exit_odds")) - 1.0) * pl.col("stake_filled"))
          .otherwise(1.0 - (pl.col("exit_odds") / pl.col("exec_odds")) * pl.col("stake_filled"))
          .alias("real_mtm_pnl_raw")
    ]).with_columns(
        pl.when(pl.col("real_mtm_pnl_raw").is_finite()).then(pl.col("real_mtm_pnl_raw")).otherwise(pl.lit(None)).alias("real_mtm_pnl")
    ).drop("real_mtm_pnl_raw")

    # Settlement P&L (requires results)
    results_df = _load_results(curated, args.sport, start_dt, valid_end)
    if results_df.height > 0:
        comm = float(args.commission)
        trades2 = (trades2.join(results_df, on=["marketId","selectionId"], how="left")
                        .with_columns(pl.col("winLabel").fill_null(0).cast(pl.Int32))
                        .with_columns(
                            pl.when(pl.col("side") == "back")
                              .then(pl.when(pl.col("winLabel") == 1)
                                        .then((pl.col("exec_odds") - 1.0) * pl.col("stake_filled") * (1.0 - comm))
                                        .otherwise(-pl.col("stake_filled")))
                              .otherwise(pl.when(pl.col("winLabel") == 1)
                                        .then(-(pl.col("exec_odds") - 1.0) * pl.col("stake_filled"))
                                        .otherwise(pl.col("stake_filled") * (1.0 - comm)))
                              .alias("real_settle_pnl")
                        ))
    else:
        trades2 = trades2.with_columns(pl.lit(None, dtype=pl.Float64).alias("real_settle_pnl"))

    # Persist trades (optional)
    if int(args.write_trades) == 1:
        trades_path = Path(args.output_dir) / f"trades_{args.asof}.parquet"
        trades2.write_parquet(trades_path)

    # Daily summary with ROI (primary and contra combined)
    daily = (
        trades2
        .with_columns(
            pl.from_epoch(pl.col("publishTimeMs").cast(pl.Int64), time_unit="ms")
              .dt.replace_time_zone("UTC").dt.date().cast(pl.Utf8).alias("day")
        )
        .group_by("day","stake_mode")
        .agg([
            pl.len().alias("n_trades"),
            pl.sum("exp_pnl").alias("exp_profit"),
            pl.mean("ev_per_1").alias("avg_ev"),
            pl.mean("stake_filled").alias("avg_stake_filled"),
            pl.mean("fill_frac").alias("avg_fill_frac"),
            pl.sum("real_mtm_pnl").alias("real_mtm_profit"),
            pl.sum("real_settle_pnl").alias("real_settle_profit"),
            pl.mean("is_contra").alias("share_contra"),
        ])
        .with_columns([
            (pl.col("exp_profit")        / pl.lit(float(args.bankroll_nom))).alias("roi_exp"),
            (pl.col("real_mtm_profit")   / pl.lit(float(args.bankroll_nom))).alias("roi_real_mtm"),
            (pl.col("real_settle_profit")/ pl.lit(float(args.bankroll_nom))).alias("roi_real_settle"),
        ])
        .sort("day")
    )
    daily_csv = Path(args.output_dir) / f"daily_{args.asof}.csv"
    daily.write_csv(daily_csv)

    # Overall summary
    total_exp_profit   = float(trades2["exp_pnl"].fill_null(0.0).sum())
    
m = trades2.select(pl.col("ev_per_1").fill_null(0.0).mean().alias("m")).to_dict(as_series=False)["m"][0] if "ev_per_1" in trades2.columns else 0.0

avg_ev = float(0.0 if m is None or (isinstance(m,float) and (math.isnan(m) or math.isinf(m))) else m)

avg_ev_scaled = avg_ev * float(args.ev_scale)
    total_real_mtm     = float(trades2["real_mtm_pnl"].fill_null(0.0).sum())
    total_real_settle  = float(trades2["real_settle_pnl"].fill_null(0.0).sum())

    roi_exp = (total_exp_profit / float(args.bankroll_nom)) if args.bankroll_nom else None
    roi_real_mtm = (total_real_mtm / float(args.bankroll_nom)) if args.bankroll_nom else None
    roi_real_settle = (total_real_settle / float(args.bankroll_nom)) if args.bankroll_nom else None

    raw_exp_sum = float(trades2["exp_pnl_raw"].fill_null(0.0).sum()) if "exp_pnl_raw" in trades2.columns else None
    suggested_scale = None
    if raw_exp_sum and abs(raw_exp_sum) > 1e-9:
        target_real = total_real_mtm if args.ev_mode == "mtm" else total_real_settle
        suggested_scale = target_real / raw_exp_sum

    summ = {
        "asof": args.asof,
        "start_date": args.start_date,
        "valid_days": args.valid_days,
        "horizon_secs": args.horizon_secs,
        "preoff_max_minutes": args.preoff_max,
        "commission": args.commission,
        "edge_thresh": args.edge_thresh,
        "ev_mode": args.ev_mode,
        "ev_cap": args.ev_cap,
        "ev_scale_used": args.ev_scale,
        "odds_min": args.odds_min,
        "odds_max": args.odds_max,
        "stake_mode": args.stake_mode,
        "kelly_cap": args.kelly_cap,
        "kelly_floor": args.kelly_floor,
        "bankroll_nom": args.bankroll_nom,
        "liquidity_levels": args.liquidity_levels,
        "enforce_liquidity_requested": bool(args.enforce_liquidity),
        "enforce_liquidity_effective": bool(effective_enforce),
        "require_book": bool(args.require_book),
        "min_fill_frac": args.min_fill_frac,
        "per_market_topk": args.per_market_topk,
        "per_market_budget": args.per_market_budget,
        "basket_sizing": args.basket_sizing,
        "exit_on_move_ticks": args.exit_on_move_ticks,
        "contra_mode": args.contra_mode,
        "contra_frac": args.contra_frac,
        "contra_hold_secs": args.contra_hold_secs,
        "contra_prob_thresh": args.contra_prob_thres
        if hasattr(args,"contra_prob_thres") else args.contra_prob_thresh,
        "contra_beta": args.contra_beta,
        "n_trades": int(trades2.height),
        "total_exp_profit": total_exp_profit,
        "avg_ev_per_1": avg_ev_scaled,
        "total_real_mtm_profit": total_real_mtm,
        "total_real_settle_profit": total_real_settle,
        "overall_roi_exp": roi_exp,
        "overall_roi_real_mtm": roi_real_mtm,
        "overall_roi_real_settle": roi_real_settle,
        "ev_scale_suggested": suggested_scale,
        "features_used": FEATURES_WITH_COUNTRY,
    }
    write_json(Path(args.output_dir) / f"summary_{args.asof}.json", summ)

    def _fmt(x):
        if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))):
            return "n/a"
        return f"{x:.6f}"
    print(f"[simulate] ROI (exp)        : {_fmt(roi_exp)}")
    print(f"[simulate] ROI (real MTM)   : {_fmt(roi_real_mtm)}")
    print(f"[simulate] ROI (settlement) : {_fmt(roi_real_settle)}")
    print(f"[simulate] Trades: {summ['n_trades']:,}  Avg EV/£1 (scaled): {avg_ev_scaled:.6f}
    if suggested_scale is not None:
        print(f"[simulate] Suggested --ev-scale ≈ {suggested_scale:.6g} (based on {args.ev_mode} PnL)")

def _to_df(records: List[Dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        records,
        schema={
            "publishTimeMs": pl.Int64,
            "marketId": pl.Utf8,
            "selectionId": pl.Int64,
            "ltp": pl.Float64,
            "p_now": pl.Float64,
            "p_pred": pl.Float64,
            "delta_pred": pl.Float64,
            "side": pl.Utf8,
            "is_contra": pl.Int8,
            "ev_per_1": pl.Float64,
            "stake_req": pl.Float64,
            "stake_filled": pl.Float64,
            "fill_frac": pl.Float64,
            "exec_odds": pl.Float64,
            "exp_pnl": pl.Float64,
            "exp_pnl_raw": pl.Float64,
            "stake_mode": pl.Utf8,
            "ts_exit_ms": pl.Int64,
        },
        orient="row",
    )

def _write_empty(outdir: Path, args):
    daily = pl.DataFrame({"day": [], "stake_mode": [], "n_trades": [], "exp_profit": [], "avg_ev": [],
                          "avg_stake_filled": [], "avg_fill_frac": [], "real_mtm_profit": [], "real_settle_profit": [],
                          "roi_exp": [], "roi_real_mtm": [], "roi_real_settle": [], "share_contra": []})
    daily.write_csv(outdir / f"daily_{args.asof}.csv")
    write_json(outdir / f"summary_{args.asof}.json", {
        "asof": args.asof, "n_trades": 0, "ev_scale_used": args.ev_scale,
        "per_market_topk": args.per_market_topk, "per_market_budget": args.per_market_budget,
        "basket_sizing": args.basket_sizing, "exit_on_move_ticks": args.exit_on_move_ticks,
        "contra_mode": args.contra_mode, "contra_frac": args.contra_frac,
        "contra_hold_secs": args.contra_hold_secs, "contra_prob_thresh": args.contra_prob_thresh,
        "enforce_liquidity_requested": bool(args.enforce_liquidity),
        "require_book": bool(args.require_book),
        "features_used": FEATURES_WITH_COUNTRY,
    })

if __name__ == "__main__":
    main()
