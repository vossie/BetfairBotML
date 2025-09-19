#!/usr/bin/env python3
"""
Temporal trainer with two heads (value + price-move), CUDA default, and no-leak labels.
- Validation window: [ASOF-1, ASOF]; Train: last --train-days ending at (ASOF-2)
- Robust value backtest knobs: RAW/CAL, (no) sum-to-one, overround comparator, per-market topK,
  flat/Kelly stakes, LTP range filter, ROI-by-odds table, Back/Lay/Both recommendations CSV,
  PM threshold sweep, side summary, and staking comparison (Flat £10 vs Kelly nominal £1000).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, date as _date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import xgboost as xgb

# Optional: probability calibration
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None

# Local modules
import features
from pm_labels import add_price_move_labels

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- utils -----------------------------
def _parse_date(s: str) -> _date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _fmt(d: _date) -> str:
    return d.strftime("%Y-%m-%d")

def _daterange_inclusive(start: _date, end: _date) -> List[str]:
    if end < start:
        start, end = end, start
    out: List[str] = []
    d = start
    while d <= end:
        out.append(_fmt(d))
        d += timedelta(days=1)
    return out

@dataclass
class SplitPlan:
    train_dates: List[str]
    valid_dates: List[str]

def build_split(asof: _date, train_days: int) -> SplitPlan:
    val1 = asof
    val0 = asof - timedelta(days=1)
    train_end = asof - timedelta(days=2)
    train_start = train_end - timedelta(days=train_days - 1)
    return SplitPlan(
        train_dates=_daterange_inclusive(train_start, train_end),
        valid_dates=[_fmt(val0), _fmt(val1)],
    )

# Polars dtype helper
try:
    from polars.datatypes import is_numeric as _isnum
except Exception:
    def _isnum(dt):
        return dt in {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

def _select_feature_cols(df: pl.DataFrame, label_cols: List[str]) -> List[str]:
    exclude = {
        "marketId", "selectionId", "ts", "ts_ms", "publishTimeMs",
        "runnerStatus", *label_cols,
        "ltpTick_fut", "ltp_fut", "ts_s_right", "ts_s_join", "future_delta_sec",
    }
    cols: List[str] = []
    for name, dtype in df.schema.items():
        lname = name.lower()
        if (
            name in exclude
            or "label" in lname
            or "target" in lname
            or lname.startswith("pm_")
            or lname.endswith("_fut")
            or lname.endswith("_join")
            or lname.endswith("_right")
            or "future" in lname
        ):
            continue
        if _isnum(dtype):
            cols.append(name)
    if not cols:
        raise RuntimeError("No numeric feature columns found after exclusions.")
    return cols

def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)

def _device_params(device: str) -> Tuple[Dict, str]:
    if device == "auto":
        try:
            import cupy as _cp  # noqa: F401
            device = "cuda"
        except Exception:
            device = "cpu"
    if device == "cuda":
        return {"device": "cuda", "tree_method": "hist"}, "Using GPU (CUDA)"
    else:
        return {"device": "cpu", "tree_method": "hist"}, "Using CPU"

# ----------------------------- value backtest helpers -----------------------------
def _group_indices(ids: np.ndarray):
    """Return list of index arrays, grouped by market id (stable order)."""
    order = np.argsort(ids, kind="mergesort")
    ids_sorted = ids[order]
    groups = []
    start = 0
    for i in range(1, len(ids_sorted) + 1):
        if i == len(ids_sorted) or ids_sorted[i] != ids_sorted[start]:
            groups.append(order[start:i])
            start = i
    return groups

def _overround_adjust(pim: np.ndarray, market_ids: np.ndarray) -> np.ndarray:
    out = pim.copy()
    for idxs in _group_indices(market_ids):
        s = out[idxs].sum()
        if s > 0:
            out[idxs] = out[idxs] / s
    return out

def _roi_by_odds_table(ltp: np.ndarray, y: np.ndarray, profit: np.ndarray) -> List[Dict[str, float]]:
    """
    Compute tiny ROI-by-odds table over taken trades.
    Returns list of dicts with: lo, hi, n, hit, roi.
    """
    bins = np.array([1.01, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 1000.0], dtype=float)
    out: List[Dict[str, float]] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (ltp >= lo) & (ltp < hi)
        n = int(m.sum())
        if n == 0:
            hit = float("nan"); roi = float("nan")
        else:
            hit = float(y[m].mean())
            roi = float(profit[m].sum() / max(1, n))
        out.append({"lo": lo, "hi": hi, "n": n, "hit": hit, "roi": roi})
    return out

# ----------------------------- value backtest (with recos & sides) -----------------------------
def backtest_value(
    df: pl.DataFrame,
    p_model: np.ndarray,
    commission: float,
    edge_thresh: float,
    do_sum_to_one: bool,
    market_prob_mode: str,    # "ltp" | "overround"
    per_market_topk: int,     # total across sides
    stake_mode: str,          # "flat" | "kelly"
    kelly_cap: float,
    ltp_min: float,
    ltp_max: float,
    side: str = "back",       # "back" | "lay" | "both"
) -> Dict[str, object]:
    """
    Select trades per market (top-K by edge) for back, lay, or both.
    Lay edge = p_market - p_model; Back edge = p_model - p_market.
    Profit model:
      Back: win => (odds-1)*(1-comm); lose => -1
      Lay:  lose => +1*(1-comm);       win  => -(odds-1)
    Stakes:
      - flat: stake=1.0 for all
      - kelly: supported for BACK only (capped). For LAY we keep flat.
    """
    assert side in {"back", "lay", "both"}

    pim = df["implied_prob"].to_numpy()
    mids = df["marketId"].to_numpy()
    y = df["winLabel"].to_numpy().astype(int)
    ltp = df["ltp"].to_numpy()

    finite = np.isfinite(p_model) & np.isfinite(pim) & np.isfinite(ltp)
    if finite.sum() == 0:
        return {"n_trades": 0, "roi": 0.0, "hit_rate": 0.0, "avg_edge": float("nan"),
                "roi_by_odds": [], "recos": [], "sel_arrays": {}}

    # base slice
    p = p_model[finite].copy()
    pm = pim[finite].copy()
    m = mids[finite]
    y = y[finite]
    l = ltp[finite]

    # LTP filter
    price_ok = (l >= ltp_min) & (l <= ltp_max)
    if not price_ok.any():
        return {"n_trades": 0, "roi": 0.0, "hit_rate": 0.0, "avg_edge": float("nan"),
                "roi_by_odds": [], "recos": [], "sel_arrays": {}}
    p, pm, m, y, l = p[price_ok], pm[price_ok], m[price_ok], y[price_ok], l[price_ok]

    # Comparator probability
    if market_prob_mode == "overround":
        pm = _overround_adjust(pm, m)

    # Optional normalize model probs within market
    if do_sum_to_one:
        for idxs in _group_indices(m):
            s = p[idxs].sum()
            if s > 0:
                p[idxs] = p[idxs] / s

    edge_back = p - pm           # > threshold -> back
    edge_lay  = pm - p           # > threshold -> lay

    # Per-market candidate list [(idx, edge, side_str)]
    take_mask = np.zeros_like(p, dtype=bool)
    take_side = np.empty_like(p, dtype=object)

    for idxs in _group_indices(m):
        cand = []

        if side in {"back", "both"}:
            cb = np.where(edge_back[idxs] > edge_thresh)[0]
            for ii in cb:
                cand.append((idxs[ii], float(edge_back[idxs][ii]), "back"))

        if side in {"lay", "both"}:
            cl = np.where(edge_lay[idxs] > edge_thresh)[0]
            for ii in cl:
                cand.append((idxs[ii], float(edge_lay[idxs][ii]), "lay"))

        if not cand:
            continue
        cand.sort(key=lambda t: t[1], reverse=True)
        for idx, _e, sd in cand[:per_market_topk]:
            take_mask[idx] = True
            take_side[idx] = sd

    if take_mask.sum() == 0:
        return {"n_trades": 0, "roi": 0.0, "hit_rate": 0.0, "avg_edge": float(np.nan),
                "roi_by_odds": [], "recos": [], "sel_arrays": {}}

    # Stakes
    stake = np.ones(int(take_mask.sum()), dtype=float)
    sel_idx = np.where(take_mask)[0]
    sel_side = take_side[take_mask]
    sel_p = p[take_mask]
    sel_l = l[take_mask]
    sel_y = y[take_mask]
    sel_pm = pm[take_mask]
    sel_edge_back = sel_p - sel_pm  # for logging

    if stake_mode == "kelly":
        # apply only for BACK selections; LAY remains flat
        for i in range(len(stake)):
            if sel_side[i] == "back":
                dec = sel_l[i]
                b = (dec - 1.0) * (1.0 - commission)
                q = 1.0 - sel_p[i]
                f = (b * sel_p[i] - q) / (b if b != 0 else 1e-12)
                f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, kelly_cap)
                stake[i] = float(f)
            else:
                stake[i] = 1.0

    # Profit per bet (per-unit stake)
    unit_ret = np.zeros_like(stake, dtype=float)
    for i in range(len(stake)):
        dec = sel_l[i]
        if sel_side[i] == "back":
            unit_ret[i] = (dec - 1.0) * (1.0 - commission) if sel_y[i] == 1 else -1.0
        else:  # lay
            unit_ret[i] = (1.0 - commission) if sel_y[i] == 0 else -(dec - 1.0)

    # Applied stakes profit (matches ROI printed here)
    profit = unit_ret * stake

    n = len(stake)
    roi = float(profit.sum() / (n if stake_mode == "flat" else max(1.0, stake.sum())))
    hit_rate = float((sel_y == 1).mean())
    avg_edge = float(np.mean(sel_edge_back[sel_side == "back"])) if np.any(sel_side == "back") else float("nan")

    # ROI-by-odds table (use decimal odds = sel_l)
    roi_tbl = _roi_by_odds_table(sel_l, sel_y, profit)

    # Build recommendation rows (absolute indices back to df)
    base_mask = np.zeros_like(p_model, dtype=bool); base_mask[finite] = True
    base_idx = np.where(base_mask)[0]
    l_finite = ltp[finite]
    price_ok_all = (l_finite >= ltp_min) & (l_finite <= ltp_max)
    price_idx = base_idx[price_ok_all]
    abs_idx = price_idx[sel_idx]

    recos = []
    mkt = df["marketId"].to_numpy()
    selid = df["selectionId"].to_numpy()
    for i in range(n):
        ai = int(abs_idx[i])
        recos.append({
            "marketId": str(mkt[ai]),
            "selectionId": int(selid[ai]),
            "ltp": float(sel_l[i]),
            "side": str(sel_side[i]),
            "stake": float(stake[i]),
            "p_model": float(sel_p[i]),
            "p_market": float(sel_pm[i]),
            "edge_back": float(sel_edge_back[i]),  # positive means model>market
        })

    sel_arrays = {
        "odds": sel_l,
        "y": sel_y.astype(int),
        "side": np.array(sel_side, dtype=object),
        "p": sel_p,
        "unit_return": unit_ret,     # per £1 stake outcome (+/-)
        "commission": float(commission),
    }

    return {
        "n_trades": n,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "roi_by_odds": roi_tbl,
        "recos": recos,
        "sel_arrays": sel_arrays,
    }

# ----------------------------- metrics & PM sweep -----------------------------
def _metrics_binary(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, p))
    except Exception:
        order = np.argsort(p)
        ranks = np.empty_like(order); ranks[order] = np.arange(len(p))
        pos = y_true == 1
        n_pos = int(pos.sum()); n_neg = len(p) - n_pos
        auc = 0.5 if (n_pos == 0 or n_neg == 0) else (float(ranks[pos].sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {"logloss": float(logloss), "auc": float(auc)}

def _pm_threshold_sweep(p: np.ndarray, y: np.ndarray, fut_ticks: np.ndarray) -> None:
    ths = np.arange(0.50, 0.95, 0.05)
    pos = (y == 1)
    n_pos = int(pos.sum())
    print("\n[PM threshold sweep]")
    print("  th   N_sel  precision  recall   F1     avg_move_ticks")
    for th in ths:
        sel = (p >= th)
        n_sel = int(sel.sum())
        if n_sel == 0 or n_pos == 0:
            prec = rec = f1 = 0.0
            avg_mv = float("nan")
        else:
            tp = int((sel & pos).sum())
            prec = tp / n_sel
            rec  = tp / n_pos
            f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
            avg_mv = float(np.mean(fut_ticks[sel])) if n_sel else float("nan")
        print(f"  {th:0.2f}  {n_sel:6d}   {prec:8.3f}  {rec:6.3f}  {f1:6.3f}   {avg_mv:14.2f}")

# ----------------------------- FS guards -----------------------------
def _has_snapshot_day(curated_root: str, sport: str, day: str) -> bool:
    p = Path(curated_root) / "orderbook_snapshots_5s" / f"sport={sport}" / f"date={day}"
    return p.exists()

def _filter_dates_with_data(curated_root: str, sport: str, dates: List[str]) -> List[str]:
    ok, missing = [], []
    for d in dates:
        (ok if _has_snapshot_day(curated_root, sport, d) else missing).append(d)
    if missing:
        print(f"WARN: Skipping {len(missing)} day(s) with no snapshots: {', '.join(missing)}")
    return ok

# ----------------------------- training -----------------------------
def train_temporal(
    curated_root: str,
    sport: str,
    asof_date: str,
    train_days: int,
    preoff_minutes: int,
    label_col: str,
    downsample_secs: Optional[int],
    device: str,
    n_estimators: int,
    learning_rate: float,
    early_stopping_rounds: int,
    commission: float,
    edge_thresh: float,
    calibrate: bool,
    pm_horizon_secs: int,
    pm_tick_threshold: int,
    pm_slack_secs: int,
    edge_prob: str,
    no_sum_to_one: bool,
    market_prob_mode: str,
    per_market_topk: int,
    stake_mode: str,
    kelly_cap: float,
    ltp_min: float,
    ltp_max: float,
    side: str,
    bankroll_nom: float,
) -> None:
    asof = _parse_date(asof_date)
    plan = build_split(asof, train_days)

    print("=== Temporal split ===")
    print(f"  Train: {plan.train_dates[0]} .. {plan.train_dates[-1]}  ({len(plan.train_dates)} days)")
    print(f"  Valid: {plan.valid_dates[0]} .. {plan.valid_dates[-1]}  (2 days)\n")

    dev_params, banner = _device_params(device)
    print(banner)

    def _build(dates: List[str]) -> pl.DataFrame:
        dates_ok = _filter_dates_with_data(curated_root, sport, dates)
        if not dates_ok:
            raise FileNotFoundError(
                f"No available data days among: {', '.join(dates)} under {curated_root}/orderbook_snapshots_5s/sport={sport}"
            )
        df, raw = features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=dates_ok,
            preoff_minutes=preoff_minutes,
            batch_markets=100,
            downsample_secs=downsample_secs,
        )
        print(f"Built features for {dates_ok[0]}..{dates_ok[-1]} → rows={df.height} (~{raw} scanned)")
        df = df.filter(pl.col(label_col).is_not_null())
        df = add_price_move_labels(
            df,
            horizon_secs=pm_horizon_secs,
            tick_threshold=pm_tick_threshold,
            slack_secs=pm_slack_secs,
        )
        return df

    df_train = _build(plan.train_dates)
    df_valid = _build(plan.valid_dates)

    # -------- value head (win prob) --------
    feat_cols = _select_feature_cols(df_train, [label_col, "pm_up", "pm_delta_ticks"])
    Xtr = _to_numpy(df_train, feat_cols)
    ytr = df_train[label_col].to_numpy().astype(np.float32)
    Xva = _to_numpy(df_valid, feat_cols)
    yva = df_valid[label_col].to_numpy().astype(np.float32)

    params_bin = {
        **dev_params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": learning_rate,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    }

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    booster_value = xgb.train(
        params=params_bin,
        dtrain=dtr,
        num_boost_round=n_estimators,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        verbose_eval=False,
    )

    p_valid_val = booster_value.predict(dva)
    p_cal_val = p_valid_val
    if calibrate and IsotonicRegression is not None:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_valid_val, yva)
            p_cal_val = iso.transform(p_valid_val)
        except Exception as e:
            print(f"WARN: Isotonic calibration failed (value head): {e}; using raw.")

    # choose RAW/CAL for edges
    use_p = p_cal_val if edge_prob == "cal" else p_valid_val

    # Debug: edge distribution on finite rows
    pim = df_valid["implied_prob"].to_numpy()
    mask = np.isfinite(use_p) & np.isfinite(pim)
    edge_dbg = use_p[mask] - pim[mask]
    for thr in [0.0, 0.003, 0.005, 0.01, 0.02]:
        print(f"edge>{thr:.3f}: n={(edge_dbg > thr).sum()} (of {mask.sum()} finite)")

    metrics_val = _metrics_binary(yva, p_valid_val)
    print(f"\n[Value head: winLabel] logloss={metrics_val['logloss']:.4f} auc={metrics_val['auc']:.3f}  n={len(yva)}")

    # ---- Backtest value head ----
    pnl = backtest_value(
        df=df_valid,
        p_model=use_p,
        commission=commission,
        edge_thresh=edge_thresh,
        do_sum_to_one=(not no_sum_to_one),
        market_prob_mode=market_prob_mode,
        per_market_topk=per_market_topk,
        stake_mode=stake_mode,
        kelly_cap=kelly_cap,
        ltp_min=ltp_min,
        ltp_max=ltp_max,
        side=side,
    )
    print("[Backtest @ validation — value]")
    print(f"  n_trades={pnl['n_trades']}  roi={pnl['roi']:.4f}  hit_rate={pnl['hit_rate']:.3f}  avg_edge={pnl['avg_edge']}")

    # ROI-by-odds table
    roi_tbl = pnl.get("roi_by_odds", [])
    if roi_tbl:
        print("\n[Value head — ROI by decimal odds bucket]")
        print("  bucket         n    hit    roi")
        for row in roi_tbl:
            lo, hi, n, hit, roi = row["lo"], row["hi"], row["n"], row["hit"], row["roi"]
            print(f"  [{lo:4.2f},{hi:6.2f})  {n:5d}  {hit:5.3f}  {roi:6.3f}")

    # --- side summary & recos CSV ---
    recs = pnl.get("recos", [])
    if recs:
        rec_df_all = pl.DataFrame(recs)
        by_side = rec_df_all.group_by("side").agg([
            pl.len().alias("n"),
            pl.mean("edge_back").alias("avg_edge"),
            pl.mean("stake").alias("avg_stake_frac"),
        ])
        print("\n[Backtest — side summary]")
        for row in by_side.iter_rows(named=True):
            stake_frac = float(row["avg_stake_frac"])
            stake_gbp = stake_frac * bankroll_nom
            print(
                f"  side={row['side']:>4}  n={row['n']:>5}  "
                f"avg_edge={row['avg_edge']:.4f}  "
                f"avg_stake_frac={stake_frac:.4f}  avg_stake_gbp=£{stake_gbp:.2f}"
            )

        rec_df = rec_df_all.select([
            "marketId","selectionId","ltp","side","stake","p_model","p_market","edge_back"
        ])
        rec_file = OUTPUT_DIR / f"edge_recos_valid_{asof_date}_T{train_days}.csv"
        rec_df.write_csv(str(rec_file))
        print(f"\nSaved recommendations → {rec_file}")

    # --- Staking comparison: Flat £10 vs Kelly (nominal £1000, use same cap as backtest) ---
    sel = pnl.get("sel_arrays", {})
    if sel:
        odds = np.asarray(sel["odds"])
        ywin = np.asarray(sel["y"]).astype(int)
        sides = np.asarray(sel["side"], dtype=object)
        pmod = np.asarray(sel["p"])
        comm = float(sel.get("commission", 0.0))

        unit_ret = np.zeros_like(odds, dtype=float)
        for i in range(len(odds)):
            dec = odds[i]
            if sides[i] == "back":
                unit_ret[i] = (dec - 1.0) * (1.0 - comm) if ywin[i] == 1 else -1.0
            else:
                unit_ret[i] = (1.0 - comm) if ywin[i] == 0 else -(dec - 1.0)

        # Flat £10
        flat_stake = 10.0
        flat_profit = float(np.sum(unit_ret * flat_stake))
        flat_n = len(unit_ret)
        flat_gross = flat_n * flat_stake
        flat_roi = flat_profit / flat_gross if flat_gross > 0 else float("nan")

        # Kelly (nominal £1000), same cap as backtest

        kelly_fraction = 1.0   # set <1 for fractional Kelly
        stake_floor = 0.0      # e.g. 5.0 to avoid tiny stakes
        cap = float(kelly_cap)

        kelly_stakes = np.zeros_like(unit_ret)
        for i in range(len(odds)):
            dec = odds[i]
            if sides[i] == "back":
                b = (dec - 1.0) * (1.0 - comm)
                q = 1.0 - pmod[i]
                f_raw = (b * pmod[i] - q) / (b if b != 0 else 1e-12)
                f = np.clip(np.nan_to_num(f_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, cap)
                f *= kelly_fraction
                kelly_stakes[i] = max(stake_floor, f * bankroll_nom)
            else:
                kelly_stakes[i] = flat_stake  # simple comparison for lays

        kelly_profit = float(np.sum(unit_ret * kelly_stakes))
        kelly_gross = float(np.sum(kelly_stakes))
        kelly_avg_stake = float(np.mean(kelly_stakes)) if len(kelly_stakes) else 0.0
        kelly_roi = kelly_profit / kelly_gross if kelly_gross > 0 else float("nan")

        print("\n[Staking comparison]")
        print(f"  Flat £10 stake    → trades={flat_n}  staked=£{flat_gross:.2f}  profit=£{flat_profit:.2f}  roi={flat_roi:.3f}")
        print(f"  Kelly (nom £1000) → trades={flat_n}  staked=£{kelly_gross:.2f}  profit=£{kelly_profit:.2f}  roi={kelly_roi:.3f}  avg_stake=£{kelly_avg_stake:.2f}")

    # -------- price-move head (short horizon) --------
    ytr_pm = df_train["pm_up"].to_numpy().astype(np.float32)
    yva_pm = df_valid["pm_up"].to_numpy().astype(np.float32)

    dtr_pm = xgb.DMatrix(Xtr, label=ytr_pm)
    dva_pm = xgb.DMatrix(Xva, label=yva_pm)

    booster_pm = xgb.train(
        params=params_bin,
        dtrain=dtr_pm,
        num_boost_round=max(300, n_estimators // 4),
        evals=[(dtr_pm, "train"), (dva_pm, "valid")],
        early_stopping_rounds=min(50, early_stopping_rounds) if early_stopping_rounds > 0 else None,
        verbose_eval=False,
    )

    p_valid_pm = booster_pm.predict(dva_pm)
    metrics_pm = _metrics_binary(yva_pm, p_valid_pm)

    preds = p_valid_pm >= 0.5
    acc = float((preds.astype(np.int8) == yva_pm.astype(np.int8)).mean())
    taken = int(preds.sum())
    avg_move_ticks = float(np.mean(df_valid["pm_delta_ticks"].to_numpy()[preds])) if taken else 0.0
    hit_up = float(yva_pm[preds].mean()) if taken else 0.0

    print(f"\n[Price-move head: horizon={pm_horizon_secs}s, threshold={pm_tick_threshold}t]")
    print(f"  logloss={metrics_pm['logloss']:.4f} auc={metrics_pm['auc']:.3f}  acc@0.5={acc:.3f}")
    print(f"  taken_signals={taken}  hit_rate={hit_up:.3f}  avg_future_move_ticks={avg_move_ticks:.2f}")

    _pm_threshold_sweep(
        p=p_valid_pm,
        y=yva_pm.astype(int),
        fut_ticks=df_valid["pm_delta_ticks"].to_numpy()
    )

    # Save artifacts
    value_name = f"edge_value_xgb_{preoff_minutes}m_{asof_date}_T{train_days}.json"
    pm_name = f"edge_price_xgb_{pm_horizon_secs}s_{preoff_minutes}m_{asof_date}_T{train_days}.json"
    out_value = OUTPUT_DIR / value_name
    out_pm = OUTPUT_DIR / pm_name
    booster_value.save_model(str(out_value))
    booster_pm.save_model(str(out_pm))
    print(f"Saved models →\n  {out_value}\n  {out_pm}")

    rep = pl.DataFrame({
        "marketId": df_valid["marketId"],
        "selectionId": df_valid["selectionId"],
        "tto_minutes": df_valid["tto_minutes"],
        "ltp": df_valid["ltp"],
        "implied_prob": df_valid["implied_prob"],
        "y_win": yva,
        "p_win_raw": p_valid_val,
        "p_win_cal": p_cal_val,
        "y_pm_up": yva_pm,
        "p_pm_up": p_valid_pm,
        "pm_delta_ticks": df_valid["pm_delta_ticks"],
    })
    rep_file = OUTPUT_DIR / f"edge_valid_both_{asof_date}_T{train_days}.csv"
    rep.write_csv(str(rep_file))
    print(f"Saved validation detail → {rep_file}")

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description=(
        "Temporal split with 2-day validation, value+price heads, calibration, robust value backtest, "
        "LTP filtering, PM threshold sweep, ROI-by-odds, side summary, staking comparison, and recos CSV."
    ))
    # Data
    ap.add_argument("--curated", required=True, help="/mnt/nvme/betfair-curated or s3://bucket")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--asof", required=True, help="Validation end date (YYYY-MM-DD)")
    ap.add_argument("--train-days", type=int, default=5, help="Number of training days ending at asof-2")
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--downsample-secs", type=int, default=5)

    # Model
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    # Trading eval (value)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--edge-thresh", type=float, default=0.02, help="Back if p_model - p_market > thresh")
    ap.add_argument("--no-calibrate", action="store_true", help="Disable isotonic calibration for value head")

    # Price-move head
    ap.add_argument("--pm-horizon-secs", type=int, default=60, help="Short-horizon seconds for price-move label")
    ap.add_argument("--pm-tick-threshold", type=int, default=1, help="Minimum future move in ticks to label as 1")
    ap.add_argument("--pm-slack-secs", type=int, default=3, help="Tolerance slack around horizon for asof join")

    # Backtest controls
    ap.add_argument("--edge-prob", choices=["raw", "cal"], default="cal",
                    help="Use raw or calibrated probabilities for edge calc")
    ap.add_argument("--no-sum-to-one", action="store_true",
                    help="Disable per-market sum-to-one normalization in backtest")
    ap.add_argument("--market-prob", choices=["ltp", "overround"], default="overround",
                    help="Comparator prob: raw LTP implied or overround-normalized within market")
    ap.add_argument("--per-market-topk", type=int, default=1,
                    help="Take top-K edges per market that exceed threshold (total across sides)")
    ap.add_argument("--stake", choices=["flat", "kelly"], default="flat",
                    help="Stake model for backtest")
    ap.add_argument("--kelly-cap", type=float, default=0.05,
                    help="Max Kelly fraction (if --stake kelly)")
    ap.add_argument("--ltp-min", type=float, default=1.01,
                    help="Minimum decimal odds to include in backtest")
    ap.add_argument("--ltp-max", type=float, default=1000.0,
                    help="Maximum decimal odds to include in backtest")
    ap.add_argument("--side", choices=["back","lay","both"], default="back",
                    help="Trade side selection policy for backtest and recos")
    ap.add_argument("--bankroll-nom", type=float, default=1000.0,
                    help="Nominal bankroll used for Kelly reporting/comparison & stake prints")

    args = ap.parse_args()

    train_temporal(
        curated_root=args.curated,
        sport=args.sport,
        asof_date=args.asof,
        train_days=args.train_days,
        preoff_minutes=args.preoff_mins,
        label_col=args.label_col,
        downsample_secs=(args.downsample_secs or None),
        device=args.device,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        commission=args.commission,
        edge_thresh=args.edge_thresh,
        calibrate=(not args.no_calibrate),
        pm_horizon_secs=args.pm_horizon_secs,
        pm_tick_threshold=args.pm_tick_threshold,
        pm_slack_secs=args.pm_slack_secs,
        edge_prob=args.edge_prob,
        no_sum_to_one=args.no_sum_to_one,
        market_prob_mode=args.market_prob,
        per_market_topk=args.per_market_topk,
        stake_mode=args.stake,
        kelly_cap=args.kelly_cap,
        ltp_min=args.ltp_min,
        ltp_max=args.ltp_max,
        side=args.side,
        bankroll_nom=args.bankroll_nom,
    )

if __name__ == "__main__":
    main()
