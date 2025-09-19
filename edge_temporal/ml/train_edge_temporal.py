#!/usr/bin/env python3
"""
Temporal trainer (value + price-move), CUDA default, robust logging, and practical backtest.

Validation window: [ASOF-1, ASOF]
Training window:   last --train-days ending at (ASOF-2)

Features:
- Value head (win probability) + optional isotonic calibration
- Price-move head (short-horizon) with no-leak labels
- PM cutoff gate (--pm-cutoff) + PM threshold sweep
- Backtest with edge-threshold, per-market topK, odds band, market prob mode,
  sum-to-one normalization, back/lay or both sides, kelly/flat staking, and
  ROI-by-odds bucket table. Saves recommendations CSV.

Outputs in project-level output/ dir.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date as _date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import xgboost as xgb

# Optional: calibration
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None

# Local modules
import features
from pm_labels import add_price_move_labels

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- date utils -----------------------------
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


# ----------------------------- device -----------------------------
def _device_params(device: str) -> Tuple[Dict, str]:
    if device == "auto":
        try:
            import cupy as _  # noqa: F401
            device = "cuda"
        except Exception:
            device = "cpu"
    if device == "cuda":
        return {"device": "cuda", "tree_method": "hist", "predictor": "gpu_predictor"}, "Using GPU (CUDA)"
    return {"device": "cpu", "tree_method": "hist"}, "Using CPU"


# ----------------------------- features -----------------------------
def _select_feature_cols(df: pl.DataFrame, label_cols: List[str]) -> List[str]:
    exclude = {
        "marketId", "selectionId", "ts", "ts_ms", "publishTimeMs",
        "runnerStatus", *label_cols,
        # explicit futures / matching columns
        "ltpTick_fut", "ltp_fut", "ts_s_right", "ts_s_join", "future_delta_sec",
        # probabilities that can be targets or derived
        "p_pm_up",
    }
    cols: List[str] = []
    for name, dt in df.schema.items():
        lname = name.lower()
        if (
            name in exclude
            or "label" in lname
            or "target" in lname
            or lname.endswith("_fut")
            or lname.endswith("_join")
            or lname.endswith("_right")
            or "future" in lname
        ):
            continue
        if dt in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64}:
            cols.append(name)
    if not cols:
        raise RuntimeError("No numeric feature columns found after exclusions.")
    return cols


def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


# ----------------------------- market probs & odds -----------------------------
def _compute_market_prob(df: pl.DataFrame, mode: str) -> np.ndarray:
    """
    mode: 'ltp' -> 1/odds; 'overround' -> per-market sum-to-one normalization of 1/odds
    Expects df['ltp'] and df['marketId'].
    """
    odds = df["ltp"].to_numpy()
    inv = np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, np.nan)
    if mode == "ltp":
        return inv
    # overround normalize within each marketId
    mids = df["marketId"].to_numpy()
    out = inv.copy()
    for m in np.unique(mids):
        mask = (mids == m) & np.isfinite(out)
        s = out[mask].sum()
        if s > 0:
            out[mask] = out[mask] / s
    return out


# ----------------------------- value backtest -----------------------------
def _back_profit_back(odds: np.ndarray, win: np.ndarray, stake: np.ndarray, commission: float) -> np.ndarray:
    # Win: (odds-1)*stake*(1-commission); Lose: -stake
    gross = (np.clip(odds - 1.0, 0.0, None)) * stake
    win_pnl = gross * (1.0 - commission)
    lose_pnl = -stake
    return np.where(win == 1, win_pnl, lose_pnl)


def _lay_profit_lay(odds: np.ndarray, win: np.ndarray, stake: np.ndarray, commission: float) -> np.ndarray:
    """
    Lay with 'stake' as the lay stake (backer's stake). Liability L = (odds-1) * stake.
    If selection loses (win=0): you win stake * (1-commission). If wins: you lose liability.
    """
    liability = (np.clip(odds - 1.0, 0.0, None)) * stake
    win_when_loses = stake * (1.0 - commission)
    lose_when_wins = -liability
    return np.where(win == 0, win_when_loses, lose_when_wins)


def _per_market_topk_mask(market_ids: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(score, dtype=bool)
    keep = np.zeros_like(score, dtype=bool)
    for m in np.unique(market_ids):
        mask = market_ids == m
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        # top-k by score
        order = np.argsort(score[idx])[::-1]
        ksel = idx[order[:k]]
        keep[ksel] = True
    return keep


def backtest_value(
    df: pl.DataFrame,
    p_model: np.ndarray,
    commission: float,
    edge_thresh: float,
    do_sum_to_one: bool,
    market_prob_mode: str,
    per_market_topk: int,
    stake_mode: str,          # 'flat' | 'kelly'
    kelly_cap: float,
    ltp_min: float,
    ltp_max: float,
    side: str,                # 'back' | 'lay' | 'both'
    bankroll_nom: float,
    kelly_fraction: float = 1.0,
    stake_floor: float = 0.0,
) -> Dict[str, object]:
    """
    Returns metrics dict with: n_trades, roi, hit_rate, avg_edge, recos (list[dict]),
    stakes (list[float, £]), sides (list[str]), odds (list[float]), unit_ret(np.ndarray)
    """
    # Odds filter
    ltp = df["ltp"].to_numpy()
    odds_mask = np.isfinite(ltp) & (ltp >= ltp_min) & (ltp <= ltp_max)

    # Market probability (baseline)
    p_market_base = _compute_market_prob(df, market_prob_mode)
    if do_sum_to_one:
        # optional sum-to-one of model probabilities per market
        mids = df["marketId"].to_numpy()
        p_norm = p_model.copy()
        for m in np.unique(mids):
            mask = mids == m
            s = p_norm[mask].sum()
            if s > 0:
                p_norm[mask] /= s
        p_used = p_norm
    else:
        p_used = p_model

    # Edges for both sides
    edge_back = p_used - p_market_base              # positive -> back value
    edge_lay = p_market_base - p_used              # positive -> lay value

    # Base mask: finite, odds band
    finite = np.isfinite(edge_back) & odds_mask & np.isfinite(df["winLabel"].to_numpy())
    if finite.sum() == 0:
        return {"n_trades": 0, "roi": 0.0, "hit_rate": 0.0, "avg_edge": float("nan"), "recos": []}

    mids = df["marketId"].to_numpy()[finite]
    selids = df["selectionId"].to_numpy()[finite]
    odds_f = ltp[finite]
    y = df["winLabel"].to_numpy().astype(int)[finite]

    eb = edge_back[finite]
    el = edge_lay[finite]

    # Select side(s)
    take_back = eb > edge_thresh
    take_lay = el > edge_thresh

    if side == "back":
        which = take_back
        side_arr = np.where(which, "back", "skip")
        score = eb
    elif side == "lay":
        which = take_lay
        side_arr = np.where(which, "lay", "skip")
        score = el
    else:  # both
        # Build combined candidates: pick better of the two per runner
        both_candidates = np.stack([eb, el], axis=1)
        side_choice = np.where(eb >= el, "back", "lay")
        best_edge = np.max(both_candidates, axis=1)
        which = best_edge > edge_thresh
        side_arr = np.where(which, side_choice, "skip")
        score = best_edge

    # Filter only selected rows
    sel_mask = which
    if sel_mask.sum() == 0:
        return {"n_trades": 0, "roi": 0.0, "hit_rate": 0.0, "avg_edge": float(np.nan), "recos": []}

    mids_s = mids[sel_mask]
    selids_s = selids[sel_mask]
    odds_s = odds_f[sel_mask]
    y_s = y[sel_mask]
    side_s = side_arr[sel_mask]
    edge_s = score[sel_mask]
    p_used_s = p_used[finite][sel_mask]
    p_mkt_s = p_market_base[finite][sel_mask]

    # Per-market topK across sides (use edge score)
    keep = _per_market_topk_mask(mids_s, edge_s, per_market_topk)
    mids_s, selids_s, odds_s, y_s, side_s, edge_s, p_used_s, p_mkt_s = \
        mids_s[keep], selids_s[keep], odds_s[keep], y_s[keep], side_s[keep], edge_s[keep], p_used_s[keep], p_mkt_s[keep]

    n = len(odds_s)
    if n == 0:
        return {"n_trades": 0, "roi": 0.0, "hit_rate": 0.0, "avg_edge": float(np.nan), "recos": []}

    # Staking
    stake = np.ones(n, dtype=float)  # flat baseline £1
    if stake_mode == "kelly":
        # Kelly fraction for BACK only; for LAY we use a simple flat £1 unless you want Kelly for lay too
        for i in range(n):
            if side_s[i] == "back":
                dec = odds_s[i]
                b = (dec - 1.0) * (1.0 - commission)
                q = 1.0 - p_used_s[i]
                denom = b if b != 0 else 1e-12
                f_raw = (b * p_used_s[i] - q) / denom
                f = np.clip(np.nan_to_num(f_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, kelly_cap)
                f *= kelly_fraction
                stake[i] = max(stake_floor, f * bankroll_nom)
            else:
                # Simple policy: flat £1 for lay (you can later add a Kelly-for-lay variant)
                stake[i] = 1.0

    # Profit per trade
    if n > 0:
        unit_ret_back = _back_profit_back(odds_s, y_s, np.ones_like(stake), commission)
        unit_ret_lay = _lay_profit_lay(odds_s, y_s, np.ones_like(stake), commission)
        unit_ret = np.where(side_s == "back", unit_ret_back, unit_ret_lay)
        pnl = unit_ret * stake
    else:
        unit_ret = np.array([])
        pnl = np.array([])

    total_profit = float(pnl.sum())
    total_staked = float(stake.sum()) if stake_mode == "kelly" else float(n)  # flat £1 baseline
    roi = float(total_profit / max(1.0, total_staked))
    hit_rate = float((y_s == 1).mean()) if n else 0.0

    # Recos payload (save both fraction proxy and £)
    recos: List[Dict] = []
    # Approximate stake fraction (for back only) for CSV transparency
    stake_frac_proxy = []
    for i in range(n):
        frac = 0.0
        if stake_mode == "kelly" and side_s[i] == "back":
            dec = odds_s[i]
            b = (dec - 1.0) * (1.0 - commission)
            q = 1.0 - p_used_s[i]
            denom = b if b != 0 else 1e-12
            f_raw = (b * p_used_s[i] - q) / denom
            frac = float(np.clip(np.nan_to_num(f_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, kelly_cap) * kelly_fraction)
        stake_frac_proxy.append(frac)

        recos.append({
            "marketId": mids_s[i],
            "selectionId": selids_s[i],
            "ltp": odds_s[i],
            "side": side_s[i],
            "stake_gbp": float(stake[i]),
            "stake_frac": frac,
            "p_model": float(p_used_s[i]),
            "p_market": float(p_mkt_s[i]),
            "edge_back": float(edge_s[i]) if side_s[i] == "back" else float(-edge_s[i]),  # for lay, negative of back edge
        })

    return {
        "n_trades": int(n),
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge": float(np.mean(edge_s)) if n else float("nan"),
        "recos": recos,
        "stakes": stake.tolist(),
        "sides": side_s.tolist(),
        "odds": odds_s.tolist(),
        "unit_ret": unit_ret,  # per-£1 return vector
    }


# ----------------------------- PM gate helper -----------------------------
def apply_pm_cutoff(trades_df: pl.DataFrame, pm_cutoff: float | None) -> pl.DataFrame:
    if not pm_cutoff or pm_cutoff <= 0:
        return trades_df
    if "p_pm_up" not in trades_df.columns:
        print("[PM gate] WARNING: 'p_pm_up' not in frame → skipping cutoff")
        return trades_df
    before = trades_df.height
    out = trades_df.filter(pl.col("p_pm_up") >= pm_cutoff)
    after = out.height
    print(f"[PM gate] Applied cutoff {pm_cutoff:.2f} → kept {after}/{before} rows")
    return out


# ----------------------------- metrics -----------------------------
def _metrics_binary(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, p))
    except Exception:
        # fallback AUC
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(p))
        pos = y_true == 1
        n_pos = int(pos.sum())
        n_neg = len(p) - n_pos
        auc = 0.5 if (n_pos == 0 or n_neg == 0) else (float(ranks[pos].sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {"logloss": float(logloss), "auc": float(auc)}


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


# ----------------------------- trainer -----------------------------
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
    pm_cutoff: float,
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
                f"No available data among: {', '.join(dates)} under {curated_root}/orderbook_snapshots_5s/sport={sport}"
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
        df = add_price_move_labels(df, horizon_secs=pm_horizon_secs, tick_threshold=pm_tick_threshold, slack_secs=pm_slack_secs)
        return df

    # Build
    df_train = _build(plan.train_dates)
    df_valid = _build(plan.valid_dates)

    # -------- value head --------
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

    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    print(f"[XGB value] elapsed={t1-t0:.2f}s  best_iter={booster_value.best_iteration}  best_score={booster_value.best_score}")

    p_valid_val = booster_value.predict(dva)
    p_cal_val = p_valid_val
    if calibrate and IsotonicRegression is not None:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_valid_val, yva)
            p_cal_val = iso.transform(p_valid_val)
        except Exception as e:
            print(f"WARN: Isotonic calibration failed (value): {e}; using raw.")
    metrics_val = _metrics_binary(yva, p_valid_val)
    print(f"\n[Value head: {label_col}] logloss={metrics_val['logloss']:.4f} auc={metrics_val['auc']:.3f}  n={len(yva)}")

    # -------- price-move head --------
    ytr_pm = df_train["pm_up"].to_numpy().astype(np.float32)
    yva_pm = df_valid["pm_up"].to_numpy().astype(np.float32)
    dtr_pm = xgb.DMatrix(Xtr, label=ytr_pm)
    dva_pm = xgb.DMatrix(Xva, label=yva_pm)
    t2 = time.perf_counter()
    booster_pm = xgb.train(
        params=params_bin,
        dtrain=dtr_pm,
        num_boost_round=max(300, n_estimators // 4),
        evals=[(dtr_pm, "train"), (dva_pm, "valid")],
        early_stopping_rounds=min(50, early_stopping_rounds) if early_stopping_rounds > 0 else None,
        verbose_eval=False,
    )
    t3 = time.perf_counter()
    print(f"[XGB pm]    elapsed={t3-t2:.2f}s  best_iter={booster_pm.best_iteration}  best_score={booster_pm.best_score}")

    p_valid_pm = booster_pm.predict(dva_pm)
    metrics_pm = _metrics_binary(yva_pm, p_valid_pm)
    preds_pm = p_valid_pm >= 0.5
    acc_pm = float((preds_pm.astype(np.int8) == yva_pm.astype(np.int8)).mean())
    taken_pm = int(preds_pm.sum())
    avg_move_ticks = float(np.mean(df_valid["pm_delta_ticks"].to_numpy()[preds_pm])) if taken_pm else 0.0
    print(f"\n[Price-move head: horizon={pm_horizon_secs}s, threshold={pm_tick_threshold}t]")
    print(f"  logloss={metrics_pm['logloss']:.4f} auc={metrics_pm['auc']:.3f}  acc@0.5={acc_pm:.3f}")
    print(f"  taken_signals={taken_pm}  hit_rate={float(yva_pm[preds_pm].mean()) if taken_pm else 0.0:.3f}  avg_future_move_ticks={avg_move_ticks:.2f}")

    # PM threshold sweep (diagnostic)
    print("\n[PM threshold sweep]")
    print("  th   N_sel  precision  recall   F1     avg_move_ticks")
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        sel = p_valid_pm >= th
        n_sel = int(sel.sum())
        if n_sel == 0:
            print(f"  {th:0.2f}       0      0.000   0.000   0.000              nan")
            continue
        prec = float(yva_pm[sel].mean())
        rec = float(yva_pm[sel].sum() / max(1, yva_pm.sum()))
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        avg_ticks = float(np.mean(df_valid["pm_delta_ticks"].to_numpy()[sel]))
        print(f"  {th:0.2f} {n_sel:6d}      {prec:0.3f}   {rec:0.3f}   {f1:0.3f}           {avg_ticks:0.2f}")

    # -------- backtest / recommendations on validation --------
    # Prepare frame for PM gate (we keep it simple: attach p_pm_up to df_valid)
    df_valid_bt = df_valid.with_columns(
        pl.Series("p_pm_up", p_valid_pm)
    )
    # PM cutoff gate (filter the rows before backtest)
    df_valid_bt = apply_pm_cutoff(df_valid_bt, pm_cutoff=pm_cutoff)

    # Choose which probs feed edge calc
    use_p = p_cal_val if edge_prob == "cal" else p_valid_val

    pnl = backtest_value(
        df=df_valid_bt,
        p_model=use_p[: df_valid_bt.height],  # align after gate
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
        bankroll_nom=bankroll_nom,
        kelly_fraction=1.0,
        stake_floor=0.0,
    )

    print("\n[Backtest @ validation — value]")
    print(f"  n_trades={pnl['n_trades']}  roi={pnl['roi']:.4f}  hit_rate={pnl['hit_rate']:.3f}  avg_edge={pnl['avg_edge']}")

    # ROI by odds buckets
    recs = pnl.get("recos", [])
    if recs:
        rec_df_all = pl.DataFrame(recs)
        # buckets
        edges = [1.01, 1.50, 2.00, 3.00, 5.00, 10.00, 50.00, 1000.0]
        out_rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (rec_df_all["ltp"] >= lo) & (rec_df_all["ltp"] < hi)
            sub = rec_df_all.filter(mask)
            if sub.height == 0:
                out_rows.append((f"[{lo:.2f}, {hi:5.2f})", 0, float("nan"), float("nan")))
                continue
            # simulate per bucket using unit_ret for side, scaled by stake_gbp
            idx = mask.to_numpy().nonzero()[0]
            unit_ret = np.asarray(pnl["unit_ret"])[idx]
            stakes = sub["stake_gbp"].to_numpy()
            wins = None  # not needed for ROI since we have unit_ret
            profit = float(np.sum(unit_ret * stakes))
            staked = float(np.sum(stakes))
            roi_b = profit / staked if staked > 0 else float("nan")
            # crude "hit" as mean of (unit_ret>0)
            hit = float((unit_ret > 0).mean()) if unit_ret.size else float("nan")
            out_rows.append((f"[{lo:.2f}, {hi:5.2f})", int(sub.height), hit, roi_b))
        print("\n[Value head — ROI by decimal odds bucket]")
        print("  bucket         n    hit    roi")
        for b, n_b, hit_b, roi_b in out_rows:
            print(f"  {b:>12} {n_b:6d}  {'' if np.isnan(hit_b) else f'{hit_b:0.3f}':>5}  {'' if np.isnan(roi_b) else f'{roi_b:0.3f}':>6}")

        # Side summary
        by_side = rec_df_all.group_by("side").agg([
            pl.len().alias("n"),
            pl.mean("edge_back").alias("avg_edge"),
            pl.mean("stake_gbp").alias("avg_stake_gbp"),
        ])
        print("\n[Backtest — side summary]")
        for row in by_side.iter_rows(named=True):
            print(f"  side={row['side']:>4}  n={row['n']:>5}  avg_edge={row['avg_edge']:.4f}  avg_stake_gbp=£{row['avg_stake_gbp']:.2f}")

        # Save recos CSV
        rec_df = rec_df_all.select([
            "marketId","selectionId","ltp","side","stake_gbp","stake_frac","p_model","p_market","edge_back"
        ])
        rec_file = OUTPUT_DIR / f"edge_recos_valid_{_fmt(asof)}_T{train_days}.csv"
        rec_df.write_csv(str(rec_file))
        print(f"\nSaved recommendations → {rec_file}")

        # Staking comparison (diagnostic): flat £10 vs Kelly (already sized in £)
        unit_ret = np.asarray(pnl["unit_ret"])
        n_trades = int(pnl["n_trades"])
        flat_stake = 10.0
        flat_profit = float(np.sum(unit_ret * flat_stake))
        flat_staked = flat_stake * n_trades
        flat_roi = flat_profit / flat_staked if flat_staked > 0 else 0.0

        kelly_stakes = np.asarray(pnl["stakes"])
        kelly_profit = float(np.sum(unit_ret * kelly_stakes))
        kelly_staked = float(np.sum(kelly_stakes))
        kelly_roi = kelly_profit / kelly_staked if kelly_staked > 0 else 0.0
        kelly_avg = float(kelly_staked / max(1, n_trades))

        print("\n[Staking comparison]")
        print(f"  Flat £10 stake    → trades={n_trades}  staked=£{flat_staked:.2f}  profit=£{flat_profit:.2f}  roi={flat_roi:.3f}")
        print(f"  Kelly (nom £{bankroll_nom:.0f}) → trades={n_trades}  staked=£{kelly_staked:.2f}  profit=£{kelly_profit:.2f}  roi={kelly_roi:.3f}  avg_stake=£{kelly_avg:.2f}")

        # Kelly stake distribution
        if stake_mode == "kelly" and kelly_stakes.size > 0:
            hist = np.percentile(kelly_stakes, [0, 25, 50, 75, 100])
            print("\n[Kelly stake distribution]")
            print(f"  min=£{hist[0]:.2f}  p25=£{hist[1]:.2f}  median=£{hist[2]:.2f}  p75=£{hist[3]:.2f}  max=£{hist[4]:.2f}")

    # Save models + validation detail
    value_name = f"edge_value_xgb_{preoff_minutes}m_{_fmt(asof)}_T{train_days}.json"
    pm_name = f"edge_price_xgb_{pm_horizon_secs}s_{preoff_minutes}m_{_fmt(asof)}_T{train_days}.json"
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
        "implied_prob": _compute_market_prob(df_valid, market_prob_mode),
        "y_win": yva,
        "p_win_raw": p_valid_val,
        "p_win_cal": p_cal_val,
        "y_pm_up": yva_pm,
        "p_pm_up": p_valid_pm,
        "pm_delta_ticks": df_valid["pm_delta_ticks"],
    })
    rep_file = OUTPUT_DIR / f"edge_valid_both_{_fmt(asof)}_T{train_days}.csv"
    rep.write_csv(str(rep_file))
    print(f"Saved validation detail → {rep_file}")


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description=(
        "Temporal split with 2-day validation, value+price heads, PM gate, calibration, value backtest."
    ))
    # Data
    ap.add_argument("--curated", required=True, help="/mnt/nvme/betfair-curated or s3://bucket")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--asof", required=True, help="Validation end date (YYYY-MM-DD). Valid=[asof-1, asof]; Train ends at asof-2.")
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
    ap.add_argument("--edge-thresh", type=float, default=0.015, help="Enter trade if model edge > threshold")
    ap.add_argument("--no-calibrate", action="store_true", help="Disable isotonic calibration for value head")
    ap.add_argument("--edge-prob", choices=["raw", "cal"], default="cal", help="Use raw or calibrated probabilities for edge calc")
    ap.add_argument("--no-sum-to-one", action="store_true", help="Disable per-market sum-to-one normalization on p_model")

    # Market prob mode and selection policy
    ap.add_argument("--market-prob", choices=["ltp", "overround"], default="overround")
    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--side", choices=["back", "lay", "both"], default="back")

    # Odds band
    ap.add_argument("--ltp-min", type=float, default=1.5)
    ap.add_argument("--ltp-max", type=float, default=5.0)

    # Staking
    ap.add_argument("--stake", choices=["flat", "kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--bankroll-nom", type=float, default=1000.0, help="Nominal bankroll for Kelly sizing/prints (GBP)")

    # Price-move head / gate
    ap.add_argument("--pm-horizon-secs", type=int, default=300, help="Seconds for price-move label")
    ap.add_argument("--pm-tick-threshold", type=int, default=1, help="Min future move in ticks for label=1")
    ap.add_argument("--pm-slack-secs", type=int, default=3, help="Tolerance slack around horizon for asof join")
    ap.add_argument("--pm-cutoff", type=float, default=0.0, help="Require p_pm_up ≥ cutoff to keep a trade (0 disables)")

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
        pm_cutoff=args.pm_cutoff,
    )


if __name__ == "__main__":
    main()
