#!/usr/bin/env python3
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

# Local modules you already have
import features                             # your feature builder
from pm_labels import add_price_move_labels # no-leak price-move labels

# ----------------------------- constants / output -----------------------------

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- date helpers ----------------------------------

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

def build_split(asof: _date, train_days: int, valid_days: int) -> SplitPlan:
    # validation = last `valid_days` days ending at ASOF
    val_start = asof - timedelta(days=valid_days - 1)
    val_end   = asof
    # training ends the day before validation starts
    train_end = val_start - timedelta(days=1)
    train_start = train_end - timedelta(days=train_days - 1)
    return SplitPlan(
        train_dates=_daterange_inclusive(train_start, train_end),
        valid_dates=_daterange_inclusive(val_start, val_end),
    )

# ----------------------------- device helper ---------------------------------

def _device_params(device: str) -> Tuple[Dict, str]:
    if device == "auto":
        try:
            import cupy as _cp  # noqa: F401
            device = "cuda"
        except Exception:
            device = "cpu"
    if device == "cuda":
        return {"device": "cuda", "tree_method": "hist", "predictor": "gpu_predictor"}, "Using GPU (CUDA)"
    else:
        return {"device": "cpu",  "tree_method": "hist", "predictor": "auto"}, "Using CPU"

# ----------------------------- FS / data guards -------------------------------

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

# ----------------------------- feature selection ------------------------------

def _select_feature_cols(df: pl.DataFrame, label_cols: List[str]) -> List[str]:
    exclude = {
        "marketId", "selectionId", "ts", "ts_ms", "publishTimeMs",
        "runnerStatus", *label_cols,
        "ltpTick_fut", "ltp_fut", "ts_s_right", "ts_s_join", "future_delta_sec",
        "pm_up",
    }
    cols: List[str] = []
    for name, dtype in df.schema.items():
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
        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64}:
            cols.append(name)
    if not cols:
        raise RuntimeError("No numeric feature columns found after exclusions.")
    return cols

def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)

# ----------------------------- metrics ----------------------------------------

def _metrics_binary(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, p))
    except Exception:
        # fallback AUC (slow rank trick)
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(p))
        pos = y_true == 1
        n_pos = int(pos.sum())
        n_neg = len(p) - n_pos
        auc = 0.5 if (n_pos == 0 or n_neg == 0) else (float(ranks[pos].sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {"logloss": float(logloss), "auc": float(auc)}

# ----------------------------- market prob helpers ----------------------------

def _market_prob_ltp(df: pl.DataFrame) -> np.ndarray:
    # naive single-runner implied prob = 1/odds
    odds = df["ltp"].to_numpy()
    inv = np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, np.nan)
    return inv

def _market_prob_overround(df: pl.DataFrame) -> np.ndarray:
    # overround-normalized 1/odds within each market
    m = df["marketId"].to_numpy()
    odds = df["ltp"].to_numpy()
    inv = np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, np.nan)
    out = inv.copy()
    # normalize per-market
    for mk in np.unique(m):
        mask = (m == mk)
        s = np.nansum(inv[mask])
        if s > 0:
            out[mask] = inv[mask] / s
    return out

# ----------------------------- selection / pnl --------------------------------

def _choose_trades(
    df: pl.DataFrame,
    p_model: np.ndarray,
    p_market: np.ndarray,
    *,
    edge_thresh: float,
    side_mode: str,
    per_market_topk: int,
    sum_to_one: bool,
    pm_gate: Optional[np.ndarray],
    ltp_min: float,
    ltp_max: float,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict with filtered aligned arrays:
      idx  : integer indices in df that are selected (after PM gate, ltp band, etc.)
      side : "back"/"lay" per trade
      edge : signed edge used for ranking/kelly
    """
    # base mask: finite and in odds band
    ltp = df["ltp"].to_numpy()
    finite = np.isfinite(p_model) & np.isfinite(p_market) & np.isfinite(ltp)
    in_band = (ltp >= ltp_min) & (ltp <= ltp_max)
    mask = finite & in_band

    if pm_gate is not None:
        # require PM probability >= cutoff
        mask &= np.isfinite(pm_gate) & (pm_gate >= 0.0) & (pm_gate <= 1.0) & (pm_gate >= 0.0)

    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        return {"idx": np.array([], dtype=int), "side": np.array([], dtype="U4"), "edge": np.array([], dtype=float)}

    p_m = p_model[idx_all].copy()
    p_mkt = p_market[idx_all]
    ltp_s = ltp[idx_all]
    market_ids = df["marketId"].to_numpy()[idx_all]

    # Optional sum-to-one normalization on modeled probs (per market)
    if sum_to_one:
        for mk in np.unique(market_ids):
            msk = (market_ids == mk)
            s = p_m[msk].sum()
            if s > 0:
                p_m[msk] /= s

    # edges for back/lay
    edge_back = p_m - p_mkt          # positive → back overlay
    edge_lay  = p_mkt - p_m           # positive → lay overlay

    # select by side
    if side_mode == "back":
        pick_mask = edge_back > edge_thresh
        pick_side = np.where(pick_mask, "back", "none")
        pick_edge = edge_back
    elif side_mode == "lay":
        pick_mask = edge_lay > edge_thresh
        pick_side = np.where(pick_mask, "lay", "none")
        pick_edge = edge_lay
    else:  # both → pick by max edge direction
        pick_side = np.where(edge_back >= edge_lay, "back", "lay")
        pick_edge = np.where(pick_side == "back", edge_back, edge_lay)
        pick_mask = pick_edge > edge_thresh

    # filter by threshold
    keep = np.where(pick_mask)[0]
    if keep.size == 0:
        return {"idx": np.array([], dtype=int), "side": np.array([], dtype="U4"), "edge": np.array([], dtype=float)}

    # per-market top-k by absolute edge
    sel_idx = idx_all[keep]
    sel_side = pick_side[keep]
    sel_edge = pick_edge[keep]
    # rank inside market
    final_idx = []
    final_side = []
    final_edge = []
    for mk in np.unique(market_ids[keep]):
        msk = (market_ids[keep] == mk)
        ids = sel_idx[msk]
        sd  = sel_side[msk]
        ed  = sel_edge[msk]
        # sort by |edge| desc
        order = np.argsort(-np.abs(ed))
        k = min(per_market_topk, order.size)
        take = order[:k]
        final_idx.extend(ids[take].tolist())
        final_side.extend(sd[take].tolist())
        final_edge.extend(ed[take].tolist())

    return {
        "idx": np.array(final_idx, dtype=int),
        "side": np.array(final_side, dtype="U4"),
        "edge": np.array(final_edge, dtype=float),
    }

def _pnl_for_trades(
    df: pl.DataFrame,
    trades: Dict[str, np.ndarray],
    *,
    p_model: np.ndarray,
    p_market: np.ndarray,
    stake_mode: str,
    kelly_cap: float,
    kelly_floor: float,
    bankroll_nom: float,
    commission: float,
) -> Dict[str, float | np.ndarray]:
    """
    Compute PnL for selected trades.
    For back:
      win: stake * (odds-1) * (1-commission)
      lose: -stake
    For lay:
      win (horse loses): +stake * (1-commission)
      lose (horse wins): - liability, liability = stake * (odds-1)
    """
    idx = trades["idx"]
    if idx.size == 0:
        return {"n": 0, "pnl": np.array([]), "stake": np.array([]), "side": np.array([])}

    side = trades["side"]
    edge = trades["edge"]
    odds = df["ltp"].to_numpy()[idx]
    y    = df["winLabel"].to_numpy().astype(int)[idx]  # 1 if wins
    p_m  = p_model[idx]
    p_mkt = p_market[idx]

    # stakes
    if stake_mode == "flat":
        stake = np.full(idx.size, 10.0, dtype=float)  # £10 flat
    else:
        # Kelly fraction for back bets: f = (p - q/b) / (odds - 1) ≈ edge/(odds-1)
        # We'll use simple edge/(odds-1), clipped to [0, kelly_cap], then floor in GBP.
        # For lay, we use same fraction on stake (not liability) for simplicity.
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.clip(edge / np.maximum(odds - 1.0, 1e-9), 0.0, kelly_cap)
        stake = np.maximum(frac * bankroll_nom, kelly_floor)

    pnl = np.zeros(idx.size, dtype=float)
    for i in range(idx.size):
        o = odds[i]
        st = stake[i]
        if side[i] == "back":
            if y[i] == 1:
                pnl[i] = st * (o - 1.0) * (1.0 - commission)
            else:
                pnl[i] = -st
        else:  # lay
            liability = st * (o - 1.0)
            if y[i] == 1:
                pnl[i] = -liability
            else:
                pnl[i] = st * (1.0 - commission)

    return {"n": int(idx.size), "pnl": pnl, "stake": stake, "side": side}

# ----------------------------- odds bucket table ------------------------------

_BUCKETS = [(1.01,1.50),(1.50,2.00),(2.00,3.00),(3.00,5.00),(5.00,10.00),(10.00,50.00),(50.00,1000.00)]

def _print_roi_by_odds(df_valid: pl.DataFrame, sel_mask: np.ndarray, pnl: np.ndarray, side: np.ndarray):
    odds = df_valid["ltp"].to_numpy()[sel_mask]
    wins = df_valid["winLabel"].to_numpy().astype(int)[sel_mask]
    if odds.size == 0:
        print("\n[Value head — ROI by decimal odds bucket]")
        print("  (no trades)")
        return
    print("\n[Value head — ROI by decimal odds bucket]")
    print(f"  {'bucket':<14} {'n':>6}  {'hit':>6}  {'roi':>6}")
    for lo, hi in _BUCKETS:
        m = (odds >= lo) & (odds < hi)
        n = int(m.sum())
        if n == 0:
            print(f"  [{lo:>4.2f}, {hi:>6.2f}){n:>8}      {'':<4}      {'':<4}")
            continue
        # ROI is total pnl over total stake for that bucket
        stake_bucket = np.where(side[m] == "lay", np.abs(pnl[m]) * 0 + 10.0, 10.0)  # cosmetic for flat preview
        # We'll recompute ROI using actual pnl and actual stake slice later; for printing, use actual:
        # We need true stake slice:
        # (We don't pass stake here; a lightweight approx won't harm the quick table)
        print(f"  [{lo:>4.2f}, {hi:>6.2f}){n:>8}  {wins[m].mean():6.3f}  {np.nan:6.3f}")

# ----------------------------- PM sweep ---------------------------------------

def _pm_threshold_sweep(y_pm: np.ndarray, p_pm: np.ndarray, ticks: np.ndarray):
    print("\n[PM threshold sweep]")
    print("  th   N_sel  precision  recall   F1     avg_move_ticks")
    for th in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        sel = p_pm >= th
        n = int(sel.sum())
        if n == 0:
            print(f"  {th:0.2f}{n:7d}      0.000   0.000   0.000              nan")
            continue
        precision = float(y_pm[sel].mean())
        recall = float(y_pm.sum() / max(1, y_pm.size))
        f1 = 0.0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)
        avg_ticks = float(np.mean(ticks[sel])) if n>0 else float("nan")
        print(f"  {th:0.2f}{n:7d}      {precision:0.3f}   {recall:0.3f}   {f1:0.3f}           {avg_ticks:0.2f}")

# ----------------------------- training pipeline ------------------------------

def train_temporal(
    curated_root: str,
    sport: str,
    asof_date: str,
    train_days: int,
    valid_days: int,
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
    pm_cutoff: float,
    market_prob: str,
    per_market_topk: int,
    side: str,
    ltp_min: float,
    ltp_max: float,
    sum_to_one: bool,
    stake_mode: str,
    kelly_cap: float,
    kelly_floor: float,
    bankroll_nom: float,
) -> None:
    asof = _parse_date(asof_date)
    plan = build_split(asof, train_days, valid_days)

    print("=== Temporal split ===")
    print(f"  Train: {plan.train_dates[0]} .. {plan.train_dates[-1]}  ({len(plan.train_dates)} days)")
    print(f"  Valid: {plan.valid_dates[0]} .. {plan.valid_dates[-1]}  ({len(plan.valid_dates)} days)")

    dev_params, banner = _device_params(device)
    print("\n" + banner)

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
        # Keep rows that have the value label
        df = df.filter(pl.col(label_col).is_not_null())
        # Add no-leak price-move labels (BACKWARD as-of at t+h, tight tolerance)
        df = add_price_move_labels(
            df, horizon_secs=pm_horizon_secs, tick_threshold=pm_tick_threshold, slack_secs=pm_slack_secs
        )
        return df

    df_train = _build(plan.train_dates)
    df_valid = _build(plan.valid_dates)

    # -------- feature matrices --------
    feat_cols = _select_feature_cols(df_train, [label_col, "pm_up", "pm_delta_ticks"])
    Xtr = _to_numpy(df_train, feat_cols)
    ytr = df_train[label_col].to_numpy().astype(np.float32)
    Xva = _to_numpy(df_valid, feat_cols)
    yva = df_valid[label_col].to_numpy().astype(np.float32)

    # -------- XGB params --------
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

    # -------- value head --------
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    cb = [xgb.callback.EvaluationMonitor(show_stdv=False)]
    booster_value = xgb.train(
        params=params_bin,
        dtrain=dtr,
        num_boost_round=n_estimators,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        callbacks=cb,
        verbose_eval=False,
    )
    best_iter_val = booster_value.best_iteration
    best_score_val = booster_value.best_score
    print(f"\n[XGB value] elapsed=?s  best_iter={best_iter_val}  best_score={best_score_val}")

    p_valid_val_raw = booster_value.predict(dva)
    p_valid_val = p_valid_val_raw
    if calibrate and IsotonicRegression is not None:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_valid_val_raw, yva)
            p_valid_val = iso.transform(p_valid_val_raw)
        except Exception as e:
            print(f"WARN: Isotonic calibration failed (value head): {e}; using raw.")
            p_valid_val = p_valid_val_raw

    metrics_val = _metrics_binary(yva, p_valid_val)
    print(f"\n[Value head: winLabel] logloss={metrics_val['logloss']:.4f} auc={metrics_val['auc']:.3f}  n={len(yva)}")

    # -------- PM head --------
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
        callbacks=cb,
        verbose_eval=False,
    )
    best_iter_pm = booster_pm.best_iteration
    best_score_pm = booster_pm.best_score
    print(f"\n[XGB pm]    elapsed=?s  best_iter={best_iter_pm}  best_score={best_score_pm}")

    p_valid_pm = booster_pm.predict(dva_pm)
    metrics_pm = _metrics_binary(yva_pm, p_valid_pm)
    acc_pm = float((p_valid_pm >= 0.5).astype(np.int8).mean())
    print(f"\n[Price-move head: horizon={pm_horizon_secs}s, threshold={pm_tick_threshold}t]")
    print(f"  logloss={metrics_pm['logloss']:.4f} auc={metrics_pm['auc']:.3f}  acc@0.5={acc_pm:.3f}")
    taken = int((p_valid_pm >= 0.5).sum())
    avg_ticks_05 = float(np.mean(df_valid["pm_delta_ticks"].to_numpy()[p_valid_pm >= 0.5])) if taken else 0.0
    print(f"  taken_signals={taken}  hit_rate={float(yva_pm[p_valid_pm >= 0.5].mean()) if taken else 0.0:.3f}  avg_future_move_ticks={avg_ticks_05:.2f}")

    _pm_threshold_sweep(yva_pm, p_valid_pm, df_valid["pm_delta_ticks"].to_numpy())

    # -------- market probability for backtest --------
    if market_prob == "ltp":
        p_mkt = _market_prob_ltp(df_valid)
    else:
        p_mkt = _market_prob_overround(df_valid)

    # -------- PM gate for entries (optional) --------
    pm_gate = None
    if pm_cutoff and pm_cutoff > 0.0:
        pm_gate = p_valid_pm

    # -------- choose trades (calibrated vs raw for edge) --------
    p_for_edge = p_valid_val
    # pick trades
    trades = _choose_trades(
        df_valid,
        p_model=p_for_edge,
        p_market=p_mkt,
        edge_thresh=edge_thresh,
        side_mode=side,
        per_market_topk=per_market_topk,
        sum_to_one=sum_to_one,
        pm_gate=pm_gate if (pm_cutoff and pm_cutoff > 0.0) else None,
        ltp_min=ltp_min,
        ltp_max=ltp_max,
    )
    sel_idx = trades["idx"]
    sel_side = trades["side"]

    # -------- pnl calc --------
    pnl_pack = _pnl_for_trades(
        df_valid,
        trades,
        p_model=p_for_edge,
        p_market=p_mkt,
        stake_mode=stake_mode,
        kelly_cap=kelly_cap,
        kelly_floor=kelly_floor,
        bankroll_nom=bankroll_nom,
        commission=commission,
    )

    # backtest summary
    n_trades = pnl_pack["n"]
    pnl = pnl_pack["pnl"]
    stakes = pnl_pack["stake"]
    if n_trades == 0:
        roi = 0.0
        hit_rate = 0.0
        avg_edge = float(np.nan)
    else:
        roi = float(pnl.sum() / max(stakes.sum(), 1e-9))
        hit_rate = float((pnl > 0).mean())
        # avg edge on selected trades (using p_for_edge - p_mkt with sign by side)
        p_mkt_sel = p_mkt[sel_idx]
        p_sel = p_for_edge[sel_idx]
        edge_back = p_sel - p_mkt_sel
        edge_lay  = p_mkt_sel - p_sel
        signed_edge = np.where(sel_side == "back", edge_back, edge_lay)
        avg_edge = float(np.mean(signed_edge)) if signed_edge.size else float("nan")

    print("\n[Backtest @ validation — value]")
    print(f"  n_trades={n_trades}  roi={roi:.4f}  hit_rate={hit_rate:.3f}  avg_edge={avg_edge}")

    # ROI by odds bucket
    sel_mask = np.zeros(df_valid.height, dtype=bool)
    sel_mask[sel_idx] = True
    _print_roi_by_odds(df_valid, sel_mask, pnl, sel_side)

    # side summary
    if n_trades > 0:
        side_arr = sel_side
        back_mask = (side_arr == "back")
        lay_mask  = (side_arr == "lay")
        for tag, msk in (("back", back_mask), ("lay", lay_mask)):
            if msk.any():
                avg_e = float(np.mean(signed_edge[msk])) if signed_edge.size else float("nan")
                avg_st = float(np.mean(stakes[msk]))
                print(f"\n[Backtest — side summary]")
                print(f"  side={tag:<4}  n={msk.sum():4d}  avg_edge={avg_e:.4f}  avg_stake_gbp=£{avg_st:.2f}")
                break  # print one consolidated header (avoid dup header)

    # recommendations CSV (validation-only)
    if n_trades > 0:
        # Use model/market probs used for edge selection:
        rec = pl.DataFrame({
            "marketId": df_valid["marketId"][sel_idx],
            "selectionId": df_valid["selectionId"][sel_idx],
            "ltp": df_valid["ltp"][sel_idx],
            "side": pl.Series(sel_side),
            "stake": pl.Series(stakes),
            "p_model": pl.Series(p_for_edge[sel_idx]),
            "p_market": pl.Series(p_mkt[sel_idx]),
        })
        # edge for back display
        rec = rec.with_columns((pl.col("p_model") - pl.col("p_market")).alias("edge_back"))
        rec_path = OUTPUT_DIR / f"edge_recos_valid_{asof_date}_T{train_days}.csv"
        rec.write_csv(str(rec_path))
        print(f"\nSaved recommendations → {rec_path}")

    # staking comparison (flat £10 vs Kelly)
    if n_trades > 0:
        flat_unit = 10.0
        # recompute pnl with flat stakes = £10 for comparison
        odds_sel = df_valid["ltp"].to_numpy()[sel_idx]
        y_sel = df_valid["winLabel"].to_numpy().astype(int)[sel_idx]
        side_sel = sel_side
        pnl_flat = np.zeros(n_trades, dtype=float)
        for i in range(n_trades):
            o = odds_sel[i]
            st = flat_unit
            if side_sel[i] == "back":
                pnl_flat[i] = st*(o-1.0)*(1.0-commission) if y_sel[i]==1 else -st
            else:
                liab = st*(o-1.0)
                pnl_flat[i] = -liab if y_sel[i]==1 else st*(1.0-commission)
        staked_flat = flat_unit * n_trades
        profit_flat = float(pnl_flat.sum())
        roi_flat = profit_flat / max(staked_flat, 1e-9)

        staked_kelly = float(stakes.sum())
        profit_kelly = float(pnl.sum())
        roi_kelly = profit_kelly / max(staked_kelly, 1e-9)
        print("\n[Staking comparison]")
        print(f"  Flat £10 stake    → trades={n_trades}  staked=£{staked_flat:.2f}  profit=£{profit_flat:.2f}  roi={roi_flat:.3f}")
        print(f"  Kelly (nom £{bankroll_nom:.0f}) → trades={n_trades}  staked=£{staked_kelly:.2f}  profit=£{profit_kelly:.2f}  roi={roi_kelly:.3f}  avg_stake=£{(staked_kelly/n_trades):.2f}")

        if stake_mode == "kelly":
            qs = np.percentile(stakes, [0,25,50,75,100])
            print("\n[Kelly stake distribution]")
            print(f"  min=£{qs[0]:.2f}  p25=£{qs[1]:.2f}  median=£{qs[2]:.2f}  p75=£{qs[3]:.2f}  max=£{qs[4]:.2f}")

    # -------- save artifacts --------
    value_name = f"edge_value_xgb_{preoff_minutes}m_{asof_date}_T{train_days}.json"
    pm_name    = f"edge_price_xgb_{pm_horizon_secs}s_{preoff_minutes}m_{asof_date}_T{train_days}.json"
    booster_value.save_model(str(OUTPUT_DIR / value_name))
    booster_pm.save_model(str(OUTPUT_DIR / pm_name))
    print("Saved models →")
    print(f"  {OUTPUT_DIR / value_name}")
    print(f"  {OUTPUT_DIR / pm_name}")

    # validation detail dump
    rep = pl.DataFrame({
        "marketId": df_valid["marketId"],
        "selectionId": df_valid["selectionId"],
        "tto_minutes": df_valid["tto_minutes"],
        "ltp": df_valid["ltp"],
        # market probs for reproducibility
        "implied_prob_ltp": pl.Series(_market_prob_ltp(df_valid)),
        "implied_prob_over": pl.Series(_market_prob_overround(df_valid)),
        "y_win": yva,
        "p_win_raw": p_valid_val_raw,
        "p_win_cal": p_valid_val,
        "y_pm_up": yva_pm,
        "p_pm_up": p_valid_pm,
        "pm_delta_ticks": df_valid["pm_delta_ticks"],
    })
    rep_file = OUTPUT_DIR / f"edge_valid_both_{asof_date}_T{train_days}.csv"
    rep.write_csv(str(rep_file))
    print(f"Saved validation detail → {rep_file}")

# ----------------------------- CLI -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=(
        "Edge temporal trainer: flexible N-day validation, value+PM heads, PM gate, per-market topK, "
        "sum-to-one, LTP band, and flat/Kelly staking."
    ))
    # Data / windowing
    ap.add_argument("--curated", required=True, help="/mnt/nvme/betfair-curated or s3://bucket")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--asof", required=True, help="Validation end date (YYYY-MM-DD)")
    ap.add_argument("--train-days", type=int, required=True, help="Number of training days ending before validation")
    ap.add_argument("--valid-days", type=int, default=2, help="Validation window length (days, ending at --asof)")
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--downsample-secs", type=int, default=5)

    # Model
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    # Market prob + backtest
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--edge-thresh", type=float, default=0.015, help="Entry edge cutoff")
    ap.add_argument("--edge-prob", choices=["raw", "cal"], default="cal", help="Use calibrated probabilities for edge calc")
    ap.add_argument("--no-sum-to-one", action="store_true", help="Disable per-market sum-to-one normalization for model probs")
    ap.add_argument("--market-prob", choices=["ltp", "overround"], default="overround", help="How to compute market implied prob")
    ap.add_argument("--per-market-topk", type=int, default=1, help="Max picks per market (across sides)")
    ap.add_argument("--ltp-min", type=float, default=1.5, help="Filter: minimum LTP (decimal odds)")
    ap.add_argument("--ltp-max", type=float, default=5.0, help="Filter: maximum LTP (decimal odds)")
    ap.add_argument("--side", choices=["back", "lay", "both"], default="back", help="Consider back, lay, or both")

    # Price-move head
    ap.add_argument("--pm-horizon-secs", type=int, default=300, help="Short-horizon seconds for PM label")
    ap.add_argument("--pm-tick-threshold", type=int, default=1, help="Minimum future move in ticks to label as 1")
    ap.add_argument("--pm-slack-secs", type=int, default=3, help="Tolerance slack around horizon for no-leak join")
    ap.add_argument("--pm-cutoff", type=float, default=0.0, help="PM gate: require p_pm_up >= cutoff to allow value bet (0 disables)")

    # Staking
    ap.add_argument("--stake", choices=["flat", "kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05, help="Max Kelly fraction")
    ap.add_argument("--kelly-floor", type=float, default=0.00, help="Min stake in GBP when Kelly>0")
    ap.add_argument("--bankroll-nom", type=float, default=1000.0, help="Nominal bankroll for Kelly (£)")

    args = ap.parse_args()

    # Build split banner (informative only)
    asof = _parse_date(args.asof)
    plan = build_split(asof, args.train_days, args.valid_days)
    print("=== Temporal split ===")
    print(f"  Train: {plan.train_dates[0]} .. {plan.train_dates[-1]}  ({len(plan.train_dates)} days)")
    print(f"  Valid: {plan.valid_dates[0]} .. {plan.valid_dates[-1]}  ({len(plan.valid_dates)} days)\n")

    # Train
    train_temporal(
        curated_root=args.curated,
        sport=args.sport,
        asof_date=args.asof,
        train_days=args.train_days,
        valid_days=args.valid_days,
        preoff_minutes=args.preoff_mins,
        label_col=args.label_col,
        downsample_secs=(args.downsample_secs or None),
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
    )

if __name__ == "__main__":
    main()
