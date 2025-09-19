#!/usr/bin/env python3
"""
Clean temporal trainer with two heads (value + price-move), CUDA default, and no-leak labels.
- Validation window: [ASOF-1, ASOF]
- Training window: last --train-days ending at (ASOF-2)
- Uses pm_labels.add_price_move_labels (BACKWARD as-of at t+h with tight tolerance)
- Skips missing dates gracefully
- Saves models + a validation CSV to project-level output/

Example:
  python train_edge_temporal.py \
    --curated /mnt/nvme/betfair-curated \
    --sport horse-racing \
    --asof 2025-09-18 \
    --train-days 13 \
    --preoff-mins 30 \
    --downsample-secs 5 \
    --commission 0.02 \
    --edge-thresh 0.02 \
    --pm-horizon-secs 60 \
    --pm-tick-threshold 1 \
    --pm-slack-secs 3
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
import features  # your feature builder
from pm_labels import add_price_move_labels  # no-leak price-move labels

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
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


# Polars dtype helper (robust across versions)
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
    }
    cols: List[str] = []
    for name, dtype in df.schema.items():
        if name in exclude:
            continue
        if "label" in name.lower() or "target" in name.lower():
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


# ----------------------------- value backtest -----------------------------

def _sum_to_one_by_market(probs: np.ndarray, market_ids: np.ndarray) -> np.ndarray:
    probs = probs.copy()
    for m in np.unique(market_ids):
        mask = market_ids == m
        s = probs[mask].sum()
        if s > 0:
            probs[mask] = probs[mask] / s
    return probs


def backtest_value(df: pl.DataFrame, p_raw: np.ndarray, commission: float, edge_thresh: float) -> Dict[str, float]:
    market_ids = df["marketId"].to_numpy()
    p = _sum_to_one_by_market(p_raw, market_ids)
    pim = df["implied_prob"].to_numpy()
    edge = p - pim
    sel = edge > edge_thresh
    if sel.sum() == 0:
        return {"n_trades": 0, "roi": 0.0, "hit_rate": 0.0, "avg_edge": float(np.nan)}
    ltp = df["ltp"].to_numpy()
    y = df["winLabel"].to_numpy().astype(int)
    profit = np.where(y == 1, (ltp - 1.0) * (1.0 - commission), -1.0)
    pnl = profit[sel].sum()
    n = int(sel.sum())
    return {
        "n_trades": n,
        "roi": float(pnl / n),
        "hit_rate": float(y[sel].mean()),
        "avg_edge": float(edge[sel].mean()),
    }


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


# ----------------------------- metrics -----------------------------

def _metrics_binary(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, p))
    except Exception:
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(p))
        pos = y_true == 1
        n_pos = int(pos.sum())
        n_neg = len(p) - n_pos
        auc = 0.5 if (n_pos == 0 or n_neg == 0) else (float(ranks[pos].sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {"logloss": float(logloss), "auc": float(auc)}


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
        df = add_price_move_labels(df, horizon_secs=pm_horizon_secs, tick_threshold=pm_tick_threshold, slack_secs=pm_slack_secs)
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

    metrics_val = _metrics_binary(yva, p_valid_val)
    print(f"\n[Value head: winLabel] logloss={metrics_val['logloss']:.4f} auc={metrics_val['auc']:.3f}  n={len(yva)}")

    pnl = backtest_value(df_valid, p_cal_val, commission=commission, edge_thresh=edge_thresh)
    print("[Backtest @ validation — value]")
    print(f"  n_trades={pnl['n_trades']}  roi={pnl['roi']:.4f}  hit_rate={pnl['hit_rate']:.3f}  avg_edge={pnl['avg_edge']}")

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
    hit_up = float(yva_pm[preds].mean()) if taken else 0.0
    avg_move_ticks = float(np.mean(df_valid["pm_delta_ticks"].to_numpy()[preds])) if taken else 0.0

    print(f"\n[Price-move head: horizon={pm_horizon_secs}s, threshold={pm_tick_threshold}t]")
    print(f"  logloss={metrics_pm['logloss']:.4f} auc={metrics_pm['auc']:.3f}  acc@0.5={acc:.3f}")
    print(f"  taken_signals={taken}  hit_rate={hit_up:.3f}  avg_future_move_ticks={avg_move_ticks:.2f}")

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
        "Temporal split with 2-day validation, value+price heads, calibration, value backtest. "
        "Validation end date is --asof; valid=[asof-1, asof]; train ends at asof-2."
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
    )


if __name__ == "__main__":
    main()
