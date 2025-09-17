# ml/train_xgb_country.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import polars as pl
import xgboost as xgb

from . import features  # uses the same loader the working trainer uses

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utilities
# ----------------------------

def _daterange(end_date_str: str, days: int) -> List[str]:
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    d = start
    out: List[str] = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _select_feature_cols(df: pl.DataFrame, label_col: str) -> List[str]:
    """
    Keep numeric columns only, excluding ids/labels/timestamps. We *include* country_feat.
    """
    exclude = {
        "marketId", "selectionId", "ts", "ts_ms", "publishTimeMs",
        label_col, "runnerStatus"
    }
    cols: List[str] = []
    schema = df.collect_schema()
    for name, dtype in zip(schema.names(), schema.dtypes()):
        if name in exclude:
            continue
        low = name.lower()
        if "label" in low or "target" in low:
            continue
        try:
            if dtype.is_numeric():
                cols.append(name)
        except Exception:
            # Be permissive on old Polars
            pass
    if not cols:
        raise RuntimeError("No numeric feature columns found to train on.")
    return cols


def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    # Robust to older Polars (avoid dtype kw). Fill NaNs sensibly.
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


def _device_params(device: str) -> Tuple[Dict, str]:
    if device == "auto":
        try:
            import cupy as _cp  # noqa: F401
            device = "cuda"
        except Exception:
            device = "cpu"
    if device == "cuda":
        return {"device": "cuda", "tree_method": "hist"}, "🚀 XGBoost on CUDA"
    return {"device": "cpu", "tree_method": "hist"}, "🚀 XGBoost on CPU"


def _metrics_binary(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))
    # Try AUC, fall back to rank-based if sklearn unavailable
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, p))
    except Exception:
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(p))
        pos = y_true == 1
        n_pos = int(np.sum(pos)); n_neg = len(p) - n_pos
        if n_pos == 0 or n_neg == 0:
            auc = 0.5
        else:
            auc = (np.sum(ranks[pos]) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {"logloss": float(logloss), "auc": float(auc)}


def _binwise_report(df: pl.DataFrame, y: np.ndarray, p: np.ndarray, edges: List[int]) -> None:
    if "tto_minutes" not in df.columns:
        return
    labels = [f"{edges[i]:02d}-{edges[i+1]:02d}" for i in range(len(edges) - 1)]
    expr = None
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i+1]
        cond = (pl.col("tto_minutes") > lo) & (pl.col("tto_minutes") <= hi)
        expr = cond.then(pl.lit(lab)) if expr is None else expr.when(cond).then(pl.lit(lab))
    tmp = df.with_columns((expr.otherwise(pl.lit(None))).alias("tto_bin"))
    print("\n[Validation by TTO bins]")
    for lab in labels:
        mask = (tmp.get_column("tto_bin") == lab).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        m = _metrics_binary(y[mask], p[mask])
        print(f"[{lab}] logloss={m['logloss']:.4f} auc={m['auc']:.3f} n={n}")


def _build_features_chunked(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int,
    batch_markets: int,
    downsample_secs: Optional[int],
    chunk_days: int,
) -> pl.DataFrame:
    parts: List[pl.DataFrame] = []
    total_rows = 0
    for i in range(0, len(dates), chunk_days):
        chunk = dates[i:i + chunk_days]
        print(f"  • building features for {chunk[0]}..{chunk[-1]}")
        df_c, raw_rows = features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=chunk,
            preoff_minutes=preoff_minutes,
            batch_markets=batch_markets,
            downsample_secs=downsample_secs,
        )
        total_rows += int(raw_rows or 0)
        if not df_c.is_empty():
            parts.append(df_c)
    if not parts:
        raise SystemExit("No features built.")
    df = pl.concat(parts, how="vertical", rechunk=True)
    print(f"  • total feature rows: {df.height} (raw: {total_rows})")
    return df


def _add_country_feature(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a numeric 'country_feat' column derived from 'country' OR 'eventCountryCode'.
    No filtering by country; purely a feature so the model can learn per-country effects.
    """
    if "country" in df.columns:
        df = df.with_columns(pl.col("country").fill_null("UNK").cast(pl.Categorical).to_physical().alias("country_feat"))
    elif "eventCountryCode" in df.columns:
        df = df.with_columns(pl.col("eventCountryCode").fill_null("UNK").cast(pl.Categorical).to_physical().alias("country_feat"))
    else:
        print("WARN: no country column found; continuing without 'country_feat'.")
    return df


def _train_xgb(
    df: pl.DataFrame,
    label_col: str,
    device: str,
    n_estimators: int,
    learning_rate: float,
    early_stopping_rounds: int,
) -> Tuple[xgb.Booster, Dict[str, float], int, List[str]]:
    feats = _select_feature_cols(df, label_col)
    print(f"[features] {len(feats)} columns")
    y = df[label_col].to_numpy().astype(np.float32)

    n = df.height
    if n < 1000:
        print(f"WARN: very few rows: {n}")

    split = int(n * 0.8)
    tr, va = df[:split], df[split:]
    Xtr = _to_numpy(tr, feats); Xva = _to_numpy(va, feats)
    ytr = y[:split]; yva = y[split:]

    dev_params, banner = _device_params(device)
    print(banner)
    params = {
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

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feats)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feats)
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=early_stopping_rounds or None,
        verbose_eval=False,
    )

    pva = bst.predict(dva)
    metrics = _metrics_binary(yva, pva)
    print(f"[valid] logloss={metrics['logloss']:.4f} auc={metrics['auc']:.3f}")
    _binwise_report(va, yva, pva, edges=[0, 30, 60, 90, 120, 180])

    best_it = bst.best_iteration if bst.best_iteration is not None else n_estimators
    return bst, metrics, best_it, feats


def _save_model_json(booster: xgb.Booster, features_used: List[str], filename: str) -> Path:
    out = OUTPUT_DIR / filename
    booster.save_model(str(out))
    # Also persist feature order (neighbor file)
    (OUTPUT_DIR / (Path(filename).stem + ".features.txt")).write_text("\n".join(features_used))
    print(f"Saved model: {out}")
    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser("Train XGBoost (adds 'country' as a feature; no filtering).")
    # Data (same semantics as train_xgb.py)
    ap.add_argument("--curated", required=True, help="Root path: local dir or s3://bucket")
    ap.add_argument("--sport", required=True, help="e.g., horse-racing")
    ap.add_argument("--date", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=11, help="Number of days ending at --date")
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0, help="0 = no downsample")
    ap.add_argument("--chunk-days", type=int, default=2, help="Feature-build chunk size")

    # Training
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    # Output
    ap.add_argument("--model-out", default=None, help="Output JSON (default: output/xgb_model_country.json)")
    args = ap.parse_args()

    dates = _daterange(args.date, args.days)
    print(f"▶ Training for date={args.date} covering {args.days} days from {dates[0]}")
    print(f"   curated={args.curated} sport={args.sport}")

    # Build features exactly like the working trainer
    df = _build_features_chunked(
        curated_root=args.curated,
        sport=args.sport,
        dates=dates,
        preoff_minutes=args.preoff_mins,
        batch_markets=args.batch_markets,
        downsample_secs=(args.downsample_secs or None),
        chunk_days=args.chunk_days,
    )

    # Add 'country' as a numeric feature (no filtering)
    df = _add_country_feature(df)

    # Ensure label present
    df = df.filter(pl.col(args.label_col).is_not_null())
    if df.is_empty():
        raise SystemExit("No labeled rows after feature build.")

    # Train
    booster, metrics, best_it, feats = _train_xgb(
        df=df,
        label_col=args.label_col,
        device=args.device,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(f"[best_iteration] {best_it}")

    out_name = args.model_out or "xgb_model_country.json"
    _save_model_json(booster, feats, out_name)


if __name__ == "__main__":
    main()
