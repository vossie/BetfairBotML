# ml/train_xgb_country.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import polars as pl
import xgboost as xgb

from . import features

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    exclude = {"marketId","selectionId","ts","ts_ms","publishTimeMs",label_col,"runnerStatus"}
    cols: List[str] = []
    schema = df.collect_schema()
    for name, dtype in zip(schema.names(), schema.dtypes()):
        if name in exclude: continue
        if "label" in name.lower() or "target" in name.lower(): continue
        if dtype.is_numeric(): cols.append(name)
    if not cols:
        raise RuntimeError("No numeric feature columns found.")
    return cols

def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)

def _device_params(device: str) -> Tuple[Dict, str]:
    if device == "auto":
        try:
            import cupy as _cp  # noqa
            device = "cuda"
        except Exception:
            device = "cpu"
    if device == "cuda":
        return {"device":"cuda","tree_method":"hist"}, "🚀 XGB on CUDA"
    return {"device":"cpu","tree_method":"hist"}, "🚀 XGB on CPU"

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
        n_pos = np.sum(pos); n_neg = len(p) - n_pos
        auc = 0.5 if (n_pos == 0 or n_neg == 0) else (np.sum(ranks[pos]) - n_pos*(n_pos-1)/2) / (n_pos*n_neg)
    return {"logloss": float(logloss), "auc": float(auc)}

def _binwise_report(df: pl.DataFrame, y: np.ndarray, p: np.ndarray, bins: List[int]) -> None:
    if "tto_minutes" not in df.columns: return
    edges = bins
    labels = [f"{edges[i]:02d}-{edges[i+1]:02d}" for i in range(len(edges)-1)]
    expr = (pl.when((pl.col("tto_minutes") > edges[0]) & (pl.col("tto_minutes") <= edges[1])).then(pl.lit(labels[0])))
    for i in range(1, len(labels)):
        lo, hi = edges[i], edges[i+1]
        expr = expr.when((pl.col("tto_minutes") > lo) & (pl.col("tto_minutes") <= hi)).then(pl.lit(labels[i]))
    expr = expr.otherwise(pl.lit(None)).alias("tto_bin")
    tmp = df.with_columns(expr)
    print("\n[Horizon bins]")
    for lab in labels:
        mask = (tmp.get_column("tto_bin") == lab).to_numpy()
        n = int(mask.sum())
        if n == 0: continue
        m = _metrics_binary(y[mask], p[mask])
        print(f"[{lab}] logloss={m['logloss']:.4f} auc={m['auc']:.3f} n={n}")

def _build_features_with_retry(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int,
    batch_markets: int,
    downsample_secs: Optional[int],
) -> Tuple[pl.DataFrame, int]:
    try:
        return features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=dates,
            preoff_minutes=preoff_minutes,
            batch_markets=batch_markets,
            downsample_secs=downsample_secs,
        )
    except Exception as e:
        print(f"WARN: feature build failed once: {e}")
        return features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=dates,
            preoff_minutes=preoff_minutes,
            batch_markets=batch_markets,
            downsample_secs=downsample_secs,
        )

def _train_one(df: pl.DataFrame, label_col: str, device: str,
               n_estimators: int, learning_rate: float,
               early_stopping_rounds: int) -> Tuple[xgb.Booster, Dict[str,float], int]:
    feats = _select_feature_cols(df, label_col)
    y = df[label_col].to_numpy().astype(np.float32)
    n = df.height; split = int(n*0.8)
    tr, va = df[:split], df[split:]
    Xtr = _to_numpy(tr, feats); Xva = _to_numpy(va, feats)
    ytr = y[:split]; yva = y[split:]

    dev_params, banner = _device_params(device)
    params = {
        **dev_params,
        "objective":"binary:logistic",
        "eval_metric":"logloss",
        "learning_rate":learning_rate,
        "max_depth":6, "min_child_weight":1,
        "subsample":0.8, "colsample_bytree":0.8,
        "reg_lambda":1.0, "reg_alpha":0.0,
    }
    print(banner)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    bst = xgb.train(params, dtr, num_boost_round=n_estimators,
                    evals=[(dtr,"train"),(dva,"valid")],
                    early_stopping_rounds=early_stopping_rounds or None,
                    verbose_eval=False)
    pva = bst.predict(dva)
    m = _metrics_binary(yva, pva)
    _binwise_report(va, yva, pva, bins=[0,30,60,90,120,180])
    best_it = bst.best_iteration if bst.best_iteration is not None else n_estimators
    return bst, m, best_it

def save_model(booster: xgb.Booster, filename: str) -> Path:
    out = OUTPUT_DIR / filename
    booster.save_model(str(out))
    print(f"Saved model to {out}")
    return out

def main():
    ap = argparse.ArgumentParser("Train XGBoost by country (local FS or S3), matching train_xgb flow.")
    # Data
    ap.add_argument("--curated", required=True, help="Root: /local/path OR s3://bucket")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=11, help="Number of days ending at --date")
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)
    ap.add_argument("--countries", nargs="+", required=True, help="Country codes to include (e.g. GB IE FR)")

    # Training
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    # Output
    ap.add_argument("--model-out", default=None, help="Output file (default: output/xgb_model_country.json)")
    args = ap.parse_args()

    dates = _daterange(args.date, args.days)
    print(f"▶ Training for date={args.date} covering {args.days} days from {dates[0]}")
    print(f"   curated={args.curated} sport={args.sport} countries={args.countries}")

    # Build features the SAME way as train_xgb.py, then filter by country.
    df_parts: List[pl.DataFrame] = []
    total_raw = 0
    for i in range(0, len(dates), args.chunk_days):
        chunk = dates[i:i+args.chunk_days]
        print(f"  • chunk {i//args.chunk_days + 1}: {chunk[0]}..{chunk[-1]}")
        df_c, raw_c = _build_features_with_retry(
            curated_root=args.curated,
            sport=args.sport,
            dates=chunk,
            preoff_minutes=args.preoff_mins,
            batch_markets=args.batch_markets,
            downsample_secs=args.downsample_secs or None,
        )
        total_raw += raw_c
        if not df_c.is_empty():
            df_parts.append(df_c)

    if not df_parts:
        raise SystemExit("No features built.")

    df = pl.concat(df_parts, how="vertical", rechunk=True)

    # ---- Country filter (supports either 'country' or 'eventCountryCode') ----
    countries = set([c.upper() for c in args.countries])
    if "country" in df.columns:
        df = df.filter(pl.col("country").str.to_uppercase().is_in(list(countries)))
    elif "eventCountryCode" in df.columns:
        df = df.filter(pl.col("eventCountryCode").str.to_uppercase().is_in(list(countries)))
    else:
        print("WARN: No country column found; proceeding without country filter.")

    # Keep rows with label
    df = df.filter(pl.col(args.label_col).is_not_null())
    if df.is_empty():
        raise SystemExit("No labeled rows after country filter.")

    # Train
    booster, metrics, best_it = _train_one(
        df=df,
        label_col=args.label_col,
        device=args.device,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(f"Metrics: {metrics}")

    out_name = args.model_out or f"xgb_model_{'_'.join(sorted(countries))}.json"
    save_model(booster, out_name)

if __name__ == "__main__":
    main()
