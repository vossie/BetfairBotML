# ml/train_xgb.py
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import polars as pl
import xgboost as xgb

from xgb.ml import features

# ----------------------------- output helpers -----------------------------

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_model(booster: xgb.Booster, filename: str) -> Path:
    out_path = OUTPUT_DIR / filename
    booster.save_model(str(out_path))
    print(f"Saved model to {out_path}")
    return out_path

def save_csv(df: pl.DataFrame, filename: str) -> Path:
    out_path = OUTPUT_DIR / filename
    df.write_csv(str(out_path))
    print(f"Saved CSV to {out_path}")
    return out_path


# ----------------------------- utils -----------------------------

def _daterange(end_date_str: str, days: int) -> List[str]:
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
        "publishTimeMs", label_col, "runnerStatus",
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


def _device_params(device: str) -> Tuple[Dict, str]:
    if device == "auto":
        try:
            import cupy as _cp  # noqa: F401
            device = "cuda"
        except Exception:
            device = "cpu"
    if device == "cuda":
        return {"device": "cuda", "tree_method": "hist"}, "🚀 Training with XGBoost on CUDA (tree_method=hist)"
    else:
        return {"device": "cpu", "tree_method": "hist"}, "🚀 Training with XGBoost on CPU (tree_method=hist)"


def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


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
        n_pos = np.sum(pos)
        n_neg = len(p) - n_pos
        auc = 0.5 if (n_pos == 0 or n_neg == 0) else (np.sum(ranks[pos]) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {"logloss": float(logloss), "auc": float(auc)}


def _binwise_report(df: pl.DataFrame, y: np.ndarray, p: np.ndarray, bins: List[int]) -> None:
    if "tto_minutes" not in df.columns:
        return
    edges = bins
    labels = [f"{edges[i]:02d}-{edges[i+1]:02d}" for i in range(len(edges) - 1)]
    expr = (
        pl.when((pl.col("tto_minutes") > edges[0]) & (pl.col("tto_minutes") <= edges[1]))
        .then(pl.lit(labels[0]))
    )
    for i in range(1, len(labels)):
        lo, hi = edges[i], edges[i + 1]
        expr = expr.when((pl.col("tto_minutes") > lo) & (pl.col("tto_minutes") <= hi)).then(pl.lit(labels[i]))
    expr = expr.otherwise(pl.lit(None)).alias("tto_bin")
    tmp = df.with_columns(expr)
    print("\n[Horizon bins: tto_minutes]")
    for lab in labels:
        mask = (tmp.get_column("tto_bin") == lab).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
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
        msg = str(e)
        print(f"WARN: First feature build attempt failed: {msg}")
        time.sleep(3.0)
        df, raw = features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=dates,
            preoff_minutes=preoff_minutes,
            batch_markets=batch_markets,
            downsample_secs=downsample_secs,
        )
        print("✅ Recovered after transient data access error; continuing.")
        return df, raw


# ----------------------------- training -----------------------------

def _train_one_run(
    df: pl.DataFrame,
    label_col: str,
    device: str,
    n_estimators: int,
    learning_rate: float,
    early_stopping_rounds: int,
) -> Tuple[xgb.Booster, Dict[str, float], int]:
    feat_cols = _select_feature_cols(df, label_col)
    y = df[label_col].to_numpy().astype(np.float32)

    n = df.height
    split_idx = int(n * 0.8)
    train_df = df[:split_idx]
    valid_df = df[split_idx:]

    Xtr = _to_numpy(train_df, feat_cols)
    Xva = _to_numpy(valid_df, feat_cols)
    ytr = y[:split_idx]
    yva = y[split_idx:]

    params, banner = _device_params(device)
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": learning_rate,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    })
    print(banner)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xva, label=yva)
    evals = [(dtrain, "train"), (dvalid, "valid")]

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        verbose_eval=False,
    )

    pva = booster.predict(dvalid)
    metrics = _metrics_binary(yva, pva)
    _binwise_report(valid_df, yva, pva, bins=[0, 30, 60, 90, 120, 180])

    best_it = booster.best_iteration if booster.best_iteration is not None else n_estimators
    return booster, metrics, best_it


def _train_with_params(
    df: pl.DataFrame,
    label_col: str,
    base_params: Dict,
    device: str,
    n_estimators: int,
    learning_rate: float,
    early_stopping_rounds: int,
) -> Tuple[xgb.Booster, Dict[str, float], int]:
    feat_cols = _select_feature_cols(df, label_col)
    y = df[label_col].to_numpy().astype(np.float32)

    n = df.height
    split_idx = int(n * 0.8)
    train_df = df[:split_idx]
    valid_df = df[split_idx:]

    Xtr = _to_numpy(train_df, feat_cols)
    Xva = _to_numpy(valid_df, feat_cols)
    ytr = y[:split_idx]
    yva = y[split_idx:]

    dev_params, banner = _device_params(device)
    params = {**dev_params, **base_params}
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": learning_rate,
    })
    print(f"{banner} | n_estimators={n_estimators} lr={learning_rate}")

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xva, label=yva)
    evals = [(dtrain, "train"), (dvalid, "valid")]

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        verbose_eval=False,
    )

    pva = booster.predict(dvalid)
    metrics = _metrics_binary(yva, pva)
    _binwise_report(valid_df, yva, pva, bins=[0, 30, 60, 90, 120, 180])

    best_it = booster.best_iteration if booster.best_iteration is not None else n_estimators
    return booster, metrics, best_it


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost on curated Betfair features (local FS or MinIO/S3).")

    # Data
    ap.add_argument("--curated", required=True, help="s3://bucket or /local/path")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", default=None, help="End date (YYYY-MM-DD). If omitted, uses provided range logic.")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    # Model / training
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    # Output
    ap.add_argument("--model-out", default=None, help="Single-horizon output path; default ../output/xgb_model.json")
    ap.add_argument("--model-prefix", default="", help="Prefix for saved model files")

    # Sweep
    ap.add_argument("--sweep", action="store_true", help="Run a small hyperparameter sweep.")
    ap.add_argument("--sweep-out", default="sweep_results.csv")

    # Dual horizon
    ap.add_argument("--dual-horizon", action="store_true", help="Train 2 models at preoff-mins=30 and 180.")

    args = ap.parse_args()

    # Resolve dates and print plan
    if args.date:
        dates = _daterange(args.date, args.days)
        print(f"Building features from curated={args.curated}, sport={args.sport}, dates={dates[0]}..{dates[-1]}")
    else:
        dates = []
        print(f"Building features from curated={args.curated}, sport={args.sport}, (dates inferred)")

    # Feature builder
    def build_all_features(preoff_minutes: int) -> pl.DataFrame:
        df_parts: List[pl.DataFrame] = []
        total_raw = 0
        if not dates:
            df, raw = _build_features_with_retry(
                args.curated, args.sport, dates, preoff_minutes, args.batch_markets, args.downsample_secs or None
            )
            total_raw += raw
            if not df.is_empty():
                df_parts.append(df)
        else:
            for i in range(0, len(dates), args.chunk_days):
                dchunk = dates[i:i + args.chunk_days]
                print(f"  • chunk {i//args.chunk_days + 1}: {dchunk[0]}..{dchunk[-1]}")
                df_c, raw_c = _build_features_with_retry(
                    args.curated, args.sport, dchunk, preoff_minutes, args.batch_markets, args.downsample_secs or None
                )
                total_raw += raw_c
                if not df_c.is_empty():
                    df_parts.append(df_c)
        if not df_parts:
            raise SystemExit("No features after streaming build.")
        df_all = pl.concat(df_parts, how="vertical", rechunk=True)
        print(f"Feature rows: {df_all.height} (from ~{df_all.height} raw snapshot rows scanned)")
        return df_all

    # ----------------- Single horizon -----------------
    if not args.dual_horizon:
        df_feat = build_all_features(args.preoff_mins).filter(pl.col(args.label_col).is_not_null())
        booster, metrics, best_iter = _train_one_run(
            df=df_feat,
            label_col=args.label_col,
            device=args.device,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        print(f"Metrics: {metrics}")
        out_name = args.model_out if args.model_out else f"{args.model_prefix}xgb_model.json"
        save_model(booster, out_name)
        return

    # ----------------- Dual horizon (30 & 180) -----------------
    print(f"\n==== Training (preoff-mins=30) → model_30.json ====")
    df_30 = build_all_features(30).filter(pl.col(args.label_col).is_not_null())

    print(f"\n==== Training (preoff-mins=180) → model_180.json ====")
    df_180 = build_all_features(180).filter(pl.col(args.label_col).is_not_null())

    def run_sweep(df: pl.DataFrame, tag_out: str) -> Tuple[xgb.Booster, Dict[str, float]]:
        search_space = [
            {"max_depth": 5, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
            {"max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 0.0},
            {"max_depth": 7, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 0.001},
            {"max_depth": 6, "min_child_weight": 1, "subsample": 0.7, "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.01},
        ]
        rows = []
        best_bst, best_m, best_it, best_hp = None, None, None, None
        for i, hp in enumerate(search_space, 1):
            print(f"\n=== Sweep {i}/{len(search_space)}: {hp} ===")
            bst, m, it = _train_with_params(
                df=df,
                label_col=args.label_col,
                base_params=hp,
                device=args.device,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            rows.append({**hp, "logloss": m["logloss"], "auc": m["auc"], "best_iteration": it})
            if (best_m is None) or (m["logloss"] < best_m["logloss"]):
                best_bst, best_m, best_it, best_hp = bst, m, it, hp
        df_rows = pl.DataFrame(rows).sort("logloss")
        save_csv(df_rows, tag_out)
        print("\n=== Sweep Summary ===")
        print(df_rows.to_string())
        print(f"\nBest params: {best_hp} \nBest metrics: {best_m}")
        return best_bst, best_m

    if args.sweep:
        bst_short, m_short = run_sweep(df_30, "sweep_results_30.csv")
        bst_long,  m_long  = run_sweep(df_180, "sweep_results_180.csv")
    else:
        bst_short, m_short, _ = _train_one_run(
            df=df_30, label_col=args.label_col, device=args.device,
            n_estimators=args.n_estimators, learning_rate=args.learning_rate,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        bst_long,  m_long,  _ = _train_one_run(
            df=df_180, label_col=args.label_col, device=args.device,
            n_estimators=args.n_estimators, learning_rate=args.learning_rate,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        print(f"\n[30] Metrics: {m_short}")
        print(f"[180] Metrics: {m_long}")

    prefix = args.model_prefix or ""
    save_model(bst_short, f"{prefix}model_30.json")
    save_model(bst_long,  f"{prefix}model_180.json")
    print("\n=== Dual-horizon saved ===")
    print(f"  - {(OUTPUT_DIR / (prefix + 'model_30.json'))}  (short horizon; best: {m_short})")
    print(f"  - {(OUTPUT_DIR / (prefix + 'model_180.json'))}   (long horizon;  best: {m_long})")


if __name__ == "__main__":
    main()
