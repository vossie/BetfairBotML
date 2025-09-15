# ml/train_xgb.py
from __future__ import annotations

import argparse
import os
import sys
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import polars as pl
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.metrics import log_loss, roc_auc_score

import pyarrow.fs as pafs
from . import features


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def _daterange(end_date_str: str, days: int) -> List[str]:
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    out: List[str] = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _has_nvidia_gpu() -> bool:
    try:
        import pynvml  # noqa: F401
        return True
    except Exception:
        return os.system("nvidia-smi -L >/dev/null 2>&1") == 0


def _xgb_base_params(n_classes: int, use_gpu: bool) -> Dict:
    params: Dict = {
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "verbosity": 1,
    }
    if n_classes > 2:
        params.update({"objective": "multi:softprob", "num_class": n_classes, "eval_metric": "mlogloss"})
    else:
        params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    return params


def _select_feature_cols(df_feat: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {"marketId", "selectionId", "ts", "ts_ms", "publishTimeMs", label_col, "runnerStatus"}
    cols: List[str] = []
    # use collect_schema to avoid perf warnings
    schema = df_feat.collect_schema()
    for c, dt in zip(schema.names(), schema.dtypes()):
        if c in exclude:
            continue
        if "label" in c.lower() or "target" in c.lower():
            continue
        if dt.is_numeric():
            cols.append(c)
    if not cols:
        raise RuntimeError("No numeric feature columns found for XGBoost.")
    return cols


def _check_store(curated_root: str):
    try:
        fs, path = pafs.FileSystem.from_uri(curated_root)
        info = fs.get_file_info([path])[0]
        prefix = "MinIO/S3" if _is_s3_uri(curated_root) else "Local FS"
        if info.type == pafs.FileType.NotFound:
            print(f"ERROR: Curated root not found: {curated_root}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ {prefix} reachable at {curated_root}")
    except Exception as e:
        print(f"ERROR: Failed to reach curated root {curated_root}: {e}", file=sys.stderr)
        sys.exit(1)


_RETRY_MATCHES = ("curlCode: 7", "curlCode: 6", "NETWORK_CONNECTION", "Connection reset", "timed out", "Timeout", "RemoteDisconnected")


def _is_transient_s3_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(s in msg for s in _RETRY_MATCHES)


def _build_features_with_retry(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int,
    batch_markets: int,
    downsample_secs: Optional[int],
    is_s3: bool,
    max_attempts: int = 5,
    base_delay: float = 1.5,
    jitter: float = 0.25,
) -> Tuple[pl.DataFrame, int]:
    if not is_s3:
        return features.build_features_streaming(curated_root, sport, dates, preoff_minutes, batch_markets, downsample_secs)

    last_err: Exception | None = None
    total_wait = 0.0
    for attempt in range(1, max_attempts + 1):
        try:
            df, total = features.build_features_streaming(
                curated_root=curated_root,
                sport=sport,
                dates=dates,
                preoff_minutes=preoff_minutes,
                batch_markets=batch_markets,
                downsample_secs=downsample_secs,
            )
            if attempt > 1:
                print(f"✅ Recovered: MinIO/S3 responded on attempt {attempt} after ~{total_wait:.1f}s of backoff.")
            return df, total
        except Exception as e:
            last_err = e
            if not _is_transient_s3_error(e) or attempt == max_attempts:
                raise
            delay = min(30.0, base_delay * (1.6 ** (attempt - 1)))
            delay *= (1.0 + random.uniform(-jitter, jitter))
            total_wait += delay
            print(
                f"WARN: Feature build attempt {attempt} failed: {e}\n"
                f"      Backing off {delay:.1f}s before retry {attempt+1}/{max_attempts}...",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def _chunks(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _binwise_report(booster: xgb.Booster, valid_df: pl.DataFrame, Xva_np: np.ndarray, yva: np.ndarray, bins: List[int] = [0, 30, 60, 90, 120, 180]):
    if "tto_minutes" not in valid_df.columns:
        return
    tto = valid_df["tto_minutes"].to_numpy()
    print("\n[Horizon bins: tto_minutes]")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (tto > lo) & (tto <= hi)
        if m.sum() < 200:
            continue
        dsub = xgb.DMatrix(Xva_np[m], label=yva[m])
        p = booster.predict(dsub)
        if p.ndim == 2 and p.shape[1] > 1:
            p = p[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        try:
            print(f"[{lo:>3}-{hi:>3}] logloss={log_loss(yva[m], p):.4f} auc={roc_auc_score(yva[m], p):.3f} n={m.sum()}")
        except Exception:
            print(f"[{lo:>3}-{hi:>3}] logloss={log_loss(yva[m], p):.4f} n={m.sum()}")


def _train_one_run(
    params: Dict,
    Xtr_np: np.ndarray,
    ytr: np.ndarray,
    Xva_np: np.ndarray,
    yva: np.ndarray,
    num_boost_round: int,
    early_stopping_rounds: int,
    metric_name: str,
    train_weights: Optional[np.ndarray] = None,
) -> Tuple[xgb.Booster, Dict]:
    dtrain = xgb.DMatrix(Xtr_np, label=ytr, weight=train_weights)
    dvalid = xgb.DMatrix(Xva_np, label=yva)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "validation_0")],
        callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=False, data_name="validation_0", metric_name=metric_name)],
    )
    pva = booster.predict(dvalid)
    if params.get("objective", "").startswith("multi:"):
        metrics = {"mlogloss": float(log_loss(yva, pva))}
    else:
        if pva.ndim == 2 and pva.shape[1] > 1:
            pva = pva[:, 1]
        pva = np.clip(pva, 1e-6, 1 - 1e-6)
        metrics = {"logloss": float(log_loss(yva, pva))}
        try:
            metrics["auc"] = float(roc_auc_score(yva, pva))
        except Exception:
            pass
    return booster, metrics


def main():
    ap = argparse.ArgumentParser(description="XGBoost trainer (GPU-capable) local or MinIO/S3, chunked, retry, early stopping, small sweep, horizon-aware.")
    ap.add_argument("--curated", required=True, help="s3://bucket[/prefix] OR /local/path (rsynced mirror)")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--days", type=int, default=1)

    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--model-out", default="xgb_model.json")

    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-out", default="sweep_results.csv")

    ap.add_argument("--weight-late", type=float, default=1.0, help="extra weight for rows with tto_minutes<=45 (1.0 means no extra weight)")

    args = ap.parse_args()

    is_s3 = _is_s3_uri(args.curated)
    _check_store(args.curated)

    dates = _daterange(args.date, args.days)
    src_label = "MinIO/S3" if is_s3 else "Local FS"
    print(f"Building features from [{src_label}] curated={args.curated}, sport={args.sport}, dates={dates[0]}..{dates[-1]} (chunk_days={args.chunk_days})")

    df_parts: List[pl.DataFrame] = []
    total_raw = 0
    for idx, dchunk in enumerate(_chunks(dates, max(1, args.chunk_days)), 1):
        print(f"  • chunk {idx}: {dchunk[0]}..{dchunk[-1]}")
        df_c, raw_c = _build_features_with_retry(
            curated_root=args.curated,
            sport=args.sport,
            dates=dchunk,
            preoff_minutes=args.preoff_mins,
            batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
            is_s3=is_s3,
            max_attempts=5,
            base_delay=1.5,
            jitter=0.25,
        )
        total_raw += raw_c
        if not df_c.is_empty():
            df_parts.append(df_c)

    if not df_parts:
        raise SystemExit("No features built (empty frame). Check curated paths/date range.")

    df_feat = pl.concat(df_parts, how="vertical", rechunk=True)
    print(f"Feature rows: {df_feat.height:,} (from ~{total_raw:,} raw snapshot rows scanned)")

    if "ts" in df_feat.columns:
        df_feat = df_feat.sort("ts")
    elif "publishTimeMs" in df_feat.columns:
        df_feat = df_feat.sort("publishTimeMs")

    # -------- CLEAN LABELS & BUILD MATRICES (DROP-IN FIX) --------
    label_col = args.label_col
    if label_col not in df_feat.columns:
        raise SystemExit(f"Label column '{label_col}' not found in features.")

    n_before = df_feat.height
    df_feat = df_feat.filter(
        pl.col(label_col).is_not_null()
        & pl.col(label_col).is_finite()
    )

    unique_labels = df_feat.select(pl.col(label_col)).unique().to_series()
    if unique_labels.len() > 0 and all(l in (0, 1) for l in unique_labels.drop_nulls().to_list()):
        df_feat = df_feat.filter(pl.col(label_col).is_in([0, 1]))
    else:
        df_feat = df_feat.filter(pl.col(label_col).is_in([0, 1]))

    n_after = df_feat.height
    dropped = n_before - n_after
    if dropped > 0:
        print(f"[clean] Dropped {dropped:,} rows with invalid '{label_col}' (null/NaN/inf/not in {{0,1}}).")

    feature_cols = _select_feature_cols(df_feat, label_col)

    n = df_feat.height
    n_valid = max(1, int(n * 0.2))
    n_train = n - n_valid
    train_df = df_feat.slice(0, n_train)
    valid_df = df_feat.slice(n_train, n_valid)

    Xtr_np = train_df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    ytr = train_df.select(label_col).to_numpy().ravel().astype(np.int32)
    Xva_np = valid_df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    yva = valid_df.select(label_col).to_numpy().ravel().astype(np.int32)
    # -------------------------------------------------------------

    if args.device == "cuda":
        use_gpu = True
    elif args.device == "cpu":
        use_gpu = False
    else:
        use_gpu = _has_nvidia_gpu()

    print(f"[diag] xgboost={xgb.__version__}, CUDA visible={use_gpu}, CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','<unset>')}")
    device_str = "CUDA" if use_gpu else "CPU"
    print(f"🚀 Training with XGBoost on {device_str} (tree_method=hist)")

    train_weights: Optional[np.ndarray] = None
    if args.weight_late > 1.0 and "tto_minutes" in train_df.columns:
        tto = train_df["tto_minutes"].to_numpy()
        w = np.ones_like(ytr, dtype=np.float32)
        w *= np.where(tto <= 45.0, args.weight_late, 1.0)
        train_weights = w

    n_classes = int(pl.Series(ytr).unique().len())
    base_params = _xgb_base_params(n_classes, use_gpu)
    base_params["learning_rate"] = args.learning_rate

    if args.sweep:
        grid = [
            {"max_depth": 5, "min_child_weight": 1,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
            {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 0.0},
            {"max_depth": 7, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 1e-3},
            {"max_depth": 6, "min_child_weight": 1,  "subsample": 0.7, "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 1e-2},
        ]
    else:
        grid = [{"max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 0.0}]

    metric_name = "mlogloss" if n_classes > 2 else "logloss"

    results = []
    best = None  # (crit, booster, params, metrics)
    for i, hp in enumerate(grid, 1):
        params = {**base_params, **hp}
        print(f"\n=== Sweep {i}/{len(grid)}: {hp} ===")
        print(f"   n_estimators={args.n_estimators}  lr={params['learning_rate']}")
        booster, metrics = _train_one_run(
            params=params,
            Xtr_np=Xtr_np,
            ytr=ytr,
            Xva_np=Xva_np,
            yva=yva,
            num_boost_round=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
            metric_name=metric_name,
            train_weights=train_weights,
        )
        row = {**hp, **metrics, "best_iteration": getattr(booster, "best_iteration", None)}
        results.append(row)
        crit = metrics.get("logloss", metrics.get("mlogloss", 1e9))
        if (best is None) or (crit < best[0]):
            best = (crit, booster, params, metrics)

    if results:
        headers = list(results[0].keys())
        print("\n=== Sweep Summary ===")
        print(" | ".join(headers))
        for r in results:
            print(" | ".join(str(r.get(h)) for h in headers))
        try:
            import csv
            with open(args.sweep_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                w.writerows(results)
            print(f"Saved sweep results to {args.sweep_out}")
        except Exception as e:
            print(f"WARN: failed to write sweep CSV: {e}", file=sys.stderr)

    if best is None:
        raise SystemExit("Training failed to produce any model.")
    _, best_booster, best_params, best_metrics = best
    _binwise_report(best_booster, valid_df, Xva_np, yva)
    best_booster.save_model(args.model_out)
    print(f"\nBest params: {{ {', '.join(f'{k}: {best_params[k]}' for k in ('max_depth','min_child_weight','subsample','colsample_bytree','reg_lambda','reg_alpha','learning_rate'))} }}")
    print(f"Best metrics: {best_metrics}")
    print(f"Saved best model to {args.model_out}")


if __name__ == "__main__":
    main()
