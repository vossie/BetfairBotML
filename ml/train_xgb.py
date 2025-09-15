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

# ---------- Optional CuPy (GPU arrays). Only used to keep arrays on-GPU if available ----------
try:
    import cupy as cp  # pip install cupy-cuda12x (or cupy-cuda11x)
    try:
        _ = cp.cuda.runtime.getVersion()
        _HAVE_CUPY = True
    except Exception:
        cp = None  # type: ignore
        _HAVE_CUPY = False
except Exception:
    cp = None  # type: ignore
    _HAVE_CUPY = False


# --------------------------- helpers ---------------------------

def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def _daterange(end_date_str: str, days: int) -> List[str]:
    """Inclusive date range: [end - (days-1), end], formatted YYYY-MM-DD."""
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    out: List[str] = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _has_nvidia_gpu() -> bool:
    """Lightweight check for an NVIDIA GPU presence."""
    try:
        import pynvml  # noqa: F401
        return True
    except Exception:
        return os.system("nvidia-smi -L >/dev/null 2>&1") == 0


def _xgb_base_params(n_classes: int, use_gpu: bool) -> Dict:
    """Base params for xgboost.train; per-run hyperparams are layered on top."""
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
        params.update({
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
        })
    else:
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
    return params


def _select_feature_cols(df_feat: pl.DataFrame, label_col: str) -> List[str]:
    """Pick numeric feature columns; drop IDs/timestamps/labels & label-derived to avoid leakage."""
    exclude = {
        "marketId", "selectionId", "ts", "ts_ms", "publishTimeMs",
        label_col, "soft_target", "sum_win_in_mkt", "runnerStatus"
    }
    cols: List[str] = []
    for c, dt in zip(df_feat.columns, df_feat.dtypes):
        if c in exclude:
            continue
        lc = c.lower()
        if "label" in lc or "target" in lc:
            continue
        if dt.is_numeric():
            cols.append(c)
    if not cols:
        raise RuntimeError("No numeric feature columns found for XGBoost.")
    return cols


def _check_store(curated_root: str):
    """Fail fast if the curated root (local or s3) is not reachable with current env."""
    try:
        fs, path = pafs.FileSystem.from_uri(curated_root)
        info = fs.get_file_info([path])[0]
        if info.type == pafs.FileType.NotFound:
            print(f"ERROR: Curated root not found: {curated_root}", file=sys.stderr)
            sys.exit(1)
        prefix = "MinIO/S3" if _is_s3_uri(curated_root) else "Local FS"
        print(f"✅ {prefix} reachable at {curated_root}")
    except Exception as e:
        print(f"ERROR: Failed to reach curated root {curated_root}: {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------- resilient S3 retry ---------------------------

_RETRY_MATCHES = (
    "curlCode: 7",                 # couldn't connect
    "curlCode: 6",                 # couldn't resolve host
    "NETWORK_CONNECTION",
    "Connection reset",
    "timed out",
    "Timeout",
    "RemoteDisconnected",
)

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
    """
    For S3/MinIO: exponential backoff with jitter on transient network errors.
    For local disk: just call once (no retry needed).
    """
    if not is_s3:
        return features.build_features_streaming(
            curated_root=curated_root,
            sport=sport,
            dates=dates,
            preoff_minutes=preoff_minutes,
            batch_markets=batch_markets,
            downsample_secs=downsample_secs,
        )

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
                waited = f"{total_wait:.1f}s"
                print(f"✅ Recovered: MinIO/S3 responded on attempt {attempt} after ~{waited} of backoff.")
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


# --------------------------- chunking (reduce S3 load) ---------------------------

def _chunks(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------- training via xgboost.train ---------------------------

def _train_one_run(
    params: Dict,
    Xtr_np: np.ndarray,
    ytr: np.ndarray,
    Xva_np: np.ndarray,
    yva: np.ndarray,
    num_boost_round: int,
    early_stopping_rounds: int,
    metric_name: str,
) -> Tuple[xgb.Booster, Dict]:
    """
    Train with xgboost.train so we can always use EarlyStopping callbacks,
    regardless of sklearn wrapper support.
    """
    # DMatrix on CPU host memory is fine; device is controlled via params["device"].
    dtrain = xgb.DMatrix(Xtr_np, label=ytr)
    dvalid = xgb.DMatrix(Xva_np, label=yva)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "validation_0")],
        callbacks=[
            EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True,
                maximize=False,
                data_name="validation_0",
                metric_name=metric_name,
            )
        ],
    )

    # Predict on validation to compute metrics
    pva = booster.predict(dvalid)  # prob per class or prob for class 1 depending on objective
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


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="XGBoost trainer (GPU-capable) reading from Local or MinIO/S3, with chunked reads, retry, early stopping, and a small param sweep."
    )
    # Source (either s3://... OR local path like /data/betfair-curated)
    ap.add_argument("--curated", required=True, help="s3://bucket[/prefix] OR /local/path (rsynced mirror)")
    ap.add_argument("--sport", required=True, help="e.g. horse-racing")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="anchor date YYYY-MM-DD (treated as END of window)")
    ap.add_argument("--days", type=int, default=1, help="number of days back from --date (inclusive)")

    # Feature builder knobs (match run_train.py)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)

    # Chunking to ease MinIO pressure
    ap.add_argument("--chunk-days", type=int, default=2, help="number of days per scan batch (smaller = gentler on MinIO).")

    # Device control
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="Force device for XGBoost. 'auto' = use GPU if available.")

    # Model knobs / training control
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=3000, help="upper bound trees for early stopping")
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--model-out", default="xgb_model.json")

    # Small param sweep toggles
    ap.add_argument("--sweep", action="store_true", help="run a small hyperparameter sweep and pick best by logloss")
    ap.add_argument("--sweep-out", default="sweep_results.csv")

    args = ap.parse_args()

    # Detect storage type and preflight
    is_s3 = _is_s3_uri(args.curated)
    _check_store(args.curated)

    dates = _daterange(args.date, args.days)
    src_label = "MinIO/S3" if is_s3 else "Local FS"
    print(f"Building features from [{src_label}] curated={args.curated}, sport={args.sport}, dates={dates[0]}..{dates[-1]} (chunk_days={args.chunk_days})")

    # Build features in chunks with resilient retry/backoff
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

    # Sort by time to do a time-based split
    if "ts" in df_feat.columns:
        df_feat = df_feat.sort("ts")
    elif "publishTimeMs" in df_feat.columns:
        df_feat = df_feat.sort("publishTimeMs")

    label_col = args.label_col
    if label_col not in df_feat.columns:
        raise SystemExit(f"Label column '{label_col}' not found in features.")

    feature_cols = _select_feature_cols(df_feat, label_col)

    # Time-based split: last 20% as validation
    n = df_feat.height
    n_valid = max(1, int(n * 0.2))
    n_train = n - n_valid
    train_df = df_feat.slice(0, n_train)
    valid_df = df_feat.slice(n_train, n_valid)

    # Build feature arrays (float32). We'll keep NumPy on host; GPU use is controlled by params['device'].
    Xtr_np = train_df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    ytr = train_df.select(label_col).to_numpy().ravel()
    Xva_np = valid_df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    yva = valid_df.select(label_col).to_numpy().ravel()

    # Decide device: user override or auto
    if args.device == "cuda":
        use_gpu = True
    elif args.device == "cpu":
        use_gpu = False
    else:
        use_gpu = _has_nvidia_gpu()

    # Diagnostics
    print(f"[diag] xgboost={xgb.__version__}, CUDA visible={use_gpu}, CuPy={_HAVE_CUPY}, CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','<unset>')}")
    device_str = "CUDA" if use_gpu else "CPU"
    print(f"🚀 Training with XGBoost on {device_str} (tree_method=hist)")

    # Binary vs multi-class
    n_classes = int(pl.Series(ytr).unique().len())
    base_params = _xgb_base_params(n_classes, use_gpu)
    base_params["learning_rate"] = args.learning_rate

    # Hyperparameter candidates
    if args.sweep:
        grid = [
            {"max_depth": 5, "min_child_weight": 1,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
            {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 0.0},
            {"max_depth": 7, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 1e-3},
            {"max_depth": 6, "min_child_weight": 1,  "subsample": 0.7, "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 1e-2},
        ]
    else:
        grid = [
            {"max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 0.0},
        ]

    # Metric name for EarlyStopping
    metric_name = "mlogloss" if n_classes > 2 else "logloss"

    # Run sweep
    results = []
    best = None  # (crit, booster, params, metrics)
    for i, hp in enumerate(grid, 1):
        params = {
            **base_params,
            "max_depth": hp["max_depth"],
            "min_child_weight": hp["min_child_weight"],
            "subsample": hp["subsample"],
            "colsample_bytree": hp["colsample_bytree"],
            "reg_lambda": hp["reg_lambda"],
            "reg_alpha": hp["reg_alpha"],
        }

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
        )

        print(f"→ Metrics: {metrics}")
        row = {**hp, **metrics, "best_iteration": booster.best_iteration if hasattr(booster, "best_iteration") else None}
        results.append(row)

        crit = metrics.get("logloss", metrics.get("mlogloss", 1e9))
        if (best is None) or (crit < best[0]):
            best = (crit, booster, params, metrics)

    # Print sweep summary
    if results:
        headers = list(results[0].keys())
        print("\n=== Sweep Summary ===")
        print(" | ".join(headers))
        for r in results:
            print(" | ".join(str(r.get(h)) for h in headers))
        # Optionally write CSV
        try:
            import csv
            with open(args.sweep_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                w.writerows(results)
            print(f"Saved sweep results to {args.sweep_out}")
        except Exception as e:
            print(f"WARN: failed to write sweep CSV: {e}", file=sys.stderr)

    # Save best model
    if best is None:
        raise SystemExit("Training failed to produce any model.")
    _, best_booster, best_params, best_metrics = best
    best_booster.save_model(args.model_out)
    print(f"\nBest params: { {k: best_params[k] for k in ('max_depth','min_child_weight','subsample','colsample_bytree','reg_lambda','reg_alpha','learning_rate')} }")
    print(f"Best metrics: {best_metrics}")
    print(f"Saved best model to {args.model_out}")


if __name__ == "__main__":
    main()
