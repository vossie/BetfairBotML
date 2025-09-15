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
from sklearn.metrics import log_loss, roc_auc_score

import pyarrow.fs as pafs
from . import features

# ---------- Optional CuPy (GPU arrays). Verify CUDA runtime is actually available ----------
try:
    import cupy as cp  # pip install cupy-cuda12x  (or cupy-cuda11x)
    try:
        _ = cp.cuda.runtime.getVersion()  # confirms CUDA runtime is visible
        _HAVE_CUPY = True
    except Exception:
        cp = None  # type: ignore
        _HAVE_CUPY = False
except Exception:
    cp = None  # type: ignore
    _HAVE_CUPY = False


# --------------------------- helpers ---------------------------

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


def _xgb_params(n_classes: int, learning_rate: float, max_depth: int, n_estimators: int, use_gpu: bool) -> Dict:
    """XGBoost >=2.0 uses device='cuda' with tree_method='hist' for GPU."""
    params: Dict = {
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "n_estimators": n_estimators,
        "random_state": 42,
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
        if "label" in lc or "target" in lc:  # extra guard
            continue
        if dt.is_numeric():
            cols.append(c)
    if not cols:
        raise RuntimeError("No numeric feature columns found for XGBoost.")
    return cols


def _check_minio(curated_root: str):
    """Fail fast if the curated root/bucket is not reachable with current env."""
    try:
        fs, path = pafs.FileSystem.from_uri(curated_root)
        info = fs.get_file_info([path])[0]
        if info.type == pafs.FileType.NotFound:
            print(f"ERROR: Curated root not found: {curated_root}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ MinIO/S3 reachable at {curated_root}")
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
    max_attempts: int = 5,
    base_delay: float = 1.5,
    jitter: float = 0.25,
) -> Tuple[pl.DataFrame, int]:
    """
    Exponential backoff with jitter. Retries ONLY on transient S3/MinIO errors.
    On recovery, prints a clear 'Recovered after N attempts' message.
    """
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


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="GPU-accelerated XGBoost trainer over curated Betfair features (chunked + resilient S3 retry).")
    # Source
    ap.add_argument("--curated", required=True, help="s3://bucket[/prefix] or /local/path")
    ap.add_argument("--sport", required=True, help="e.g. horse-racing")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="anchor date YYYY-MM-DD (treated as END of window)")
    ap.add_argument("--days", type=int, default=1, help="number of days back from --date (inclusive)")

    # Feature builder knobs (match run_train.py)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)

    # Chunking to ease MinIO pressure
    ap.add_argument("--chunk-days", type=int, default=2, help="number of days per S3 scan batch (smaller = gentler on MinIO)")

    # Model knobs
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--model-out", default="xgb_model.json")

    args = ap.parse_args()

    # Sanity check MinIO/S3 path first (catches endpoint/region issues early)
    _check_minio(args.curated)

    dates = _daterange(args.date, args.days)
    print(f"Building features from curated={args.curated}, sport={args.sport}, dates={dates[0]}..{dates[-1]} (chunk_days={args.chunk_days})")

    # Build features in chunks with resilient retry/backoff
    df_parts: List[pl.DataFrame] = []
    total_raw = 0
    for idx, dchunk in enumerate(_chunks(dates, args.chunk_days), 1):
        print(f"  • chunk {idx}: {dchunk[0]}..{dchunk[-1]}")
        df_c, raw_c = _build_features_with_retry(
            curated_root=args.curated,
            sport=args.sport,
            dates=dchunk,
            preoff_minutes=args.preoff_mins,
            batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
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

    # Build feature arrays (float32). We'll move to CuPy if GPU is available.
    Xtr_np = train_df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    ytr = train_df.select(label_col).to_numpy().ravel()
    Xva_np = valid_df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    yva = valid_df.select(label_col).to_numpy().ravel()

    # Decide GPU usage; require CuPy so BOTH train/val live on GPU to avoid device mismatch.
    use_gpu = _has_nvidia_gpu() and _HAVE_CUPY

    # Convert BOTH training and validation features to CuPy when using GPU; otherwise keep NumPy.
    if use_gpu:
        Xtr = cp.asarray(Xtr_np, dtype=cp.float32)
        Xva = cp.asarray(Xva_np, dtype=cp.float32)
    else:
        Xtr, Xva = Xtr_np, Xva_np

    # Binary vs multi-class
    n_classes = int(pl.Series(ytr).unique().len())
    params = _xgb_params(
        n_classes=n_classes,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        use_gpu=use_gpu,
    )

    # Announce device choice
    device = params["device"]
    print(f"🚀 Training with XGBoost on {device.upper()} (tree_method={params['tree_method']})")

    clf = xgb.XGBClassifier(**params)
    clf.fit(Xtr, ytr)

    # Metrics
    if n_classes > 2:
        pva = clf.predict_proba(Xva)
        if use_gpu and isinstance(pva, cp.ndarray):
            pva = pva.get()
        metrics = {"mlogloss": float(log_loss(yva, pva))}
    else:
        pva = clf.predict_proba(Xva)[:, 1]
        if use_gpu and isinstance(pva, cp.ndarray):
            pva = pva.get()
        pva = np.clip(pva, 1e-6, 1 - 1e-6)
        metrics = {"logloss": float(log_loss(yva, pva))}
        try:
            metrics["auc"] = float(roc_auc_score(yva, pva))
        except Exception:
            pass

    print("Metrics:", metrics)

    # Save model
    clf.save_model(args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
