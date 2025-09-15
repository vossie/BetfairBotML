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


# ---------------------------- utils ----------------------------

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


_RETRY_MATCHES = (
    "curlCode: 7",
    "curlCode: 6",
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
    if not is_s3:
        return features.build_features_streaming(
            curated_root, sport, dates, preoff_minutes, batch_markets, downsample_secs
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
                print(
                    f"✅ Recovered: MinIO/S3 responded on attempt {attempt} after ~{total_wait:.1f}s of backoff."
                )
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


def _binwise_report(
    booster: xgb.Booster,
    valid_df: pl.DataFrame,
    Xva_np: np.ndarray,
    yva: np.ndarray,
    bins: List[int] = [0, 30, 60, 90, 120, 180],
):
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
            print(
                f"[{lo:>3}-{hi:>3}] logloss={log_loss(yva[m], p):.4f} auc={roc_auc_score(yva[m], p):.3f} n={m.sum()}"
            )
        except Exception:
            print(f"[{lo:>3}-{hi:>3}] logloss={log_loss(yva[m], p):.4f} n={m.sum()}")


def _clean_labels_binary(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
    df2 = df.filter(pl.col(label_col).is_not_null() & pl.col(label_col).is_finite())
    # keep only {0,1}
    df2 = df2.filter(pl.col(label_col).is_in([0, 1]))
    return df2


def _split_train_valid_time(df_feat: pl.DataFrame, label_col: str, valid_frac: float = 0.2):
    if "ts" in df_feat.columns:
        df_feat = df_feat.sort("ts")
    elif "publishTimeMs" in df_feat.columns:
        df_feat = df_feat.sort("publishTimeMs")
    n = df_feat.height
    n_valid = max(1, int(n * valid_frac))
    n_train = n - n_valid
    return df_feat.slice(0, n_train), df_feat.slice(n_train, n_valid)


def _np_mats(df_tr: pl.DataFrame, df_va: pl.DataFrame, feature_cols: List[str], label_col: str):
    Xtr = df_tr.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    ytr = df_tr.select(label_col).to_numpy().ravel().astype(np.int32)
    Xva = df_va.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    yva = df_va.select(label_col).to_numpy().ravel().astype(np.int32)
    return Xtr, ytr, Xva, yva


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


def _sweep_train(
    df: pl.DataFrame,
    label_col: str,
    use_gpu: bool,
    n_estimators: int,
    learning_rate: float,
    early_stopping_rounds: int,
    sweep_out: Optional[str] = None,
    sample_weight_late: float = 1.0,
) -> Tuple[xgb.Booster, Dict, Dict]:
    """Run a small hyperparam sweep on df; return best booster, best metrics, best params."""
    df = _clean_labels_binary(df, label_col)
    feature_cols = _select_feature_cols(df, label_col)
    tr, va = _split_train_valid_time(df, label_col, valid_frac=0.2)
    Xtr, ytr, Xva, yva = _np_mats(tr, va, feature_cols, label_col)

    # optional horizon-based weights (favor tto<=45)
    train_weights: Optional[np.ndarray] = None
    if sample_weight_late > 1.0 and "tto_minutes" in tr.columns:
        tto = tr["tto_minutes"].to_numpy()
        w = np.ones_like(ytr, dtype=np.float32)
        w *= np.where(tto <= 45.0, sample_weight_late, 1.0)
        train_weights = w

    n_classes = int(pl.Series(ytr).unique().len())
    base_params = _xgb_base_params(n_classes, use_gpu)
    base_params["learning_rate"] = learning_rate

    grid = [
        {"max_depth": 5, "min_child_weight": 1,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 6, "min_child_weight": 5,  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 7, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 3.0, "reg_alpha": 1e-3},
        {"max_depth": 6, "min_child_weight": 1,  "subsample": 0.7, "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 1e-2},
    ]
    metric_name = "mlogloss" if n_classes > 2 else "logloss"

    results = []
    best = None
    for i, hp in enumerate(grid, 1):
        params = {**base_params, **hp}
        print(f"\n=== Sweep {i}/{len(grid)}: {hp} ===")
        print(f"   n_estimators={n_estimators}  lr={params['learning_rate']}")
        booster, metrics = _train_one_run(
            params=params,
            Xtr_np=Xtr,
            ytr=ytr,
            Xva_np=Xva,
            yva=yva,
            num_boost_round=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            metric_name=metric_name,
            train_weights=train_weights,
        )
        row = {**hp, **metrics, "best_iteration": getattr(booster, "best_iteration", None)}
        results.append(row)
        crit = metrics.get("logloss", metrics.get("mlogloss", 1e9))
        if (best is None) or (crit < best[0]):
            best = (crit, booster, params, metrics)

    # optional CSV
    if sweep_out and results:
        headers = list(results[0].keys())
        try:
            import csv
            with open(sweep_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                w.writerows(results)
            print(f"Saved sweep results to {sweep_out}")
        except Exception as e:
            print(f"WARN: failed to write sweep CSV: {e}", file=sys.stderr)

    if best is None:
        raise SystemExit("Sweep produced no model.")
    _, booster, best_params, best_metrics = best

    # report bin-wise on validation
    _binwise_report(booster, va, Xva, yva)

    print(
        f"\nBest params: {{ max_depth: {best_params['max_depth']}, min_child_weight: {best_params['min_child_weight']}, "
        f"subsample: {best_params['subsample']}, colsample_bytree: {best_params['colsample_bytree']}, "
        f"reg_lambda: {best_params['reg_lambda']}, reg_alpha: {best_params['reg_alpha']}, "
        f"learning_rate: {best_params['learning_rate']} }}"
    )
    print(f"Best metrics: {best_metrics}")
    return booster, best_metrics, best_params


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="XGBoost trainer (GPU-capable) local or MinIO/S3; chunked reads, retry, sweep, and optional dual-horizon training."
    )
    ap.add_argument("--curated", required=True, help="s3://bucket[/prefix] OR /local/path (rsynced mirror)")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--days", type=int, default=1)

    ap.add_argument("--preoff-mins", type=int, default=180, help="Max minutes pre-off to include in feature build (set 180 to allow dual-horizon split).")
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    ap.add_argument("--model-out", default="xgb_model.json", help="Used for single-model runs.")
    ap.add_argument("--sweep-out", default="sweep_results.csv")

    ap.add_argument("--dual-horizon", action="add_true", help=argparse.SUPPRESS)  # legacy guard
    ap.add_argument("--dual_horizon", action="store_true",
                    help="Train two models in one run: short (<=30 min) and long (30..preoff). Saves model_30.json and model_180.json (or with your prefix).")

    ap.add_argument("--model-prefix", default="", help="Prefix for dual-horizon outputs, e.g., 'hr_' to get hr_model_30.json / hr_model_180.json")
    ap.add_argument("--sweep-out-30", default="sweep_results_30.csv")
    ap.add_argument("--sweep-out-180", default="sweep_results_180.csv")

    ap.add_argument("--weight-late", type=float, default=1.0,
                    help="Extra weight for rows with tto_minutes<=45 (applies to each model independently; 1.0 = no extra weight).")

    args = ap.parse_args()

    # accept either flag spelling
    dual_horizon = getattr(args, "dual_horizon", False) or getattr(args, "dual-horizon", False)

    is_s3 = _is_s3_uri(args.curated)
    _check_store(args.curated)

    dates = _daterange(args.date, args.days)
    src_label = "MinIO/S3" if is_s3 else "Local FS"
    print(
        f"Building features from [{src_label}] curated={args.curated}, sport={args.sport}, dates={dates[0]}..{dates[-1]} (chunk_days={args.chunk_days})"
    )

    # Build once (with max horizon to allow splitting)
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

    # Device decision
    if args.device == "cuda":
        use_gpu = True
    elif args.device == "cpu":
        use_gpu = False
    else:
        use_gpu = _has_nvidia_gpu()

    print(
        f"[diag] xgboost={xgb.__version__}, CUDA visible={use_gpu}, CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','<unset>')}"
    )
    device_str = "CUDA" if use_gpu else "CPU"
    print(f"🚀 Training with XGBoost on {device_str} (tree_method=hist)")

    # Single-model path (no dual split)
    if not dual_horizon:
        booster, best_metrics, best_params = _sweep_train(
            df=df_feat,
            label_col=args.label_col,
            use_gpu=use_gpu,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            early_stopping_rounds=args.early_stopping_rounds,
            sweep_out=args.sweep_out,
            sample_weight_late=args.weight_late,
        )
        _binwise_report(
            booster,
            * _split_train_valid_time(_clean_labels_binary(df_feat, args.label_col), args.label_col, valid_frac=0.2)[1:],
        )  # prints bin report again on same split
        booster.save_model(args.model_out)
        print(f"Saved best model to {args.model_out}")
        return

    # Dual-horizon: split by tto_minutes
    if "tto_minutes" not in df_feat.columns:
        raise SystemExit("dual-horizon requires 'tto_minutes' feature. Ensure features.py adds it.")
    short_df = df_feat.filter(pl.col("tto_minutes") <= 30.0)
    long_df = df_feat.filter(pl.col("tto_minutes") > 30.0)

    if short_df.is_empty() or long_df.is_empty():
        raise SystemExit(
            f"dual-horizon split empty: short={short_df.height}, long={long_df.height}. "
            f"Check preoff-mins ({args.preoff_mins}) and data coverage."
        )

    print(f"\n=== Training short-horizon model (tto <= 30) : {short_df.height:,} rows ===")
    bst_short, m_short, p_short = _sweep_train(
        df=short_df,
        label_col=args.label_col,
        use_gpu=use_gpu,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        sweep_out=args.sweep_out_30,
        sample_weight_late=args.weight_late,
    )

    print(f"\n=== Training long-horizon model  (30 < tto <= {args.preoff_mins}) : {long_df.height:,} rows ===")
    bst_long, m_long, p_long = _sweep_train(
        df=long_df,
        label_col=args.label_col,
        use_gpu=use_gpu,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        sweep_out=args.sweep_out_180,
        sample_weight_late=args.weight_late,
    )

    # Save both
    prefix = args.model_prefix or ""
    out_short = f"{prefix}model_30.json"
    out_long = f"{prefix}model_180.json"
    bst_short.save_model(out_short)
    bst_long.save_model(out_long)
    print("\n=== Dual-horizon saved ===")
    print(f"  - {out_short}  (short horizon; best: {m_short})")
    print(f"  - {out_long}   (long horizon;  best: {m_long})")


if __name__ == "__main__":
    main()
