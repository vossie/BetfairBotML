# ml/train_xgb_country.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import polars as pl
import xgboost as xgb

from . import features

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Utilities ----------------

def _daterange(end_date_str: str, days: int) -> List[str]:
    from datetime import datetime, timedelta
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start = end - timedelta(days=days - 1)
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _select_feature_cols(df: pl.DataFrame, label_col: str) -> List[str]:
    exclude = {
        "sport", "marketId", "selectionId",
        "ts", "ts_ms", "publishTimeMs", label_col,
        "runnerStatus", "countryCode"
    }
    cols: List[str] = []
    for name, dtype in df.schema.items():
        if name in exclude:
            continue
        lname = name.lower()
        if "label" in lname or "target" in lname:
            continue
        if dtype.is_numeric():
            cols.append(name)
    if not cols:
        raise RuntimeError("No numeric feature columns found.")
    return cols


def _to_numpy(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)


def _device_params(device: str) -> Tuple[Dict, str]:
    dev = device
    if dev == "auto":
        try:
            import cupy as _  # noqa: F401
            dev = "cuda"
        except Exception:
            dev = "cpu"
    if dev == "cuda":
        return {"device": "cuda", "tree_method": "hist"}, "CUDA"
    else:
        return {"device": "cpu", "tree_method": "hist"}, "CPU"


# ---------------- Country encoding ----------------

def topk_countries(df: pl.DataFrame, k: int) -> List[str]:
    if "countryCode" not in df.columns:
        return []
    vc = (
        df.select("countryCode")
          .drop_nulls()
          .group_by("countryCode")
          .len()
          .sort("len", descending=True)
          .head(k)
    )
    return vc["countryCode"].to_list()


def add_country_onehots(df: pl.DataFrame, vocab: List[str], other_token: str = "__OTHER__") -> pl.DataFrame:
    if not vocab or "countryCode" not in df.columns:
        return df
    df2 = df.with_columns(
        pl.when(pl.col("countryCode").is_in(vocab))
          .then(pl.col("countryCode"))
          .otherwise(other_token)
          .alias("cc_bucket")
    )
    # one-hots
    for c in vocab + [other_token]:
        df2 = df2.with_columns(
            pl.when(pl.col("cc_bucket") == c).then(1).otherwise(0).cast(pl.Int8).alias(f"cc__{c}")
        )
    return df2


# ---------------- Training ----------------

def train(
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
    split_idx = max(1, int(n * 0.8))
    tr, va = df[:split_idx], df[split_idx:]
    Xtr, Xva = _to_numpy(tr, feat_cols), _to_numpy(va, feat_cols)
    ytr, yva = y[:split_idx], y[split_idx:]

    dev_params, banner = _device_params(device)
    params = {
        **dev_params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": learning_rate,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    print(f"[{banner}] n_estimators={n_estimators} lr={learning_rate}")
    dtr, dva = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xva, label=yva)
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=early_stopping_rounds or None,
        verbose_eval=False,
    )

    p = bst.predict(dva)
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)
    logloss = float(-np.mean(yva * np.log(p_clip) + (1 - yva) * np.log(1 - p_clip)))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yva, p)) if len(yva) and len(set(yva)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    best_it = bst.best_iteration if bst.best_iteration is not None else n_estimators

    # Stratified validation metrics by country bucket
    if "cc_bucket" in va.columns and va.height:
        try:
            buckets = va["cc_bucket"].to_list()
            unique = sorted(set(buckets))
            print("\nValidation metrics by country:")
            print(f"{'country':<10} {'n':>6} {'logloss':>12} {'auc':>8}")
            try:
                from sklearn.metrics import roc_auc_score as _auc
            except Exception:
                _auc = None
            for c in unique:
                idx = [i for i, b in enumerate(buckets) if b == c]
                if not idx:
                    continue
                y_sub = yva[idx]
                p_sub = p[idx]
                p_sub_c = np.clip(p_sub, eps, 1 - eps)
                ll = float(-np.mean(y_sub * np.log(p_sub_c) + (1 - y_sub) * np.log(1 - p_sub_c)))
                if _auc is not None and len(set(y_sub)) > 1:
                    auc_c = float(_auc(y_sub, p_sub))
                else:
                    auc_c = float("nan")
                print(f"{c:<10} {len(idx):>6} {ll:>12.6f} {auc_c:>8.3f}")
            print()
        except Exception as e:
            print(f"[warn] could not compute per-country metrics: {e}")

    return bst, {"logloss": logloss, "auc": auc}, best_it


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost with country one-hots")
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--preoff-mins", type=int, default=180)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)

    ap.add_argument("--country-topk", type=int, default=8, help="Number of most frequent countries to one-hot; rest -> __OTHER__.")
    ap.add_argument("--model-out", default="xgb_country.json")
    ap.add_argument("--meta-out", default="xgb_country_meta.json")

    args = ap.parse_args()

    # Resolve dates
    if args.date:
        dates = _daterange(args.date, args.days)
    else:
        dates = []

    # Build (possibly chunked) features
    def build(preoff_minutes: int) -> pl.DataFrame:
        parts: List[pl.DataFrame] = []
        if not dates:
            df, _ = features.build_features_streaming(
                args.curated, args.sport, dates, preoff_minutes, args.batch_markets, args.downsample_secs or None
            )
            if not df.is_empty():
                parts.append(df)
        else:
            for i in range(0, len(dates), args.chunk_days):
                dchunk = dates[i:i + args.chunk_days]
                df_c, _ = features.build_features_streaming(
                    args.curated, args.sport, dchunk, preoff_minutes, args.batch_markets, args.downsample_secs or None
                )
                if not df_c.is_empty():
                    parts.append(df_c)
        if not parts:
            print("No features produced.")
            return pl.DataFrame([])
        return pl.concat(parts, how="vertical", rechunk=True)

    df = build(args.preoff_mins)
    if df.is_empty():
        raise SystemExit("No features produced.")

    # Country encoding (top-K + __OTHER__)
    vocab = topk_countries(df, args.country_topk)
    df_enc = add_country_onehots(df, vocab, other_token="__OTHER__")

    # Filter to rows with label
    df_enc = df_enc.filter(pl.col(args.label_col).is_not_null())

    # Train
    bst, metrics, best_it = train(
        df_enc, args.label_col, args.device, args.n_estimators, args.learning_rate, args.early_stopping_rounds
    )

    out_model = OUTPUT_DIR / args.model_out
    bst.save_model(str(out_model))

    meta = {"country_vocab": vocab, "other_token": "__OTHER__"}
    out_meta = OUTPUT_DIR / args.meta_out
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"Saved model to {out_model}")
    print(f"Saved meta to  {out_meta}")
    print(f"Metrics: {metrics} | best_iteration={best_it}")


if __name__ == "__main__":
    main()
