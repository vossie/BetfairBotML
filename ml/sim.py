# ml/sim.py
from __future__ import annotations
import argparse, os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import numpy as np
import polars as pl
import xgboost as xgb

from . import features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

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
    exclude = {"marketId","selectionId","ts","ts_ms","publishTimeMs",label_col,"runnerStatus"}
    cols = []
    sch = df.collect_schema()
    for name, dt in zip(sch.names(), sch.dtypes()):
        if name in exclude: continue
        if "label" in name.lower() or "target" in name.lower(): continue
        if dt.is_numeric(): cols.append(name)
    if not cols: raise RuntimeError("No numeric features found.")
    return cols

def _load_booster(path: str) -> xgb.Booster:
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        alt = OUTPUT_DIR / p.name
        if alt.exists():
            p = alt
    bst = xgb.Booster()
    bst.load_model(str(p))
    return bst

def _predict_proba(df: pl.DataFrame, booster: xgb.Booster, feature_cols: List[str]) -> np.ndarray:
    X = df.select(feature_cols).fill_null(strategy="mean").to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X)
    p = booster.predict(dm)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] > 1 else p.ravel()
    return p

# (… keep your staking/EV helpers and binning/report functions unchanged …)

def main():
    ap = argparse.ArgumentParser(description="Simulate wagering from trained model(s).")
    # MODE: do not set defaults to any paths here!
    ap.add_argument("--model", help="Single-model path (JSON).", default=None)
    ap.add_argument("--model-30", help="Short-horizon model path.", default=None)
    ap.add_argument("--model-180", help="Long-horizon model path.", default=None)
    ap.add_argument("--gate-mins", type=float, default=45.0)

    # data
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)

    # selection & staking
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--side", choices=["back","lay","auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=1.0)
    ap.add_argument("--stake-cap-day", type=float, default=10.0)

    # outputs
    ap.add_argument("--bets-out", default="bets.csv")
    ap.add_argument("--agg-out", default="bets_by_market.csv")
    ap.add_argument("--bin-out", default="pnl_by_tto_bin.csv")
    args = ap.parse_args()

    # --- Mode resolution (mutually exclusive) ---
    single = args.model is not None
    dual = (args.model_30 is not None) or (args.model_180 is not None)

    if single and dual:
        raise SystemExit("Provide either --model OR (--model-30 and --model-180), not both.")

    if not single and not dual:
        # No flags provided: try dual defaults in ./output, else single default
        m30 = OUTPUT_DIR / "model_30.json"
        m180 = OUTPUT_DIR / "model_180.json"
        if m30.exists() and m180.exists():
            args.model_30 = str(m30)
            args.model_180 = str(m180)
            dual = True
        else:
            m1 = OUTPUT_DIR / "xgb_model.json"
            if not m1.exists():
                raise SystemExit("No model paths provided and no default models found in ./output/")
            args.model = str(m1)
            single = True

    # Load boosters
    booster_30 = booster_180 = None
    if single:
        booster_30 = _load_booster(args.model)
    else:
        if args.model_30 is None or args.model_180 is None:
            raise SystemExit("Dual mode requires both --model-30 and --model-180.")
        booster_30 = _load_booster(args.model_30)
        booster_180 = _load_booster(args.model_180)

    dates = _daterange(args.date, args.days)

    # Build features in chunks (unchanged; call your features.build_features_streaming)
    df_parts: List[pl.DataFrame] = []
    total_raw = 0
    for i in range(0, len(dates), args.chunk_days):
        dchunk = dates[i:i+args.chunk_days]
        df_c, raw_c = features.build_features_streaming(
            curated_root=args.curated, sport=args.sport, dates=dchunk,
            preoff_minutes=args.preoff_mins, batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
        )
        total_raw += raw_c
        if not df_c.is_empty():
            df_parts.append(df_c)
    if not df_parts:
        raise SystemExit("No features produced for given date range.")

    df_feat = pl.concat(df_parts, how="vertical", rechunk=True)

    # Score
    def _score(df: pl.DataFrame) -> pl.DataFrame:
        if booster_30 and booster_180:
            early = df.filter(pl.col("tto_minutes") > args.gate_mins)
            late  = df.filter(pl.col("tto_minutes") <= args.gate_mins)
            parts = []
            if not early.is_empty():
                fcols = _select_feature_cols(early, args.label_col)
                p = _predict_proba(early, booster_180, fcols)
                parts.append(early.with_columns(pl.lit(p).alias("p_hat")))
            if not late.is_empty():
                fcols = _select_feature_cols(late, args.label_col)
                p = _predict_proba(late, booster_30, fcols)
                parts.append(late.with_columns(pl.lit(p).alias("p_hat")))
            return pl.concat(parts, how="vertical") if parts else df.with_columns(pl.lit(None).alias("p_hat"))
        else:
            fcols = _select_feature_cols(df, args.label_col)
            p = _predict_proba(df, booster_30, fcols)
            return df.with_columns(pl.lit(p).alias("p_hat"))

    df_scored = _score(df_feat)

    # Build bets, apply caps, compute pnl, write CSVs
    # (use your existing _build_bets_table, _pick_topn_per_market, _cap_stakes, _pnl_columns, _binwise_pnl)
    # Ensure outputs go to ./output if relative:
    def _outpath(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute(): return str(pp)
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        return str(OUTPUT_DIR / pp.name)

    # ---- your existing post-scoring code here ----
    # bets = _build_bets_table(...)
    # bets = _pick_topn_per_market(bets, args.top_n_per_market)
    # bets = _cap_stakes(...)
    # bets = _pnl_columns(bets, args.commission)
    #
    # bets.write_csv(_outpath(args.bets_out))
    # agg.write_csv(_outpath(args.agg_out))
    # bin_pnl.write_csv(_outpath(args.bin_out))

if __name__ == "__main__":
    main()
