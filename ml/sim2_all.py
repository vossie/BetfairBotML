from __future__ import annotations
import argparse, itertools
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any, Optional
import polars as pl
from .sim2 import simulate_once, _select_feature_cols, _predict_proba, _outpath, _load_booster, _daterange
from . import features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _coerce_like(example: Any, s: str) -> Any:
    if isinstance(example, bool): return str(s).strip().lower() in ("1","true","yes","y","on")
    if isinstance(example, int):
        try: return int(float(s))
        except Exception: return int(s)
    if isinstance(example, float): return float(s)
    return s

def _parse_list_or_range(val: str, example: Any) -> List[Any]:
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if ":" in inner and inner.count(":") in (1,2):
            parts = inner.split(":")
            if len(parts)==2: start, stop = float(parts[0]), float(parts[1]); step = (stop - start)
            else: start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
            out=[]; x=start
            if step==0: out=[start]
            else:
                eps=abs(step)*1e-9+1e-12
                cmp=(lambda a,b:a<=b+eps) if step>0 else (lambda a,b:a>=b-eps)
                while cmp(x,stop): out.append(x); x+=step
            return [_coerce_like(example,str(v)) for v in out]
        vals=[v.strip() for v in inner.split(",")] if inner else []
        return [_coerce_like(example,v) for v in vals]
    return [_coerce_like(example,val)]

def _parse_grid_spec(grid_str: str, args_ns: argparse.Namespace) -> Dict[str, List[Any]]:
    parts, buf, depth = [], [], 0
    for ch in grid_str:
        if ch=='[': depth+=1
        elif ch==']': depth=max(0,depth-1)
        if ch==',' and depth==0: parts.append(''.join(buf)); buf=[]
        else: buf.append(ch)
    if buf: parts.append(''.join(buf))
    grid={}
    for item in parts:
        if '=' not in item: continue
        k,v = item.split('=',1); k=k.strip(); v=v.strip()
        if k not in vars(args_ns): continue
        vals=_parse_list_or_range(v, getattr(args_ns,k))
        if vals: grid[k]=vals
    return grid

_DF_IPC_CACHE: Optional[pl.DataFrame] = None
def _load_df_from_ipc(path: Optional[str]) -> Optional[pl.DataFrame]:
    global _DF_IPC_CACHE
    if path is None: return None
    if _DF_IPC_CACHE is None: _DF_IPC_CACHE = pl.read_ipc(path)
    return _DF_IPC_CACHE

def _run_combo(task):
    (idx, keys, vals, args_dict, ipc_path, model_paths) = task
    class A: pass
    a = A(); a.__dict__.update(args_dict); a.__dict__.update({k:v for k,v in zip(keys, vals)})
    df_feat = _load_df_from_ipc(ipc_path)
    booster_single = booster_30 = booster_180 = None
    if model_paths["single"]:
        booster_single = _load_booster(model_paths["model"])
    else:
        booster_30 = _load_booster(model_paths["model_30"])
        booster_180 = _load_booster(model_paths["model_180"])
    summary = simulate_once(a, df_feat, booster_single, booster_30, booster_180,
                            out_prefix=f"combo_{idx}", write_bets=False)
    return {"combo": idx, **{k:v for k,v in zip(keys, vals)}, **summary}

def main():
    ap = argparse.ArgumentParser(description="Load-once + score-once sweep to find best combo (by ROI).")
    ap.add_argument("--model", default=None); ap.add_argument("--model-30", default=None); ap.add_argument("--model-180", default=None)
    ap.add_argument("--gate-mins", type=float, default=45.0)
    ap.add_argument("--curated", required=True); ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True); ap.add_argument("--days-before", type=int, default=0)
    ap.add_argument("--preoff-mins", type=int, default=30); ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0); ap.add_argument("--chunk-days", type=int, default=2)
    ap.add_argument("--stream-bucket-secs", type=int, default=5); ap.add_argument("--latency-ms", type=int, default=300)
    ap.add_argument("--cooldown-secs", type=int, default=60); ap.add_argument("--max-open-per-market", type=int, default=1)
    ap.add_argument("--max-exposure-day", type=float, default=5000.0); ap.add_argument("--place-until-mins", type=float, default=1.0)
    ap.add_argument("--persistence", choices=["lapse","keep"], default="lapse"); ap.add_argument("--rest-secs", type=int, default=10)
    ap.add_argument("--min-stake", type=float, default=1.0); ap.add_argument("--tick-snap", action="store_true"); ap.add_argument("--slip-ticks", type=int, default=1)
    ap.add_argument("--label-col", default="winLabel"); ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25); ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--side", choices=["back","lay","auto"], default="auto"); ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=50.0); ap.add_argument("--stake-cap-day", type=float, default=2000.0)
    ap.add_argument("--odds-min", type=float, default=1.6); ap.add_argument("--odds-max", type=float, default=6.0)
    ap.add_argument("--back-odds-min", type=float, default=None); ap.add_argument("--back-odds-max", type=float, default=None)
    ap.add_argument("--lay-odds-min", type=float, default=None); ap.add_argument("--lay-odds-max", type=float, default=None)
    ap.add_argument("--max-stake-per-bet", type=float, default=5.0); ap.add_argument("--max-liability-per-bet", type=float, default=20.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--bets-out", default="bets.csv"); ap.add_argument("--agg-out", default="bets_by_market.csv")
    ap.add_argument("--bin-out", default="pnl_by_tto_bin.csv"); ap.add_argument("--stream-log-out", default="stream_log.csv")
    ap.add_argument("--sweep-grid", required=True); ap.add_argument("--sweep-parallel", type=int, default=0)
    args = ap.parse_args()

    single = args.model is not None
    dual = (args.model_30 is not None) or (args.model_180 is not None)
    if single and dual: raise SystemExit("Provide either --model OR (--model-30 and --model-180).")
    if not single and not dual:
        default = OUTPUT_DIR / "xgb_model.json"
        if not default.exists(): raise SystemExit("No model supplied and ./output/xgb_model.json not found.")
        args.model = str(default); single = True

    model_paths = {"single": single, "model": None, "model_30": None, "model_180": None}
    if single:
        model_paths["model"] = args.model
        booster_single = _load_booster(args.model); booster_30 = booster_180 = None
    else:
        model_paths["model_30"] = args.model_30; model_paths["model_180"] = args.model_180
        booster_single = None; booster_30 = _load_booster(args.model_30); booster_180 = _load_booster(args.model_180)

    dates = _daterange(args.date, int(args.days_before) + 1)
    parts: List[pl.DataFrame] = []
    for i in range(0, len(dates), args.chunk_days):
        dchunk = dates[i:i + args.chunk_days]
        df_c, _ = features.build_features_streaming(
            curated_root=args.curated, sport=args.sport, dates=dchunk,
            preoff_minutes=args.preoff_mins, batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
        )
        if not df_c.is_empty(): parts.append(df_c)
    if not parts:
        Path(_outpath("sweep_results.csv")).write_text("combo\n"); print("No features produced.")
        return
    df_feat = pl.concat(parts, how="vertical", rechunk=True)

    # score once
    if single:
        fcols = _select_feature_cols(df_feat, args.label_col)
        df_feat = df_feat.with_columns(pl.lit(_predict_proba(df_feat, booster_single, fcols)).alias("p_hat"))
    else:
        early = df_feat.filter(pl.col("tto_minutes") > args.gate_mins)
        late  = df_feat.filter(pl.col("tto_minutes") <= args.gate_mins)
        chunks = []
        if not early.is_empty():
            f30 = _select_feature_cols(early, args.label_col)
            chunks.append(early.with_columns(pl.lit(_predict_proba(early, booster_180, f30)).alias("p_hat")))
        if not late.is_empty():
            f180 = _select_feature_cols(late, args.label_col)
            chunks.append(late.with_columns(pl.lit(_predict_proba(late, booster_30, f180)).alias("p_hat")))
        if chunks: df_feat = pl.concat(chunks, how="vertical", rechunk=True)
    keep = ["sport","marketId","selectionId","publishTimeMs","tto_minutes","winLabel","p_hat",
            "ltp","odds","bestBackOdds","bestBackSize","bestLayOdds","bestLaySize","backSizes","laySizes"]
    have = [c for c in keep if c in df_feat.columns]
    df_feat = df_feat.select(have).rechunk()

    # save for workers
    ipc_path = str(OUTPUT_DIR / "sweep_features.arrow")
    df_feat.write_ipc(ipc_path)

    grid = _parse_grid_spec(args.sweep_grid, args)
    if not grid: raise SystemExit("Empty --sweep-grid.")
    keys = list(grid.keys()); combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total combos: {len(combos)}")

    args_dict = vars(args).copy()
    payloads = [(i, keys, vals, args_dict, ipc_path, model_paths) for i, vals in enumerate(combos, 1)]

    workers = cpu_count() if args.sweep_parallel == 0 else max(1, args.sweep_parallel)
    results: List[Dict[str, Any]] = []
    if workers == 1:
        for p in payloads: results.append(_run_combo(p))
    else:
        with Pool(processes=workers) as pool:
            for row in pool.imap_unordered(_run_combo, payloads): results.append(row)

    if not results: print("No results."); return
    df_res = pl.DataFrame(results).with_columns(pl.col("roi").fill_null(0.0)).sort("roi", descending=True)
    df_res.write_csv(_outpath("sweep_results.csv"))
    best = df_res.row(0, named=True)
    print("\n=== BEST COMBO ===")
    print(" ".join(f"{k}={best[k]}" for k in ["combo"] + keys))
    print(f"result: n_bets={best['n_bets']} stake={best['stake']:.2f} pnl={best['pnl']:.2f} ROI={best['roi']:.3%}")

if __name__ == "__main__":
    main()