from __future__ import annotations
import argparse, itertools, os, time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any, Optional
import polars as pl

from .sim2 import (
    simulate_once,
    _select_feature_cols,
    _predict_proba,
    _outpath,
    _load_booster,
    _daterange,
)
from . import features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Global Worker Caches ------------------
_DF_IPC_CACHE: Optional[pl.DataFrame] = None
_BOOSTER_SINGLE = None
_BOOSTER_30 = None
_BOOSTER_180 = None
_MODEL_PATHS: Dict[str, Optional[str]] = {"single": True, "model": None, "model_30": None, "model_180": None}

def _ensure_worker_init(ipc_path: Optional[str], model_paths: Dict[str, Optional[str]]):
    """Load heavy assets once per worker."""
    global _DF_IPC_CACHE, _BOOSTER_SINGLE, _BOOSTER_30, _BOOSTER_180, _MODEL_PATHS
    if _DF_IPC_CACHE is None and ipc_path:
        _DF_IPC_CACHE = pl.read_ipc(ipc_path)
    _MODEL_PATHS = model_paths
    if _MODEL_PATHS.get("single"):
        if _BOOSTER_SINGLE is None and _MODEL_PATHS.get("model"):
            _BOOSTER_SINGLE = _load_booster(_MODEL_PATHS["model"])
    else:
        if _BOOSTER_30 is None and _MODEL_PATHS.get("model_30"):
            _BOOSTER_30 = _load_booster(_MODEL_PATHS["model_30"])
        if _BOOSTER_180 is None and _MODEL_PATHS.get("model_180"):
            _BOOSTER_180 = _load_booster(_MODEL_PATHS["model_180"])

def _run_combo(task):
    """Worker: run a single parameter combo."""
    (idx, keys, vals, args_dict, ipc_path, model_paths) = task
    _ensure_worker_init(ipc_path, model_paths)

    class A: pass
    a = A(); a.__dict__.update(args_dict); a.__dict__.update({k: v for k, v in zip(keys, vals)})

    # Choose cached boosters
    if _MODEL_PATHS.get("single"):
        bs = _BOOSTER_SINGLE; b30 = b180 = None
    else:
        bs = None; b30 = _BOOSTER_30; b180 = _BOOSTER_180

    summary = simulate_once(a, _DF_IPC_CACHE, bs, b30, b180,
                            out_prefix=f"combo_{idx}", write_bets=False)
    return {"combo": idx, **{k: v for k, v in zip(keys, vals)}, **summary}

# ------------------ Grid parsing helpers ------------------
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
        if ":" in inner and inner.count(":") in (1, 2):
            p = inner.split(":")
            if len(p) == 2:
                start, stop = float(p[0]), float(p[1]); step = (stop - start)
            else:
                start, stop, step = float(p[0]), float(p[1]), float(p[2])
            out = []
            if step == 0: out = [start]
            else:
                x = start; eps = abs(step) * 1e-9 + 1e-12
                cmp = (lambda a, b: a <= b + eps) if step > 0 else (lambda a, b: a >= b - eps)
                while cmp(x, stop): out.append(x); x += step
            return [_coerce_like(example, str(v)) for v in out]
        vals = [v.strip() for v in inner.split(",")] if inner else []
        return [_coerce_like(example, v) for v in vals]
    return [_coerce_like(example, val)]

def _parse_grid_spec(grid_str: str, args_ns: argparse.Namespace) -> Dict[str, List[Any]]:
    parts, buf, depth = [], [], 0
    for ch in grid_str:
        if ch == '[': depth += 1
        elif ch == ']': depth = max(0, depth - 1)
        if ch == ',' and depth == 0:
            parts.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    if buf: parts.append(''.join(buf))
    grid = {}
    for item in parts:
        if '=' not in item: continue
        k, v = item.split('=', 1); k = k.strip(); v = v.strip()
        if k not in vars(args_ns):  # ignore unknown keys
            continue
        vals = _parse_list_or_range(v, getattr(args_ns, k))
        if vals: grid[k] = vals
    return grid

# ------------------ Main ------------------
def main():
    # let Polars format nicely; threads set via env in launcher
    pl.Config.set_global_string_cache(True)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_fmt_str_lengths(120)

    ap = argparse.ArgumentParser(description="Load-once, score-once parameter sweep to find best combo (by ROI).")
    # models
    ap.add_argument("--model", default=None); ap.add_argument("--model-30", default=None); ap.add_argument("--model-180", default=None)
    ap.add_argument("--gate-mins", type=float, default=45.0)
    # data
    ap.add_argument("--curated", required=True); ap.add_argument("--sport", required=True)
    ap.add_argument("--date", required=True); ap.add_argument("--days-before", type=int, default=0)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--batch-markets", type=int, default=100)
    ap.add_argument("--downsample-secs", type=int, default=0)
    ap.add_argument("--chunk-days", type=int, default=2)
    # streaming realism
    ap.add_argument("--stream-bucket-secs", type=int, default=5)
    ap.add_argument("--latency-ms", type=int, default=300)
    ap.add_argument("--cooldown-secs", type=int, default=60)
    ap.add_argument("--max-open-per-market", type=int, default=1)
    ap.add_argument("--max-exposure-day", type=float, default=5000.0)
    ap.add_argument("--place-until-mins", type=float, default=1.0)
    # persistence
    ap.add_argument("--persistence", choices=["lapse","keep"], default="lapse")
    ap.add_argument("--rest-secs", type=int, default=10)
    # execution
    ap.add_argument("--min-stake", type=float, default=1.0)
    ap.add_argument("--tick-snap", action="store_true")
    ap.add_argument("--slip-ticks", type=int, default=1)
    # selection & staking
    ap.add_argument("--label-col", default="winLabel")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--side", choices=["back","lay","auto"], default="auto")
    ap.add_argument("--top-n-per-market", type=int, default=1)
    ap.add_argument("--stake-cap-market", type=float, default=50.0)
    ap.add_argument("--stake-cap-day", type=float, default=2000.0)
    # guardrails / odds / EV
    ap.add_argument("--odds-min", type=float, default=1.6)
    ap.add_argument("--odds-max", type=float, default=6.0)
    ap.add_argument("--back-odds-min", type=float, default=None)
    ap.add_argument("--back-odds-max", type=float, default=None)
    ap.add_argument("--lay-odds-min", type=float, default=None)
    ap.add_argument("--lay-odds-max", type=float, default=None)
    ap.add_argument("--max-stake-per-bet", type=float, default=5.0)
    ap.add_argument("--max-liability-per-bet", type=float, default=20.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    # outputs
    ap.add_argument("--bets-out", default="bets.csv")
    ap.add_argument("--agg-out", default="bets_by_market.csv")
    ap.add_argument("--bin-out", default="pnl_by_tto_bin.csv")
    ap.add_argument("--stream-log-out", default="stream_log.csv")
    # sweep
    ap.add_argument("--sweep-grid", required=True)
    ap.add_argument("--sweep-parallel", type=int, default=0, help="0=cpu_count, 1=serial, N=workers")

    args = ap.parse_args()

    # Resolve model mode
    single = args.model is not None
    dual = (args.model_30 is not None) or (args.model_180 is not None)
    if single and dual:
        raise SystemExit("Provide either --model OR (--model-30 AND --model-180).")
    if not single and not dual:
        default = OUTPUT_DIR / "xgb_model.json"
        if not default.exists():
            raise SystemExit("No model supplied and ./output/xgb_model.json not found.")
        args.model = str(default); single = True

    # Load model(s) once in parent — we KEEP references and also pass paths to workers
    model_paths = {"single": single, "model": None, "model_30": None, "model_180": None}
    booster_single = booster_30 = booster_180 = None
    if single:
        model_paths["model"] = args.model
        booster_single = _load_booster(args.model)
    else:
        model_paths["model_30"] = args.model_30
        model_paths["model_180"] = args.model_180
        booster_30 = _load_booster(args.model_30)
        booster_180 = _load_booster(args.model_180)

    # Build & concat features once
    t0 = time.time()
    dates = _daterange(args.date, int(args.days_before) + 1)
    parts: List[pl.DataFrame] = []
    for i in range(0, len(dates), args.chunk_days):
        dchunk = dates[i:i + args.chunk_days]
        df_c, _ = features.build_features_streaming(
            curated_root=args.curated,
            sport=args.sport,
            dates=dchunk,
            preoff_minutes=args.preoff_mins,
            batch_markets=args.batch_markets,
            downsample_secs=(args.downsample_secs or None),
        )
        if not df_c.is_empty():
            parts.append(df_c)
    if not parts:
        Path(_outpath("sweep_results.csv")).write_text("combo\n")
        print("No features produced for given date range.")
        return
    df_feat = pl.concat(parts, how="vertical", rechunk=True)
    t1 = time.time()

    # Score once (using in-memory boosters)
    if single:
        fcols = _select_feature_cols(df_feat, args.label_col)
        df_feat = df_feat.with_columns(
            pl.lit(_predict_proba(df_feat, booster_single, fcols)).alias("p_hat")
        )
    else:
        early = df_feat.filter(pl.col("tto_minutes") > args.gate_mins)
        late  = df_feat.filter(pl.col("tto_minutes") <= args.gate_mins)
        chunks = []
        if not early.is_empty():
            f180 = _select_feature_cols(early, args.label_col)
            chunks.append(early.with_columns(pl.lit(_predict_proba(early, booster_180, f180)).alias("p_hat")))
        if not late.is_empty():
            f30 = _select_feature_cols(late, args.label_col)
            chunks.append(late.with_columns(pl.lit(_predict_proba(late, booster_30, f30)).alias("p_hat")))
        if chunks:
            df_feat = pl.concat(chunks, how="vertical", rechunk=True)

    # Trim to needed columns
    keep = [
        "sport","marketId","selectionId","publishTimeMs","tto_minutes",
        "winLabel","p_hat",
        "ltp","odds","bestBackOdds","bestBackSize","bestLayOdds","bestLaySize",
        "backSizes","laySizes",
    ]
    have = [c for c in keep if c in df_feat.columns]
    df_feat = df_feat.select(have).rechunk()

    # Save once for workers (zero-copy Arrow IPC per process)
    ipc_path = str(OUTPUT_DIR / "sweep_features.arrow")
    df_feat.write_ipc(ipc_path)
    t2 = time.time()

    # Build grid
    grid = _parse_grid_spec(args.sweep_grid, args)
    if not grid:
        raise SystemExit("Empty --sweep-grid.")
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total combos: {len(combos)}  |  build+concat={t1-t0:.2f}s  score={t2-t1:.2f}s")

    # Payloads
    base_args = vars(args).copy()
    payloads = [(i, keys, vals, base_args, ipc_path, model_paths) for i, vals in enumerate(combos, 1)]

    # Run (parallel by default)
    workers = cpu_count() if args.sweep_parallel == 0 else max(1, args.sweep_parallel)
    print(f"Using {workers} workers")
    results: List[Dict[str, Any]] = []

    if workers == 1:
        for j, p in enumerate(payloads, 1):
            if j % 100 == 0 or j == len(payloads):
                print(f"… {j}/{len(payloads)} combos")
            results.append(_run_combo(p))
    else:
        with Pool(processes=workers, initializer=_ensure_worker_init, initargs=(ipc_path, model_paths)) as pool:
            for j, row in enumerate(pool.imap_unordered(_run_combo, payloads, chunksize=16), 1):
                results.append(row)
                if j % 100 == 0 or j == len(payloads):
                    print(f"… {j}/{len(payloads)} combos")

    if not results:
        print("No results.")
        return

    df_res = pl.DataFrame(results).with_columns(pl.col("roi").fill_null(0.0)).sort("roi", descending=True)
    df_res.write_csv(_outpath("sweep_results.csv"))

    best = df_res.row(0, named=True)
    print("\n=== BEST COMBO ===")
    print(" ".join(f"{k}={best[k]}" for k in ["combo"] + keys))
    print(f"result: n_bets={best['n_bets']} stake={best['stake']:.2f} pnl={best['pnl']:.2f} ROI={best['roi']:.3%}")

if __name__ == "__main__":
    main()
