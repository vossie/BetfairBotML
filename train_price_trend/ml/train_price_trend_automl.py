#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, json, math, itertools, subprocess, shlex
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    p = argparse.ArgumentParser("AutoML for price-trend simulator (robust logging)")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--commission", type=float, default=0.02)

    p.add_argument("--train-device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--sim-device", choices=["cuda","cpu"], default="cpu")
    p.add_argument("--max-parallel", type=int, default=4)

    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--edge-grid", default="0.0005,0.001,0.0015,0.002,0.003")
    p.add_argument("--stake-grid", default="flat,kelly")
    p.add_argument("--odds-grid", default="2.2:3.6,1.5:5.0,none")
    p.add_argument("--liquidity-levels-grid", default="1,3")
    p.add_argument("--enforce-liquidity-only", action="store_true")
    p.add_argument("--min-fill-frac-grid", default="0.25")
    p.add_argument("--per-market-topk-grid", default="1")
    p.add_argument("--per-market-budget-grid", default="5,10")
    p.add_argument("--exit-on-move-ticks-grid", default="0,1,2")

    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)

    p.add_argument("--batch-size", type=int, default=75_000)
    p.add_argument("--ev-cap", type=float, default=1.0)
    p.add_argument("--ev-scale-grid", default="1.0,0.01,0.005,0.001")

    p.add_argument("--min-trades", type=int, default=20000)
    p.add_argument("--prefer", choices=["mtm","settlement"], default="settlement")

    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    p.add_argument("--tag", default="automl")
    p.add_argument("--force-train", action="store_true")
    return p.parse_args()

def parse_float_list(s: str) -> list[float]:
    out=[]
    for x in s.split(","):
        x=x.strip()
        if not x: continue
        out.append(float(x))
    return out

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_odds_item(item: str):
    item=item.strip()
    if item=="none":
        return (None, None)
    if ":" in item:
        a,b=item.split(":",1)
        return (float(a), float(b))
    raise ValueError(f"bad odds item: {item}")

def ensure_dirs(base_out: Path, asof: str, tag: str):
    root = base_out / "automl" / asof / tag
    (root / "automl").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root

def model_exists(models_dir: Path) -> bool:
    return (models_dir / "xgb_trend_reg.json").exists()

def train_if_needed(args, out_root: Path):
    models_dir = args.output_dir + "/models"
    if model_exists(Path(models_dir)) and not args.force_train:
        print(f"[automl] Model exists at {models_dir}/xgb_trend_reg.json — skipping training (use --force-train to retrain).")
        return
    # call trainer with training-only args
    cmd = [
        sys.executable, "/opt/BetfairBotML/train_price_trend/ml/train_price_trend.py",
        "--curated", args.curated,
        "--asof", args.asof,
        "--start-date", args.start_date,
        "--valid-days", str(args.valid_days),
        "--sport", args.sport,
        "--horizon-secs", str(args.horizon_secs),
        "--preoff-max", str(args.preoff_max),
        "--commission", str(args.commission),
        "--device", args.train_device,
        "--output-dir", args.output_dir,
    ]
    print("[automl] Training…")
    subprocess.run(cmd, check=True)

def run_trial(sim_cmd: list[str], env: dict, log_out: Path, log_err: Path) -> tuple[int, dict]:
    with open(log_out, "wb") as fo, open(log_err, "wb") as fe:
        rc = subprocess.Popen(sim_cmd, stdout=fo, stderr=fe, env=env).wait()
    # parse summary json if present
    summary = {}
    outdir = None
    for i, tok in enumerate(sim_cmd):
        if tok == "--output-dir" and i+1 < len(sim_cmd):
            outdir = Path(sim_cmd[i+1]); break
    if outdir:
        summ_path = outdir / f"summary_{env.get('ASOF_OVERRIDE','') or ''}.json"
        # fallback: scan for any summary_<asof>.json in outdir
        if not summ_path.exists():
            for p in outdir.glob("summary_*.json"):
                summ_path = p; break
        if summ_path.exists():
            try:
                summary = json.loads(summ_path.read_text())
            except Exception:
                summary = {}
    return rc, summary

def score(summary: dict, prefer: str, min_trades: int) -> float:
    n = int(summary.get("n_trades", 0))
    if n < min_trades:
        # treat as valid (but bad) instead of crashing the run
        return -1e12 + n
    key = "overall_roi_real_mtm" if prefer=="mtm" else "overall_roi_real_settle"
    val = summary.get(key, None)
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return -1e11
    return float(val)

def main():
    a = parse_args()

    # Output structure
    root = ensure_dirs(Path(a.output_dir), a.asof, "automl")
    automl_dir = root / "automl"
    logs_dir   = root / "logs"
    trials_csv = automl_dir / f"trials_{a.asof}.csv"
    failures   = automl_dir / "failures.txt"
    best_json  = automl_dir / "best_config.json"

    # Train (or skip)
    train_if_needed(a, root)

    # Grids
    edges   = parse_float_list(a.edge_grid)
    stakes  = [s.strip() for s in a.stake_grid.split(",") if s.strip()]
    odds_bands = [parse_odds_item(x) for x in a.odds_grid.split(",") if x.strip()]
    liq_levels = parse_int_list(a.liquidity_levels_grid)
    min_fill_fracs = [float(x) for x in a.min_fill_frac_grid.split(",") if x.strip()]
    topk_grid = parse_int_list(a.per_market_topk_grid)
    budget_grid = parse_float_list(a.per_market_budget_grid)
    exit_ticks_grid = parse_int_list(a.exit_on_move_ticks_grid)
    ev_scales = parse_float_list(a.ev_scale_grid)

    # Fixed env for sims
    env = os.environ.copy()
    env["POLARS_MAX_THREADS"] = env.get("POLARS_MAX_THREADS", "4")
    env["XGB_FORCE_NTHREADS"] = env.get("XGB_FORCE_NTHREADS", "4")
    env["ASOF_OVERRIDE"] = a.asof  # used to locate summary files safely

    trials = []
    for edge, stake, (omin,omax), liq, mff, topk, bud, xticks, evs in itertools.product(
        edges, stakes, odds_bands, liq_levels, min_fill_fracs, topk_grid, budget_grid, exit_ticks_grid, ev_scales
    ):
        tag = f"edge{edge}_{stake}_odds{('none' if omin is None else f'{omin}:{omax}')}_liq{liq}_mff{mff}_k{topk}_b{bud}_xt{xticks}_evs{evs}"
        out_dir = root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        sim = [
            sys.executable, "/opt/BetfairBotML/train_price_trend/ml/simulate_stream.py",
            "--curated", a.curated,
            "--asof", a.asof,
            "--start-date", a.start_date,
            "--valid-days", str(a.valid_days),
            "--sport", a.sport,
            "--horizon-secs", str(a.horizon_secs),
            "--preoff-max", str(a.preoff_max),
            "--commission", str(a.commission),
            "--edge-thresh", str(edge),
            "--stake-mode", stake,
            "--bankroll-nom", str(a.bankroll_nom),
            "--kelly-cap", str(a.kelly_cap),
            "--kelly-floor", str(a.kelly_floor),
            "--device", a.sim_device,
            "--ev-mode", a.ev_mode,
            "--ev-scale", str(evs),
            "--output-dir", str(out_dir),
            "--per-market-topk", str(topk),
            "--per-market-budget", str(bud),
            "--exit-on-move-ticks", str(xticks),
            "--batch-size", str(a.batch_size),
            "--ev-cap", str(a.ev_cap),
        ]
        # liquidity
        if a.enforce_liquidity_only:
            sim += ["--enforce-liquidity", "--liquidity-levels", str(liq)]
            sim += ["--min-fill-frac", str(mff)]
        else:
            # allow liq=0 by skipping flag; liq>0 enforces
            if liq > 0:
                sim += ["--enforce-liquidity", "--liquidity-levels", str(liq)]
                sim += ["--min-fill-frac", str(mff)]
        # odds
        if omin is not None: sim += ["--odds-min", str(omin)]
        if omax is not None: sim += ["--odds-max", str(omax)]

        trials.append((tag, sim, out_dir))

    print(f"[automl] Running {len(trials)} trials with max_parallel={a.max_parallel} on sim_device={a.sim_device} …")
    failures_lines=[]
    scored=[]
    with ThreadPoolExecutor(max_workers=a.max_parallel) as ex:
        fut2tag={}
        for tag, sim, out_dir in trials:
            log_out = (root / "logs" / f"{tag}.out")
            log_err = (root / "logs" / f"{tag}.err")
            fut = ex.submit(run_trial, sim, env, log_out, log_err)
            fut2tag[fut]=(tag, sim, out_dir, log_out, log_err)

        for fut in as_completed(fut2tag):
            tag, sim, out_dir, log_out, log_err = fut2tag[fut]
            rc, summ = fut.result()
            if rc != 0:
                # capture tail of stderr for quick debugging
                tail = ""
                try:
                    with open(log_err, "rb") as fe:
                        data = fe.read().decode("utf-8", errors="ignore")
                        tail = "\n".join(data.strip().splitlines()[-5:])
                except Exception:
                    pass
                failures_lines.append(f"{tag}\trc={rc}\n{tail}\nCMD: {' '.join(shlex.quote(t) for t in sim)}\n")
                continue
            sc = score(summ, a.prefer, a.min_trades)
            scored.append((sc, tag, summ))

    # write failures
    if failures_lines:
        with open(failures, "w") as f:
            f.write("\n\n".join(failures_lines))
        print(f"[automl] {len(failures_lines)} failures logged to {failures}")

    if not scored:
        print("[automl] No successful runs.")
        sys.exit(0)

    # pick best
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    _, tag, summ = best
    best_cfg = {
        "asof": a.asof,
        "start_date": a.start_date,
        "valid_days": a.valid_days,
        "sport": a.sport,
        "preoff_max": a.preoff_max,
        "horizon_secs": a.horizon_secs,
        "commission": a.commission,
        "device": a.train_device,
        "ev_mode": a.ev_mode,
        "bankroll_nom": a.bankroll_nom,
        "kelly_cap": a.kelly_cap,
        "kelly_floor": a.kelly_floor,
        "edge_thresh": float(summ.get("edge_thresh", "nan")) if "edge_thresh" in summ else None,
        "stake_mode": summ.get("stake_mode"),
        "odds_min": summ.get("odds_min"),
        "odds_max": summ.get("odds_max"),
        "enforce_liquidity": bool(summ.get("enforce_liquidity_effective", False)),
        "liquidity_levels": int(summ.get("liquidity_levels", 0)),
        "min_fill_frac": float(summ.get("min_fill_frac", 0.0)),
        "per_market_topk": int(summ.get("per_market_topk", 1)),
        "per_market_budget": float(summ.get("per_market_budget", 10.0)),
        "exit_on_move_ticks": int(summ.get("exit_on_move_ticks", 0)),
        "ev_scale": float(summ.get("ev_scale_used", 1.0)),
        "n_trades": int(summ.get("n_trades", 0)),
        "overall_roi_exp": summ.get("overall_roi_exp"),
        "overall_roi_real_mtm": summ.get("overall_roi_real_mtm"),
        "overall_roi_real_settle": summ.get("overall_roi_real_settle"),
        "avg_ev_per_1": summ.get("avg_ev_per_1"),
        "output_dir": str(out_dir),
        "trials_csv": str(trials_csv),
        "sim_device": a.sim_device,
        "max_parallel": a.max_parallel,
    }
    (automl_dir / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    print(f"[automl] Wrote {trials_csv}")
    print(f"[automl] Best → {automl_dir / 'best_config.json'}")
    print(f"[automl] Best config: tag={tag}  ROI_{a.prefer}={best[0]:.6g}")

if __name__ == "__main__":
    main()
