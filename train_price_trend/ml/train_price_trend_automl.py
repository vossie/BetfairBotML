#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, json, math, itertools, subprocess, shlex, signal, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------ Arg parsing ------------------------

def parse_args():
    p = argparse.ArgumentParser("AutoML for price-trend simulator (robust, with timeouts & logs)")
    # Data window
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--commission", type=float, default=0.02)

    # Devices / parallel
    p.add_argument("--train-device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--sim-device", choices=["cuda","cpu"], default="cpu")
    p.add_argument("--max-parallel", type=int, default=4)

    # Search grids
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
    p.add_argument("--ev-scale-grid", default="1.0,0.01,0.005,0.001")
    p.add_argument("--batch-size", type=int, default=75_000)
    p.add_argument("--ev-cap", type=float, default=1.0)

    # Scoring / constraints
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--min-trades", type=int, default=20_000)
    p.add_argument("--prefer", choices=["mtm","settlement"], default="settlement")

    # IO / behavior
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    p.add_argument("--tag", default="automl")
    p.add_argument("--force-train", action="store_true")

    # Anti-hang
    p.add_argument("--trial-timeout-secs", type=int, default=1800, help="Kill a trial if it exceeds this time.")
    p.add_argument("--tail-stderr-lines", type=int, default=40)
    return p.parse_args()

# ------------------------ Helpers ------------------------

def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_odds_item(item: str):
    item = item.strip()
    if item == "none":
        return (None, None)
    if ":" in item:
        a, b = item.split(":", 1)
        return (float(a), float(b))
    raise ValueError(f"bad odds item: {item}")

def ensure_dirs(base_out: Path, asof: str, tag: str):
    root = base_out / "automl" / asof / tag
    (root / "automl").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root

def model_exists(models_dir: Path) -> bool:
    return (models_dir / "xgb_trend_reg.json").exists()

def train_if_needed(args, models_dir: Path):
    if model_exists(models_dir) and not args.force_train:
        print(f"[automl] Model exists at {models_dir}/xgb_trend_reg.json — skipping training (use --force-train to retrain).")
        return
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
        "--output-dir", str(models_dir.parent),
    ]
    print("[automl] Training…")
    subprocess.run(cmd, check=True)

def tail_file(p: Path, n: int) -> str:
    try:
        data = p.read_text(errors="ignore").splitlines()
        return "\n".join(data[-n:])
    except Exception:
        return ""

def score_summary(summary: dict, prefer: str, min_trades: int) -> float:
    n = int(summary.get("n_trades", 0))
    if n < min_trades:
        # Valid but bad score (don’t fail the whole AutoML)
        return -1e12 + n
    key = "overall_roi_real_mtm" if prefer == "mtm" else "overall_roi_real_settle"
    val = summary.get(key, None)
    try:
        val = float(val)
    except Exception:
        return -1e11
    if math.isnan(val) or math.isinf(val):
        return -1e11
    return val

# ------------------------ Trial runner with timeout ------------------------

def run_trial_with_timeout(sim_cmd: list[str], env: dict, log_out: Path, log_err: Path, timeout: int) -> tuple[int, dict]:
    """
    Runs a single simulate command with:
      - stdout/stderr redirected to files
      - process group created; if timeout, kill -TERM, then -KILL
      - returns (rc, parsed_summary_dict|{})
    """
    # Ensure output dir known to read summary later
    out_dir = None
    for i, tok in enumerate(sim_cmd):
        if tok == "--output-dir" and i + 1 < len(sim_cmd):
            out_dir = Path(sim_cmd[i + 1])
            break

    # Spawn child in its own process group so we can kill the whole tree
    with open(log_out, "wb") as fo, open(log_err, "wb") as fe:
        proc = subprocess.Popen(
            sim_cmd,
            stdout=fo,
            stderr=fe,
            env=env,
            preexec_fn=os.setsid  # new process group (POSIX)
        )
        start = time.time()
        rc = None
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if timeout and (time.time() - start) > timeout:
                # soft kill
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
                # wait a bit
                for _ in range(20):
                    rc = proc.poll()
                    if rc is not None:
                        break
                    time.sleep(0.1)
                if rc is None:
                    # hard kill
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    rc = 124  # timeout code
                break
            time.sleep(0.25)

    # Parse summary if present
    summary = {}
    if out_dir and out_dir.exists():
        # Prefer exact ASOF summary, else any summary_*.json in that out_dir
        asof = env.get("ASOF_OVERRIDE", "")
        cand = out_dir / f"summary_{asof}.json" if asof else None
        try_paths = []
        if cand:
            try_paths.append(cand)
        try_paths.extend(sorted(out_dir.glob("summary_*.json")))
        for p in try_paths:
            if p.exists():
                try:
                    summary = json.loads(p.read_text())
                    break
                except Exception:
                    pass
    return rc, summary

# ------------------------ Main ------------------------

def main():
    a = parse_args()

    root = ensure_dirs(Path(a.output_dir), a.asof, a.tag)
    automl_dir = root / "automl"
    logs_dir   = root / "logs"
    trials_csv = automl_dir / f"trials_{a.asof}.csv"
    failures   = automl_dir / "failures.txt"
    best_json  = automl_dir / "best_config.json"

    models_dir = Path(a.output_dir) / "models"
    train_if_needed(a, models_dir)

    # Build grids
    edges   = parse_float_list(a.edge_grid)
    stakes  = [s.strip() for s in a.stake_grid.split(",") if s.strip()]
    odds_bands = [parse_odds_item(x) for x in a.odds_grid.split(",") if x.strip()]
    liq_levels = parse_int_list(a.liquidity_levels_grid)
    min_fill_fracs = [float(x) for x in a.min_fill_frac_grid.split(",") if x.strip()]
    topk_grid = parse_int_list(a.per_market_topk_grid)
    budget_grid = parse_float_list(a.per_market_budget_grid)
    exit_ticks_grid = parse_int_list(a.exit_on_move_ticks_grid)
    ev_scales = parse_float_list(a.ev_scale_grid)

    # Environment for child sims
    env = os.environ.copy()
    env.setdefault("POLARS_MAX_THREADS", str(max(1, a.max_parallel * 2)))
    env.setdefault("XGB_FORCE_NTHREADS", str(max(1, a.max_parallel * 2)))
    env["ASOF_OVERRIDE"] = a.asof

    # Compose trials
    trials = []
    for edge in edges:
        for stake in stakes:
            for (omin, omax) in odds_bands:
                for liq in liq_levels:
                    for mff in min_fill_fracs:
                        for topk in topk_grid:
                            for bud in budget_grid:
                                for xticks in exit_ticks_grid:
                                    for evs in ev_scales:
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
                                        # Liquidity flags
                                        if a.enforce_liquidity_only or liq > 0:
                                            sim += ["--enforce-liquidity", "--liquidity-levels", str(liq), "--min-fill-frac", str(mff)]
                                        # Odds
                                        if omin is not None: sim += ["--odds-min", str(omin)]
                                        if omax is not None: sim += ["--odds-max", str(omax)]
                                        trials.append((tag, sim, out_dir))

    total = len(trials)
    print(f"[automl] Running {total} trials with max_parallel={a.max_parallel} on sim_device={a.sim_device} …")

    # Run trials with timeouts
    failures_lines = []
    results_rows = []
    best = None  # (score, tag, summary, out_dir)
    completed = 0

    def row_from_summary(tag: str, summ: dict) -> list:
        return [
            tag,
            summ.get("edge_thresh"),
            summ.get("stake_mode"),
            summ.get("odds_min"),
            summ.get("odds_max"),
            summ.get("enforce_liquidity_effective"),
            summ.get("liquidity_levels"),
            summ.get("min_fill_frac"),
            summ.get("per_market_topk"),
            summ.get("per_market_budget"),
            summ.get("exit_on_move_ticks"),
            summ.get("ev_scale_used"),
            summ.get("n_trades"),
            summ.get("overall_roi_exp"),
            summ.get("overall_roi_real_mtm"),
            summ.get("overall_roi_real_settle"),
            summ.get("avg_ev_per_1"),
        ]

    with ThreadPoolExecutor(max_workers=a.max_parallel) as ex:
        fut2info = {}
        for tag, sim, out_dir in trials:
            log_out = (logs_dir / f"{tag}.out")
            log_err = (logs_dir / f"{tag}.err")
            fut = ex.submit(run_trial_with_timeout, sim, env, log_out, log_err, a.trial_timeout_secs)
            fut2info[fut] = (tag, sim, out_dir, log_out, log_err)

        for fut in as_completed(fut2info):
            tag, sim, out_dir, log_out, log_err = fut2info[fut]
            rc, summ = fut.result()
            completed += 1
            if rc != 0:
                tail = tail_file(log_err, a.tail_stderr_lines)
                failures_lines.append(f"{tag}\trc={rc}\n{tail}\nCMD: {' '.join(shlex.quote(t) for t in sim)}\n")
                print(f"[automl] ✗ {tag} (rc={rc})  [{completed}/{total}]")
                continue

            sc = score_summary(summ, a.prefer, a.min_trades)
            results_rows.append(row_from_summary(tag, summ))
            if (best is None) or (sc > best[0]):
                best = (sc, tag, summ, out_dir)
            print(f"[automl] ✓ {tag}  score={sc:.6g}  n={summ.get('n_trades',0)}  [{completed}/{total}]")

    # Write failures, trials CSV, and best config
    if failures_lines:
        failures.write_text("\n\n".join(failures_lines))
        print(f"[automl] {len(failures_lines)} failures logged to {failures}")

    # Minimal CSV writer (no pandas dependency)
    if results_rows:
        header = [
            "tag","edge_thresh","stake_mode","odds_min","odds_max",
            "enforce_liquidity","liquidity_levels","min_fill_frac",
            "per_market_topk","per_market_budget","exit_on_move_ticks",
            "ev_scale","n_trades","roi_exp","roi_real_mtm","roi_real_settle","avg_ev_per_1"
        ]
        with open(trials_csv, "w") as f:
            f.write(",".join(header) + "\n")
            for r in results_rows:
                f.write(",".join("" if v is None else str(v) for v in r) + "\n")
        print(f"[automl] Wrote {trials_csv}")

    if best is None:
        print("[automl] No successful runs.")
        return

    sc, tag, summ, out_dir = best
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
        "edge_thresh": float(summ.get("edge_thresh")) if summ.get("edge_thresh") is not None else None,
        "stake_mode": summ.get("stake_mode"),
        "odds_min": summ.get("odds_min"),
        "odds_max": summ.get("odds_max"),
        "enforce_liquidity": bool(summ.get("enforce_liquidity_effective", False)),
        "liquidity_levels": int(summ.get("liquidity_levels", 0) or 0),
        "min_fill_frac": float(summ.get("min_fill_frac", 0.0) or 0.0),
        "per_market_topk": int(summ.get("per_market_topk", 1) or 1),
        "per_market_budget": float(summ.get("per_market_budget", 10.0) or 10.0),
        "exit_on_move_ticks": int(summ.get("exit_on_move_ticks", 0) or 0),
        "ev_scale": float(summ.get("ev_scale_used", 1.0) or 1.0),
        "n_trades": int(summ.get("n_trades", 0) or 0),
        "overall_roi_exp": summ.get("overall_roi_exp"),
        "overall_roi_real_mtm": summ.get("overall_roi_real_mtm"),
        "overall_roi_real_settle": summ.get("overall_roi_real_settle"),
        "avg_ev_per_1": summ.get("avg_ev_per_1"),
        "output_dir": str(out_dir),
        "trials_csv": str(trials_csv),
        "sim_device": a.sim_device,
        "max_parallel": a.max_parallel,
        "score_metric": a.prefer,
        "score_value": sc,
    }
    best_json.write_text(json.dumps(best_cfg, indent=2))
    print(f"[automl] Best → {best_json}")
    print(f"[automl] Best tag={tag}  score({a.prefer})={sc:.6g}")

if __name__ == "__main__":
    main()
