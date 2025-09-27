#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import polars as pl

# ------------- CLI -------------
def parse_args():
    p = argparse.ArgumentParser(
        description="AutoML/sweep for price-trend project: train once then parallel-simulate best config on CPU."
    )
    # Core data/time
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--horizon-secs", type=int, default=120)
    p.add_argument("--commission", type=float, default=0.02)

    # Devices
    p.add_argument("--train-device", choices=["cuda","cpu"], default="cuda",
                   help="Device for model training (XGBoost). GPU is usually faster.")
    p.add_argument("--sim-device", choices=["cuda","cpu"], default="cpu",
                   help="Device for simulation inference. CPU is recommended for many parallel trials.")

    # Output / runtime
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    p.add_argument("--force-train", action="store_true", help="force retrain even if model exists")
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--batch-size", type=int, default=200_000, help="simulate_stream batch size")

    # Staking
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.01)
    p.add_argument("--kelly-floor", type=float, default=0.001)

    # Sweep grids
    p.add_argument("--edge-grid", default="0.0005,0.001,0.0015,0.002,0.003")
    p.add_argument("--stake-grid", default="flat,kelly")
    p.add_argument("--odds-grid", default="none,2.2:3.6,1.5:5.0")

    # Liquidity (off by default; turn on to include liq sweeps)
    p.add_argument("--enforce-liquidity", action="store_true",
                   help="Include liquidity-enforced runs in the sweep.")
    p.add_argument("--liquidity-levels-grid", default="1,3")

    # Parallelism
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    p.add_argument("--max-parallel", type=int, default=default_workers,
                   help="Concurrent simulation trials (CPU processes).")

    # Tag for results
    p.add_argument("--tag", default="automl")
    return p.parse_args()

# ------------- Utils -------------
def _parse_floats(csv: str) -> list[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]

def _parse_strs(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]

def _parse_bands(csv: str) -> list[tuple[float|None,float|None,str]]:
    out = []
    for token in _parse_strs(csv):
        if token.lower() == "none":
            out.append((None, None, "none"))
        else:
            lo, hi = token.split(":")
            out.append((float(lo), float(hi), token))
    return out

def _run(cmd: list[str], cwd: str | None = None) -> tuple[int,str,str]:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return res.returncode, res.stdout, res.stderr

# ------------- Train once -------------
def maybe_train(args, base_dir: Path, model_path: Path):
    if model_path.exists() and not args.force_train:
        print(f"[automl] Model exists at {model_path} — skipping training (use --force-train to retrain).")
        return
    train_py = base_dir / "ml" / "train_price_trend.py"
    cmd = [
        sys.executable, str(train_py),
        "--curated", args.curated,
        "--asof", args.asof,
        "--start-date", args.start_date,
        "--valid-days", str(args.valid_days),
        "--sport", args.sport,
        "--device", args.train_device,
        "--horizon-secs", str(args.horizon_secs),
        "--preoff-max", str(args.preoff_max),
        "--commission", str(args.commission),
        "--stake-mode", "kelly",
        "--kelly-cap", str(args.kelly_cap),
        "--kelly-floor", str(args.kelly_floor),
        "--bankroll-nom", str(args.bankroll_nom),
        "--ev-mode", args.ev_mode,
        "--output-dir", args.output_dir,
    ]
    print("[automl] Training model…")
    rc, out, err = _run(cmd)
    auto_dir = Path(args.output_dir) / "automl" / args.asof / args.tag
    auto_dir.mkdir(parents=True, exist_ok=True)
    (auto_dir / "train.stdout").write_text(out)
    (auto_dir / "train.stderr").write_text(err)
    if rc != 0:
        print(out)
        print(err, file=sys.stderr)
        raise RuntimeError("Training failed")

# ------------- One simulation (child process safe) -------------
def simulate_one(sim_args: dict) -> tuple[str, dict | None, str]:
    """
    Returns (run_id, result_dict_or_None, error_message_or_empty)
    """
    base_dir = Path(sim_args["base_dir"])
    outdir = Path(sim_args["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    sim_py = base_dir / "ml" / "simulate_stream.py"

    cmd = [
        sys.executable, str(sim_py),
        "--curated", sim_args["curated"],
        "--asof", sim_args["asof"],
        "--start-date", sim_args["start_date"],
        "--valid-days", str(sim_args["valid_days"]),
        "--sport", sim_args["sport"],
        "--preoff-max", str(sim_args["preoff_max"]),
        "--commission", str(sim_args["commission"]),
        "--edge-thresh", str(sim_args["edge_thresh"]),
        "--stake-mode", sim_args["stake_mode"],
        "--kelly-cap", str(sim_args["kelly_cap"]),
        "--kelly-floor", str(sim_args["kelly_floor"]),
        "--bankroll-nom", str(sim_args["bankroll_nom"]),
        "--ev-mode", sim_args["ev_mode"],
        "--model-path", sim_args["model_path"],
        "--output-dir", str(outdir),
        "--device", sim_args["sim_device"],
        "--horizon-secs", str(sim_args["horizon_secs"]),
        "--batch-size", str(sim_args["batch_size"]),
    ]
    if sim_args["odds_min"] is not None:
        cmd += ["--odds-min", str(sim_args["odds_min"])]
    if sim_args["odds_max"] is not None:
        cmd += ["--odds-max", str(sim_args["odds_max"])]
    if sim_args["enforce_liq"]:
        cmd += ["--enforce-liquidity", "--liquidity-levels", str(sim_args["liq_levels"])]

    rc, out, err = _run(cmd)
    (outdir / "stdout.log").write_text(out)
    (outdir / "stderr.log").write_text(err)
    if rc != 0:
        return sim_args["run_id"], None, f"rc={rc}"

    summ_path = outdir / f"summary_{sim_args['asof']}.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        res = {
            "run_id": sim_args["run_id"],
            "edge_thresh": sim_args["edge_thresh"],
            "stake_mode": sim_args["stake_mode"],
            "odds_min": sim_args["odds_min"],
            "odds_max": sim_args["odds_max"],
            "liquidity_enforced": bool(sim_args["enforce_liq"]),
            "liquidity_levels": int(sim_args["liq_levels"]) if sim_args["enforce_liq"] else 0,
            "n_trades": summ.get("n_trades"),
            "total_exp_profit": summ.get("total_exp_profit"),
            "total_real_mtm_profit": summ.get("total_real_mtm_profit"),
            "total_real_settle_profit": summ.get("total_real_settle_profit"),
            "overall_roi_exp": summ.get("overall_roi_exp"),
            "overall_roi_real_mtm": summ.get("overall_roi_real_mtm"),
            "overall_roi_real_settle": summ.get("overall_roi_real_settle"),
            "avg_ev_per_1": summ.get("avg_ev_per_1"),
            "output_dir": str(outdir),
        }
        return sim_args["run_id"], res, ""
    return sim_args["run_id"], None, "no summary json"

# ------------- Main -------------
def main():
    args = parse_args()
    base_dir = Path("/opt/BetfairBotML/train_price_trend")
    model_path = Path(args.output_dir) / "models" / "xgb_trend_reg.json"

    # 1) Train once
    maybe_train(args, base_dir, model_path)

    # 2) Sweep grid
    edges = _parse_floats(args.edge_grid)
    stakes = _parse_strs(args.stake_grid)
    bands  = _parse_bands(args.odds_grid)
    liq_levels = [int(x) for x in _parse_strs(args.liquidity_levels_grid)]
    enforce_opts = [True] if args.enforce_liquidity else [False]

    automl_root = Path(args.output_dir) / "automl" / args.asof / args.tag
    automl_root.mkdir(parents=True, exist_ok=True)

    # Build sim tasks
    tasks: list[dict] = []
    for edge, stake, (odds_min, odds_max, band_name), enforce in product(edges, stakes, bands, enforce_opts):
        if enforce:
            for L in liq_levels:
                run_id = f"edge{edge:g}_{stake}_odds{band_name}_liq{L}"
                outdir = automl_root / run_id
                tasks.append(dict(
                    base_dir=str(base_dir),
                    curated=args.curated,
                    asof=args.asof,
                    start_date=args.start_date,
                    valid_days=args.valid_days,
                    sport=args.sport,
                    preoff_max=args.preoff_max,
                    commission=args.commission,
                    edge_thresh=edge,
                    stake_mode=stake,
                    kelly_cap=args.kelly_cap,
                    kelly_floor=args.kelly_floor,
                    bankroll_nom=args.bankroll_nom,
                    ev_mode=args.ev_mode,
                    model_path=str(model_path),
                    outdir=str(outdir),
                    sim_device=args.sim_device,
                    horizon_secs=args.horizon_secs,
                    batch_size=args.batch_size,
                    odds_min=odds_min,
                    odds_max=odds_max,
                    enforce_liq=True,
                    liq_levels=L,
                    run_id=run_id,
                ))
        else:
            run_id = f"edge{edge:g}_{stake}_odds{band_name}_liq0"
            outdir = automl_root / run_id
            tasks.append(dict(
                base_dir=str(base_dir),
                curated=args.curated,
                asof=args.asof,
                start_date=args.start_date,
                valid_days=args.valid_days,
                sport=args.sport,
                preoff_max=args.preoff_max,
                commission=args.commission,
                edge_thresh=edge,
                stake_mode=stake,
                kelly_cap=args.kelly_cap,
                kelly_floor=args.kelly_floor,
                bankroll_nom=args.bankroll_nom,
                ev_mode=args.ev_mode,
                model_path=str(model_path),
                outdir=str(outdir),
                sim_device=args.sim_device,
                horizon_secs=args.horizon_secs,
                batch_size=args.batch_size,
                odds_min=odds_min,
                odds_max=odds_max,
                enforce_liq=False,
                liq_levels=0,
                run_id=run_id,
            ))

    if not tasks:
        print("[automl] No trials configured.")
        return

    # 3) Run in parallel (CPU many)
    print(f"[automl] Running {len(tasks)} trials with max_parallel={args.max_parallel} on sim_device={args.sim_device} …")
    rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(simulate_one, t): t["run_id"] for t in tasks}
        for fut in as_completed(futs):
            run_id = futs[fut]
            try:
                rid, res, err = fut.result()
                if res is None:
                    failures.append((run_id, err))
                    print(f"[automl] ✗ {run_id} ({err})")
                else:
                    rows.append(res)
                    print(f"[automl] ✓ {run_id}")
            except Exception as e:
                failures.append((run_id, str(e)))
                print(f"[automl] ✗ {run_id} (exception)")

    if failures:
        fail_log = automl_root / "failures.txt"
        fail_log.write_text("\n".join([f"{rid}\t{err}" for rid, err in failures]))
        print(f"[automl] {len(failures)} failures logged to {fail_log}")

    if not rows:
        print("[automl] No successful runs.")
        return

    df = pl.DataFrame(rows)

    # 4) Rank trials: realised MTM ROI, then realised settle ROI, then expected ROI, then expected profit
    score_cols = []
    if "overall_roi_real_mtm" in df.columns:
        score_cols.append("overall_roi_real_mtm")
    if "overall_roi_real_settle" in df.columns:
        score_cols.append("overall_roi_real_settle")
    score_cols.append("overall_roi_exp")
    score_cols.append("total_exp_profit")

    # Normalize None → very small to sort properly
    for c in score_cols:
        if c in df.columns:
            df = df.with_columns(
                pl.when(pl.col(c).is_null()).then(pl.lit(-1e18)).otherwise(pl.col(c)).alias(f"_{c}_score")
            )
    sort_by = [f"_{c}_score" for c in score_cols if f"_{c}_score" in df.columns]
    df = df.sort(by=sort_by, descending=[True]*len(sort_by))

    trials_path = automl_root / f"trials_{args.asof}.csv"
    df.write_csv(trials_path)

    best = df.row(0, named=True)
    best_cfg = {
        "asof": args.asof,
        "start_date": args.start_date,
        "valid_days": args.valid_days,
        "sport": args.sport,
        "preoff_max": args.preoff_max,
        "horizon_secs": args.horizon_secs,
        "commission": args.commission,
        "device": args.train_device,          # device used for training
        "ev_mode": args.ev_mode,
        "bankroll_nom": args.bankroll_nom,
        "kelly_cap": args.kelly_cap,
        "kelly_floor": args.kelly_floor,
        "edge_thresh": float(best["edge_thresh"]),
        "stake_mode": best["stake_mode"],
        "odds_min": None if best["odds_min"] is None else float(best["odds_min"]),
        "odds_max": None if best["odds_max"] is None else float(best["odds_max"]),
        "enforce_liquidity": bool(best["liquidity_enforced"]),
        "liquidity_levels": int(best["liquidity_levels"]),
        "n_trades": int(best["n_trades"]) if best["n_trades"] is not None else None,
        "overall_roi_exp": float(best["overall_roi_exp"]) if best["overall_roi_exp"] is not None else None,
        "overall_roi_real_mtm": float(best["overall_roi_real_mtm"]) if best["overall_roi_real_mtm"] is not None else None,
        "overall_roi_real_settle": float(best["overall_roi_real_settle"]) if best["overall_roi_real_settle"] is not None else None,
        "avg_ev_per_1": float(best["avg_ev_per_1"]) if best["avg_ev_per_1"] is not None else None,
        "output_dir": best["output_dir"],
        "trials_csv": str(trials_path),
        "sim_device": args.sim_device,
        "max_parallel": args.max_parallel,
    }
    best_path = automl_root / "best_config.json"
    best_path.write_text(json.dumps(best_cfg, indent=2))

    print(f"[automl] Wrote {trials_path}")
    print(f"[automl] Best → {best_path}")
    print(
        f"[automl] Best config: edge≥{best_cfg['edge_thresh']}  stake={best_cfg['stake_mode']}  "
        f"odds=[{best_cfg['odds_min']},{best_cfg['odds_max']}]  "
        f"liq_enf={best_cfg['enforce_liquidity']}@L={best_cfg['liquidity_levels']}  "
        f"ROI_real_mtm={best_cfg.get('overall_roi_real_mtm')}"
    )

if __name__ == "__main__":
    main()
