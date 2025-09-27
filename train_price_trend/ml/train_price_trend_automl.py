#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, textwrap
from pathlib import Path
from itertools import product
import polars as pl

def parse_args():
    p = argparse.ArgumentParser(
        description="AutoML/sweep for price-trend project: train once then sweep simulate_stream configs."
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

    # Model/runtime
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--output-dir", default="/opt/BetfairBotML/train_price_trend/output")
    p.add_argument("--force-train", action="store_true", help="force retrain even if model exists")
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")

    # Staking
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.01)
    p.add_argument("--kelly-floor", type=float, default=0.001)

    # Sweep grids (comma-separated)
    p.add_argument("--edge-grid", default="0.0005,0.001,0.0015,0.002,0.003")
    p.add_argument("--stake-grid", default="flat,kelly")
    p.add_argument("--odds-grid", default="none,2.2:3.6,1.5:5.0")

    # Liquidity
    p.add_argument("--enforce-liquidity", action="store_true")
    p.add_argument("--liquidity-levels-grid", default="1,3")

    # Misc
    p.add_argument("--tag", default="automl")
    p.add_argument("--batch-size", type=int, default=200_000)
    return p.parse_args()

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
        "--device", args.device,
        "--horizon-secs", str(args.horizon_secs),
        "--preoff-max", str(args.preoff_max),
        "--commission", str(args.commission),
        "--stake-mode", "kelly",                    # training-time fields not critical for dp-model
        "--kelly-cap", str(args.kelly_cap),
        "--kelly-floor", str(args.kelly_floor),
        "--bankroll-nom", str(args.bankroll_nom),
        "--ev-mode", args.ev_mode,
        "--output-dir", args.output_dir,
    ]
    print("[automl] Training model…")
    rc, out, err = _run(cmd)
    (Path(args.output_dir) / "automl" / f"train_{args.asof}.stdout").parent.mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "automl" / f"train_{args.asof}.stdout").write_text(out)
    (Path(args.output_dir) / "automl" / f"train_{args.asof}.stderr").write_text(err)
    if rc != 0:
        print(out)
        print(err, file=sys.stderr)
        raise RuntimeError("Training failed")

def simulate_one(args, base_dir: Path, model_path: Path,
                 edge: float, stake: str,
                 odds_min: float|None, odds_max: float|None,
                 liq_levels: int|None, enforce_liq: bool,
                 sweep_root: Path, run_id: str) -> dict | None:
    outdir = sweep_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    sim_py = base_dir / "ml" / "simulate_stream.py"

    cmd = [
        sys.executable, str(sim_py),
        "--curated", args.curated,
        "--asof", args.asof,
        "--start-date", args.start_date,
        "--valid-days", str(args.valid_days),
        "--sport", args.sport,
        "--preoff-max", str(args.preoff_max),
        "--commission", str(args.commission),
        "--edge-thresh", str(edge),
        "--stake-mode", stake,
        "--kelly-cap", str(args.kelly_cap),
        "--kelly-floor", str(args.kelly_floor),
        "--bankroll-nom", str(args.bankroll_nom),
        "--ev-mode", args.ev_mode,
        "--model-path", str(model_path),
        "--output-dir", str(outdir),
        "--device", args.device,
        "--horizon-secs", str(args.horizon_secs),
        "--batch-size", str(args.batch_size),
    ]
    if odds_min is not None:
        cmd += ["--odds-min", str(odds_min)]
    if odds_max is not None:
        cmd += ["--odds-max", str(odds_max)]
    if enforce_liq:
        cmd += ["--enforce-liquidity", "--liquidity-levels", str(liq_levels or 1)]

    print(f"[automl] Sim run {run_id} …")
    rc, out, err = _run(cmd)
    (outdir / "stdout.log").write_text(out)
    (outdir / "stderr.log").write_text(err)
    if rc != 0:
        print(f"[automl] WARN run {run_id} failed (see logs).")
        return None

    # prefer summary; fallback to daily/trades
    summ_path = outdir / f"summary_{args.asof}.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        return {
            "run_id": run_id,
            "edge_thresh": edge,
            "stake_mode": stake,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "liquidity_enforced": bool(enforce_liq),
            "liquidity_levels": int(liq_levels or 1),
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
    # fallback: daily
    daily_path = outdir / f"daily_{args.asof}.csv"
    if daily_path.exists():
        df = pl.read_csv(daily_path)
        # last row = last day in window
        last = df.tail(1).row(0, named=True)
        return {
            "run_id": run_id,
            "edge_thresh": edge,
            "stake_mode": stake,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "liquidity_enforced": bool(enforce_liq),
            "liquidity_levels": int(liq_levels or 1),
            "n_trades": int(last["n_trades"]),
            "total_exp_profit": float(last["exp_profit"]),
            "total_real_mtm_profit": float(last["real_mtm_profit"]) if "real_mtm_profit" in df.columns else None,
            "total_real_settle_profit": float(last["real_settle_profit"]) if "real_settle_profit" in df.columns else None,
            "overall_roi_exp": float(last["roi_exp"]),
            "overall_roi_real_mtm": float(last["roi_real_mtm"]) if "roi_real_mtm" in df.columns else None,
            "overall_roi_real_settle": float(last["roi_real_settle"]) if "roi_real_settle" in df.columns else None,
            "avg_ev_per_1": float(last["avg_ev"]),
            "output_dir": str(outdir),
        }

    print(f"[automl] WARN no summary/daily found for {run_id}")
    return None

def main():
    args = parse_args()
    base_dir = Path("/opt/BetfairBotML/train_price_trend")
    model_path = Path(args.output_dir) / "models" / "xgb_trend_reg.json"

    # 1) Train once (if needed)
    maybe_train(args, base_dir, model_path)

    # 2) Build sweep grid
    edges = _parse_floats(args.edge_grid)
    stakes = _parse_strs(args.stake_grid)
    bands  = _parse_bands(args.odds_grid)
    liq_levels = [int(x) for x in _parse_strs(args.liquidity_levels_grid)]
    enforce_opts = [bool(args.enforce_liquidity)] if args.enforce_liquidity else [False]  # off by default

    # 3) Output root for AutoML
    automl_root = Path(args.output_dir) / "automl" / args.asof / args.tag
    automl_root.mkdir(parents=True, exist_ok=True)

    # 4) Sweep
    rows = []
    for edge, stake, (odds_min, odds_max, band_name), enforce in product(edges, stakes, bands, enforce_opts):
        if enforce:
            for L in liq_levels:
                run_id = f"edge{edge:g}_{stake}_odds{band_name}_liq{L}"
                res = simulate_one(args, base_dir, model_path, edge, stake, odds_min, odds_max, L, True, automl_root, run_id)
                if res: rows.append(res)
        else:
            run_id = f"edge{edge:g}_{stake}_odds{band_name}_liq0"
            res = simulate_one(args, base_dir, model_path, edge, stake, odds_min, odds_max, None, False, automl_root, run_id)
            if res: rows.append(res)

    if not rows:
        print("[automl] No successful runs.")
        return

    df = pl.DataFrame(rows)

    # 5) Ranking priority: realised MTM ROI, then realised settle ROI, then expected ROI, then exp profit
    score_cols = []
    if "overall_roi_real_mtm" in df.columns:
        score_cols.append("overall_roi_real_mtm")
    if "overall_roi_real_settle" in df.columns:
        score_cols.append("overall_roi_real_settle")
    score_cols.append("overall_roi_exp")
    score_cols.append("total_exp_profit")

    # Replace None with very small to sort properly
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
        "device": args.device,
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
