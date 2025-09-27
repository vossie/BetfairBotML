#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from itertools import product
import polars as pl

def parse_args():
    p = argparse.ArgumentParser(description="Sweep EV thresholds / staking / odds bands for price-trend simulator")
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--base-output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.01)   # tighter by default for sweeps
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=200_000)
    p.add_argument("--min-trades", type=int, default=10_000, help="filter out configs with too few trades")
    # grids (comma-separated)
    p.add_argument("--edge-thresh", default="0.0005,0.001,0.002,0.003,0.005")
    p.add_argument("--stake-modes", default="flat,kelly")
    p.add_argument("--odds-bands", default="none,2.2:3.6,1.5:5.0")
    p.add_argument("--tag", default="", help="optional tag to nest sweep outputs")
    return p.parse_args()

def _parse_floats(csv: str) -> list[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]

def _parse_strs(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]

def _parse_bands(csv: str) -> list[tuple[float|None,float|None]]:
    out = []
    for token in _parse_strs(csv):
        if token.lower() == "none":
            out.append((None, None))
        else:
            lo, hi = token.split(":")
            out.append((float(lo), float(hi)))
    return out

def main():
    args = parse_args()

    # Build grid
    edges = _parse_floats(args.edge_thresh)
    stakes = _parse_strs(args.stake_modes)
    bands  = _parse_bands(args.odds_bands)

    # Output root for this sweep
    sweep_root = Path(args.base_output_dir) / "sweeps" / args.asof / (args.tag or "untagged")
    sweep_root.mkdir(parents=True, exist_ok=True)

    trials_rows = []

    # Iterate grid
    combo_idx = 0
    for edge, stake, (odds_min, odds_max) in product(edges, stakes, bands):
        combo_idx += 1
        run_id = f"edge{edge:g}_{stake}"
        if odds_min is not None or odds_max is not None:
            run_id += f"_odds{(odds_min or 0):g}-{(odds_max or 999):g}"
        run_dir = sweep_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Each run writes into its own output-dir to avoid clobbering
        outdir = run_dir

        cmd = [
            sys.executable, "/opt/BetfairBotML/train_price_trend/ml/simulate_stream.py",
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
            "--model-path", args.model_path,
            "--output-dir", str(outdir),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
        ]
        if odds_min is not None:
            cmd += ["--odds-min", str(odds_min)]
        if odds_max is not None:
            cmd += ["--odds-max", str(odds_max)]

        print(f"[sweep] Running {run_id} …")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            (run_dir / "stderr.log").write_text(res.stderr)
            (run_dir / "stdout.log").write_text(res.stdout)
            print(f"[sweep] WARN {run_id} failed (see logs).")
            continue
        (run_dir / "stdout.log").write_text(res.stdout)

        # Load summary (preferred) else compute from trades if needed
        summ_path = outdir / f"summary_{args.asof}.json"
        if summ_path.exists():
            summ = json.loads(summ_path.read_text())
            n_trades = summ.get("n_trades")
            total_exp = summ.get("total_exp_profit")
            avg_ev    = summ.get("avg_ev_per_1")
            overall_roi = summ.get("overall_roi")
        else:
            # Fallback: compute from trades
            trades_path = outdir / f"trades_{args.asof}.csv"
            if not trades_path.exists():
                print(f"[sweep] WARN no summary/trades for {run_id}")
                continue
            df = pl.read_csv(trades_path)
            n_trades = int(df.height)
            total_exp = float(df["exp_pnl"].sum())
            avg_ev = float(df["ev_per_1"].mean())
            overall_roi = total_exp / float(args.bankroll_nom) if args.bankroll_nom else None

        # Skip tiny runs if configured
        if n_trades is None or n_trades < args.min_trades:
            print(f"[sweep] SKIP {run_id} (n_trades={n_trades} < {args.min_trades})")
            continue

        trials_rows.append({
            "run_id": run_id,
            "edge_thresh": edge,
            "stake_mode": stake,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "n_trades": n_trades,
            "total_exp_profit": total_exp,
            "overall_roi": overall_roi,
            "avg_ev_per_1": avg_ev,
            "output_dir": str(outdir),
        })

    if not trials_rows:
        print("[sweep] No successful runs to record.")
        return

    trials = pl.DataFrame(trials_rows)
    trials = trials.sort(by=["overall_roi", "total_exp_profit"], descending=[True, True])
    trials_path = sweep_root / "trials.csv"
    trials.write_csv(trials_path)

    # Best config
    best_row = trials.row(0, named=True)
    best = {
        "asof": args.asof,
        "start_date": args.start_date,
        "valid_days": args.valid_days,
        "preoff_max": args.preoff_max,
        "commission": args.commission,
        "ev_mode": args.ev_mode,
        "device": args.device,
        "bankroll_nom": args.bankroll_nom,
        "kelly_cap": args.kelly_cap,
        "kelly_floor": args.kelly_floor,
        "edge_thresh": best_row["edge_thresh"],
        "stake_mode": best_row["stake_mode"],
        "odds_min": best_row["odds_min"],
        "odds_max": best_row["odds_max"],
        "n_trades": int(best_row["n_trades"]),
        "total_exp_profit": float(best_row["total_exp_profit"]),
        "overall_roi": float(best_row["overall_roi"]) if best_row["overall_roi"] is not None else None,
        "avg_ev_per_1": float(best_row["avg_ev_per_1"]),
        "output_dir": best_row["output_dir"],
        "trials_csv": str(trials_path),
    }
    best_path = sweep_root / "best_config.json"
    best_path.write_text(json.dumps(best, indent=2))
    print(f"[sweep] Wrote {trials_path}")
    print(f"[sweep] Best → {best_path}")
    print(f"[sweep] Best config: edge≥{best['edge_thresh']}  stake={best['stake_mode']}  "
          f"odds=[{best['odds_min']},{best['odds_max']}]  ROI={best['overall_roi']:.6f}  "
          f"n_trades={best['n_trades']:,}")

if __name__ == "__main__":
    main()
