#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, subprocess, sys, uuid
from pathlib import Path
from itertools import product
import polars as pl

def parse_args():
    p = argparse.ArgumentParser(description="Grid sweep over trading configs; rank by realised ROI (MTM)")
    # Core
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--base-output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")

    # Portfolio baseline
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=200_000)
    p.add_argument("--min-trades", type=int, default=5000, help="skip configs with too few trades")

    # Grids (CSV)
    p.add_argument("--edge-thresh", default="0.0005,0.001,0.002")
    p.add_argument("--stake-modes", default="flat,kelly")
    p.add_argument("--odds-bands", default="1.5:5.0,2.2:3.6")
    p.add_argument("--ev-scale-grid", default="0.03,0.05,0.08,0.12")
    p.add_argument("--ev-cap-grid", default="0.03,0.05")
    p.add_argument("--exit-ticks-grid", default="0,1")
    p.add_argument("--topk-grid", default="1")
    p.add_argument("--budget-grid", default="5,10")
    p.add_argument("--liq-enforce-grid", default="1")
    p.add_argument("--min-fill-frac-grid", default="5.0")
    p.add_argument("--tag", default="grid1")
    return p.parse_args()

def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_csv_ints(s: str) -> list[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]

def _parse_csv_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_bands(csv: str) -> list[tuple[float|None,float|None]]:
    out=[]
    for token in _parse_csv_strs(csv):
        if token.lower()=="none":
            out.append((None,None))
        else:
            lo,hi = token.split(":")
            out.append((float(lo), float(hi)))
    return out

def run_one(cmd: list[str], env: dict, run_dir: Path) -> tuple[dict|None, str|None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir/"stdout.log","w") as so, open(run_dir/"stderr.log","w") as se:
        proc = subprocess.run(cmd, stdout=so, stderr=se)
    summ_path = run_dir / f"summary_{env['ASOF']}.json"
    if not summ_path.exists():
        return None, f"missing summary: {summ_path}"
    try:
        summ = json.loads(summ_path.read_text())
        return summ, None
    except Exception as e:
        return None, f"bad summary json: {e}"

def main():
    a = parse_args()
    base = Path(a.base_output_dir)
    sweeps_root = base / "sweeps" / a.asof / (a.tag or "untagged")
    sweeps_root.mkdir(parents=True, exist_ok=True)

    edges   = _parse_csv_floats(a.edge_thresh)
    stakes  = [s.lower() for s in _parse_csv_strs(a.stake_modes)]
    bands   = _parse_bands(a.odds_bands)
    scales  = _parse_csv_floats(a.ev_scale_grid)
    caps    = _parse_csv_floats(a.ev_cap_grid)
    exits   = _parse_csv_ints(a.exit_ticks_grid)
    topks   = _parse_csv_ints(a.topk_grid)
    budgets = _parse_csv_floats(a.budget_grid)
    liqs    = _parse_csv_ints(a.liq_enforce_grid)
    minfills= _parse_csv_floats(a.min_fill_frac_grid)

    trials_rows = []
    best = None  # (score, run_id, summ, run_dir)

    for edge, stake, (omin, omax), scale, cap, exit_ticks, topk, budget, liq, mff in product(
        edges, stakes, bands, scales, caps, exits, topks, budgets, liqs, minfills
    ):
        tag = f"edge{edge:g}_{stake}"
        if omin is not None or omax is not None:
            tag += f"_odds{(omin or 0):g}-{(omax or 999):g}"
        tag += f"_es{scale:g}_ec{cap:g}_x{exit_ticks}_k{topk}_b{int(budget)}_liq{liq}"
        run_dir = sweeps_root / tag

        cmd = [
            sys.executable, str(Path(__file__).with_name("simulate_stream.py")),
            "--curated", a.curated,
            "--asof", a.asof, "--start-date", a.start_date, "--valid-days", str(a.valid_days),
            "--sport", a.sport, "--preoff-max", str(a.preoff_max), "--horizon-secs", "120", "--commission", str(a.commission),
            "--model-path", a.model_path,
            "--edge-thresh", f"{edge}",
            "--stake-mode", stake, "--bankroll-nom", f"{a.bankroll_nom}", "--kelly-cap", f"{a.kelly_cap}", "--kelly-floor", f"{a.kelly_floor}",
            "--odds-min", f"{omin}" if omin is not None else "1.01", "--odds-max", f"{omax}" if omax is not None else "1000",
            "--per-market-topk", f"{topk}", "--per-market-budget", f"{budget}",
            "--exit-on-move-ticks", f"{exit_ticks}",
            "--ev-scale", f"{scale}", "--ev-cap", f"{cap}",
            "--device", a.device,
            "--output-dir", str(run_dir),
        ]
        if liq:
            cmd.append("--enforce-liquidity")
            cmd += ["--min-fill-frac", f"{mff}"]

        env = dict(os.environ)
        env["ASOF"] = a.asof

        print(f"[sweep] Running {tag} …")
        summ, err = run_one(cmd, env, run_dir)
        if summ is None:
            print(f"[sweep] WARN {tag} failed ({err}).")
            continue

        n_trades = int(summ.get("n_trades") or 0)
        if n_trades < a.min_trades:
            print(f"[sweep] SKIP {tag} (n_trades={n_trades} < min_trades={a.min_trades})")
            continue

        # Extract metrics (prefer realised ROI)
        roi_real = summ.get("overall_roi_real_mtm")
        roi_exp  = summ.get("overall_roi_exp")
        avg_ev   = summ.get("avg_ev_per_1")
        total_exp= summ.get("total_exp_profit")

        trials_rows.append({
            "run_id": tag,
            "edge_thresh": edge,
            "stake_mode": stake,
            "odds_min": omin, "odds_max": omax,
            "ev_scale_used": scale, "ev_cap": cap,
            "exit_on_move_ticks": exit_ticks,
            "per_market_topk": topk, "per_market_budget": budget,
            "enforce_liquidity": bool(liq), "min_fill_frac": mff,
            "n_trades": n_trades,
            "overall_roi_real_mtm": float(roi_real) if roi_real is not None else None,
            "overall_roi_exp": float(roi_exp) if roi_exp is not None else None,
            "avg_ev_per_1": float(avg_ev) if avg_ev is not None else None,
            "total_exp_profit": float(total_exp) if total_exp is not None else None,
            "output_dir": str(run_dir),
        })

        # scoring: realised ROI first, else expected
        score = roi_real if roi_real is not None else (roi_exp if roi_exp is not None else -1e12)
        if (best is None) or (float(score) > float(best[0])):
            best = (float(score), tag, summ, run_dir)

    if not trials_rows:
        print("[sweep] No successful runs to record.")
        return

    trials = pl.DataFrame(trials_rows)
    # Sort by realised ROI desc, then by exp ROI desc, then by avg_ev
    trials = trials.sort(by=["overall_roi_real_mtm","overall_roi_exp","avg_ev_per_1"], descending=[True, True, True])
    trials_path = sweeps_root / "trials.csv"
    trials.write_csv(trials_path)
    print(f"[sweep] Wrote {trials_path}")

    if best is not None:
        score, tag, summ, run_dir = best
        best_cfg = {
            "asof": a.asof,
            "start_date": a.start_date,
            "valid_days": a.valid_days,
            "preoff_max": a.preoff_max,
            "commission": a.commission,
            "ev_mode": a.ev_mode,
            "device": a.device,
            "bankroll_nom": a.bankroll_nom,
            "kelly_cap": a.kelly_cap,
            "kelly_floor": a.kelly_floor,
            "edge_thresh": float(summ.get("edge_thresh", 0.0)),
            "stake_mode": summ.get("stake_mode", "flat"),
            "odds_min": summ.get("odds_min", None),
            "odds_max": summ.get("odds_max", None),
            "enforce_liquidity_effective": bool(summ.get("enforce_liquidity_effective", False)),
            "liquidity_levels": int(summ.get("liquidity_levels", 0)),
            "min_fill_frac": float(summ.get("min_fill_frac", 0.0)),
            "per_market_topk": int(summ.get("per_market_topk", 1)),
            "per_market_budget": float(summ.get("per_market_budget", 10.0)),
            "exit_on_move_ticks": int(summ.get("exit_on_move_ticks", 0)),
            "ev_scale_used": float(summ.get("ev_scale_used", scales[0] if scales else 0.05)),
            "ev_cap": float(summ.get("ev_cap", caps[0] if caps else 0.05)),
            "n_trades": int(summ.get("n_trades", 0)),
            "total_exp_profit": float(summ.get("total_exp_profit", 0.0)),
            "overall_roi": float(summ.get("overall_roi_exp", 0.0)),
            "overall_roi_real_mtm": float(summ.get("overall_roi_real_mtm", 0.0)),
            "avg_ev_per_1": float(summ.get("avg_ev_per_1", 0.0)),
            "output_dir": str(run_dir),
            "trials_csv": str(trials_path),
        }
        (sweeps_root / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
        print(f"[sweep] Best → {tag}  score={score:.6g}")

if __name__ == "__main__":
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(300)
    main()
