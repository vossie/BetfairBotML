#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time, math, shutil, glob
from pathlib import Path
from itertools import product
from typing import Optional, Tuple, Dict, List
import subprocess
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ CLI ============
def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive sweep with parallelism, timeouts, and live progress."
    )
    # Core data/sim args
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--preoff-max", type=int, default=30)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    p.add_argument("--model-path", required=True)
    p.add_argument("--base-output-dir", required=True)

    # Portfolio baseline
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)

    # Non-edge grids
    p.add_argument("--stake-modes", default="flat,kelly")
    p.add_argument("--odds-bands", default="1.5:5.0,2.2:3.6")
    p.add_argument("--ev-scale-grid", default="0.05,0.1,0.2")
    p.add_argument("--ev-cap-grid", default="0.05")
    p.add_argument("--exit-ticks-grid", default="0,1")
    p.add_argument("--topk-grid", default="1")
    p.add_argument("--budget-grid", default="5,10")
    p.add_argument("--liq-enforce-grid", default="1")
    p.add_argument("--min-fill-frac-grid", default="5.0")

    # Adaptive edge targeting
    p.add_argument("--edge-min", type=float, default=1e-4)
    p.add_argument("--edge-max", type=float, default=5e-3)
    p.add_argument("--edge-init-grid", default="0.0005,0.001,0.002")
    p.add_argument("--max-edge-iterations", type=int, default=4)
    p.add_argument("--target-trades-per-day-min", type=float, default=1000.0)
    p.add_argument("--target-trades-per-day-max", type=float, default=2000.0)
    p.add_argument("--min-trades", type=int, default=2000)

    # Performance / orchestration
    p.add_argument("--parallel", type=int, default=2, help="How many simulate_stream runs in parallel")
    p.add_argument("--timeout-secs", type=int, default=3600, help="Per-run timeout (seconds)")
    p.add_argument("--batch-size", type=int, default=200_000)

    # Optional sampling passthrough
    p.add_argument("--max-files-per-day", type=int)
    p.add_argument("--file-sample-mode", choices=["uniform","head","tail"])
    p.add_argument("--row-sample-secs", type=int)

    # EV density logging
    p.add_argument("--ev-hist-bins", default="0,0.0005,0.001,0.002,0.003,0.005,0.01")
    p.add_argument("--sample-ev-density", action="store_true")

    # Bookkeeping
    p.add_argument("--tag", default="tuned")
    return p.parse_args()

# ============ Helpers ============
def _csv_floats(s: str) -> List[float]: return [float(x.strip()) for x in s.split(",") if x.strip()]
def _csv_ints(s: str)   -> List[int]:   return [int(float(x.strip())) for x in s.split(",") if x.strip()]
def _csv_strs(s: str)   -> List[str]:   return [x.strip() for x in s.split(",") if x.strip()]

def _bands(csv: str) -> List[Tuple[Optional[float], Optional[float]]]:
    out=[]
    for token in _csv_strs(csv):
        if token.lower()=="none": out.append((None,None))
        else:
            lo,hi = token.split(":")
            out.append((float(lo), float(hi)))
    return out

def trades_per_day(summary: Dict, valid_days: int) -> float:
    return float(summary.get("n_trades") or 0.0) / max(1, valid_days)

def within(x, lo, hi): return (x>=lo) and (x<=hi)

def ev_density(parquet_glob: str, bins: List[float]) -> Optional[pl.DataFrame]:
    files = sorted(glob.glob(parquet_glob))
    if not files: return None
    try:
        lf = pl.scan_parquet(files).select(pl.col("ev_per_1"))
        df = lf.collect()
        if df.height == 0: return None
        hist = (pl.DataFrame({"bin": pl.cut(df["ev_per_1"], bins=bins)})
                .group_by("bin").len().rename({"len":"count"}))
        total = float(hist["count"].sum())
        return hist.with_columns((pl.col("count")/total).alias("pct"))
    except Exception:
        return None

def sim_cmd(a, run_dir: Path, edge, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff):
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("simulate_stream.py")),
           "--curated", a.curated,
           "--asof", a.asof, "--start-date", a.start_date, "--valid-days", str(a.valid_days),
           "--sport", a.sport, "--preoff-max", str(a.preoff_max), "--horizon-secs", "120", "--commission", str(a.commission),
           "--model-path", a.model_path,
           "--edge-thresh", f"{edge}",
           "--stake-mode", stake_mode, "--bankroll-nom", f"{a.bankroll_nom}",
           "--kelly-cap", f"{a.kelly_cap}", "--kelly-floor", f"{a.kelly_floor}",
           "--odds-min", f"{omin}" if omin is not None else "1.01",
           "--odds-max", f"{omax}" if omax is not None else "1000",
           "--per-market-topk", f"{topk}", "--per-market-budget", f"{budget}",
           "--exit-on-move-ticks", f"{exit_ticks}",
           "--ev-scale", f"{scale}", "--ev-cap", f"{cap}",
           "--device", a.device,
           "--output-dir", str(run_dir)]
    if liq:
        cmd += ["--enforce-liquidity", "--min-fill-frac", f"{mff}"]
    # sampling passthrough
    if a.max_files_per_day is not None:
        cmd += ["--max-files-per-day", str(a.max_files_per_day)]
    if a.file_sample_mode is not None:
        cmd += ["--file-sample-mode", a.file_sample_mode]
    if a.row_sample_secs is not None:
        cmd += ["--row-sample-secs", str(a.row_sample_secs)]
    return cmd

def run_one_sim(a, run_dir: Path, edge, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff, timeout_s: int) -> Optional[Dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = sim_cmd(a, run_dir, edge, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff)
    with open(run_dir/"stdout.log","w") as so, open(run_dir/"stderr.log","w") as se:
        try:
            subprocess.run(cmd, stdout=so, stderr=se, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            (run_dir/"TIMEOUT.txt").write_text(f"Timeout after {timeout_s}s\n")
            return None
    summ_path = run_dir / f"summary_{a.asof}.json"
    if not summ_path.exists(): return None
    try:
        return json.loads(summ_path.read_text())
    except Exception:
        return None

# ============ Adaptive edge search (sequential within one combo) ============
def adaptive_edge_for_combo(a, sweeps_root: Path, base_tag: str,
                            stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff,
                            target_lo, target_hi, e_lo, e_hi, probe_edges, max_iters, ev_bins, timeout_s) -> Tuple[Optional[Dict], Optional[Path], List[Dict]]:
    tried: List[Dict] = []
    best: Optional[Tuple[float, Dict, Path]] = None

    def record(e: float, summ: Dict, run_dir: Path):
        tpd = trades_per_day(summ, a.valid_days)
        summ["edge_thresh"] = e
        summ["trades_per_day"] = tpd
        row = {
            "run_id": f"{base_tag}_edge{e:g}",
            "edge_thresh": e,
            "stake_mode": stake_mode,
            "odds_min": omin, "odds_max": omax,
            "ev_scale_used": scale, "ev_cap": cap,
            "exit_on_move_ticks": exit_ticks, "per_market_topk": topk, "per_market_budget": budget,
            "enforce_liquidity": bool(liq), "min_fill_frac": mff,
            "n_trades": int(summ.get("n_trades", 0)), "trades_per_day": tpd,
            "overall_roi_real_mtm": summ.get("overall_roi_real_mtm"),
            "overall_roi_exp": summ.get("overall_roi_exp"),
            "avg_ev_per_1": summ.get("avg_ev_per_1"),
            "output_dir": str(run_dir)
        }
        tried.append(row)
        # choose by realised ROI first, else expected ROI
        score = summ.get("overall_roi_real_mtm")
        if score is None: score = summ.get("overall_roi_exp", -1e12)
        score = float(score) if score is not None else -1e12
        if within(tpd, target_lo, target_hi):
            nonlocal best
            if (best is None) or (score > best[0]):
                best = (score, summ, run_dir)

    # Seed probes
    for e in sorted({max(e_lo, min(e_hi, x)) for x in probe_edges}):
        run_dir = sweeps_root / f"{base_tag}_edge{e:g}"
        print(f"[sweep] {base_tag}: probe edge={e:g}")
        summ = run_one_sim(a, run_dir, e, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff, timeout_s)
        if summ: record(e, summ, run_dir)

    # Bisection if needed
    lo, hi = e_lo, e_hi
    it = 0
    while (best is None) and (it < max_iters):
        it += 1
        mid = (lo + hi) / 2.0
        e = max(e_lo, min(e_hi, mid))
        if any(abs(r["edge_thresh"] - e) < 1e-12 for r in tried):
            # nudge
            e = min(e_hi, e * 1.15)
            if any(abs(r["edge_thresh"] - e) < 1e-12 for r in tried):
                break
        run_dir = sweeps_root / f"{base_tag}_edge{e:g}"
        print(f"[sweep] {base_tag}: bisect#{it} edge={e:g}")
        summ = run_one_sim(a, run_dir, e, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff, timeout_s)
        if not summ:
            # push bounds slightly to keep moving
            lo = min(e_hi, lo * 1.1)
            continue
        record(e, summ, run_dir)
        tpd = tried[-1]["trades_per_day"]
        if tpd > target_hi:
            lo = max(lo, e)  # need higher edge
        elif tpd < target_lo:
            hi = min(hi, e)  # need lower edge

    return (best[1], best[2], tried) if best else (None, None, tried)

# ============ MAIN ============
def main():
    a = parse_args()
    # make args accessible
    a.stakes  = [s.lower() for s in _csv_strs(a.stake_modes)]
    a.bands   = _bands(a.odds_bands)
    a.scales  = _csv_floats(a.ev_scale_grid)
    a.caps    = _csv_floats(a.ev_cap_grid)
    a.exits   = _csv_ints(a.exit_ticks_grid)
    a.topks   = _csv_ints(a.topk_grid)
    a.budgets = _csv_floats(a.budget_grid)
    a.liqs    = _csv_ints(a.liq_enforce_grid)
    a.mffs    = _csv_floats(a.min_fill_frac_grid)
    a.probes  = _csv_floats(a.edge_init_grid)
    a.ev_bins = _csv_floats(a.ev_hist_bins)

    sweeps_root = Path(a.base_output_dir) / "sweeps" / a.asof / (a.tag or "tuned")
    sweeps_root.mkdir(parents=True, exist_ok=True)
    trials_path = sweeps_root / "trials.csv"

    # Prepare job list (each job = one parameter combo that will do its own mini adaptive edge)
    jobs = []
    for stake_mode, (omin, omax), scale, cap, exit_ticks, topk, budget, liq, mff in product(
        a.stakes, a.bands, a.scales, a.caps, a.exits, a.topks, a.budgets, a.liqs, a.mffs
    ):
        base_tag = f"{stake_mode}"
        if omin is not None or omax is not None:
            base_tag += f"_odds{(omin or 0):g}-{(omax or 999):g}"
        base_tag += f"_es{scale:g}_ec{cap:g}_x{exit_ticks}_k{topk}_b{int(budget)}_liq{liq}"
        jobs.append((base_tag, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff))

    print(f"[sweep] total combos: {len(jobs)}; parallel={a.parallel}; timeout/run={a.timeout_secs}s")

    all_rows: List[Dict] = []
    best_global: Optional[Tuple[float, Dict, Path]] = None

    # Parallel executor over combos (each combo tunes edge sequentially inside)
    def run_combo(job):
        base_tag, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff = job
        print(f"[sweep] Tuning → {base_tag}")
        best_summ, best_dir, tried = adaptive_edge_for_combo(
            a, sweeps_root, base_tag,
            stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff,
            target_lo=a.target_trades_per_day_min, target_hi=a.target_trades_per_day_max,
            e_lo=a.edge_min, e_hi=a.edge_max, probe_edges=a.probes, max_iters=a.max_edge_iterations,
            ev_bins=a.ev_bins, timeout_s=a.timeout_secs
        )
        return base_tag, best_summ, best_dir, tried

    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        futs = {ex.submit(run_combo, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            base_tag, best_summ, best_dir, tried = fut.result()
            # append tried rows and flush trials.csv incrementally
            all_rows.extend(tried)
            try:
                df = pl.DataFrame(all_rows)
                df.write_csv(trials_path)
            except Exception:
                pass
            if best_summ:
                score = best_summ.get("overall_roi_real_mtm")
                if score is None:
                    score = best_summ.get("overall_roi_exp", -1e12)
                score = float(score)
                if (best_global is None) or (score > best_global[0]):
                    best_global = (score, best_summ, best_dir)
            print(f"[sweep] progress {i}/{len(jobs)} — wrote trials.csv (rows={len(all_rows)})")

    if not all_rows:
        print("[sweep] No successful runs to record.")
        return

    # Final trials.csv sorted nicely
    df = pl.DataFrame(all_rows).with_columns([
        pl.col("overall_roi_real_mtm").fill_null(-1e12),
        pl.col("overall_roi_exp").fill_null(-1e12),
        pl.col("trades_per_day").fill_null(0.0),
    ]).sort(by=["overall_roi_real_mtm","overall_roi_exp","trades_per_day"], descending=[True, True, False])
    df.write_csv(trials_path)
    print(f"[sweep] Wrote {trials_path}")

    # Best config (guard: min trades)
    if best_global:
        score, summ, run_dir = best_global
        if int(summ.get("n_trades",0)) >= int(a.min_trades):
            best_cfg = {
                "asof": a.asof,
                "start_date": a.start_date,
                "valid_days": a.valid_days,
                "preoff_max": a.preoff_max,
                "commission": a.commission,
                "device": a.device,
                "bankroll_nom": a.bankroll_nom,
                "kelly_cap": a.kelly_cap,
                "kelly_floor": a.kelly_floor,
                "edge_thresh": float(summ.get("edge_thresh", 0.0)),
                "stake_mode": summ.get("stake_mode", "flat"),
                "odds_min": summ.get("odds_min", None),
                "odds_max": summ.get("odds_max", None),
                "enforce_liquidity_effective": bool(summ.get("enforce_liquidity_effective", True)),
                "liquidity_levels": int(summ.get("liquidity_levels", 1)),
                "min_fill_frac": float(summ.get("min_fill_frac", 5.0)),
                "per_market_topk": int(summ.get("per_market_topk", 1)),
                "per_market_budget": float(summ.get("per_market_budget", 10.0)),
                "exit_on_move_ticks": int(summ.get("exit_on_move_ticks", 0)),
                "ev_scale_used": float(summ.get("ev_scale_used", 0.05)),
                "ev_cap": float(summ.get("ev_cap", 0.05)),
                "n_trades": int(summ.get("n_trades", 0)),
                "trades_per_day": float(summ.get("trades_per_day", 0.0)),
                "total_exp_profit": float(summ.get("total_exp_profit", 0.0)),
                "overall_roi_real_mtm": float(summ.get("overall_roi_real_mtm", 0.0)),
                "overall_roi_exp": float(summ.get("overall_roi_exp", 0.0)),
                "avg_ev_per_1": float(summ.get("avg_ev_per_1", 0.0)),
                "output_dir": str(run_dir),
                "trials_csv": str(trials_path),
            }
            (sweeps_root / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
            print(f"[sweep] Best → edge{best_cfg['edge_thresh']}  tpd={best_cfg['trades_per_day']:.0f}  ROI={best_cfg['overall_roi_real_mtm']:.4f}")
        else:
            print("[sweep] No best config met min trades; see trials.csv")
    else:
        print("[sweep] No best found; see trials.csv")

if __name__ == "__main__":
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(300)
    main()
