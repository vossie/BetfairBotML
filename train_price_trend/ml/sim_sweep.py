#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, math, glob, shutil
from pathlib import Path
from itertools import product
from typing import Tuple, Dict, List, Optional

import polars as pl

# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive sweep: tune edge_thresh per combo to hit target trades/day; rank by realised ROI (MTM)."
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
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")
    p.add_argument("--base-output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")
    p.add_argument("--batch-size", type=int, default=200_000)

    # Portfolio baseline
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.02)
    p.add_argument("--kelly-floor", type=float, default=0.001)

    # Non-edge grids we still sweep
    p.add_argument("--stake-modes", default="flat,kelly")
    p.add_argument("--odds-bands", default="1.5:5.0,2.2:3.6")
    p.add_argument("--ev-scale-grid", default="0.05,0.1,0.2")
    p.add_argument("--ev-cap-grid", default="0.05")
    p.add_argument("--exit-ticks-grid", default="0,1")
    p.add_argument("--topk-grid", default="1")
    p.add_argument("--budget-grid", default="5,10")
    p.add_argument("--liq-enforce-grid", default="1")
    p.add_argument("--min-fill-frac-grid", default="5.0")

    # Adaptive edge tuning controls
    p.add_argument("--edge-min", type=float, default=1e-4, help="Lower bound for edge search")
    p.add_argument("--edge-max", type=float, default=5e-3, help="Upper bound for edge search")
    p.add_argument("--edge-init-grid", default="0.0005,0.001,0.002", help="Seed probes before bisection")
    p.add_argument("--max-edge-iterations", type=int, default=6, help="Max bisection steps per combo")
    p.add_argument("--target-trades-per-day-min", type=float, default=1000.0)
    p.add_argument("--target-trades-per-day-max", type=float, default=2000.0)
    p.add_argument("--min-trades", type=int, default=2000, help="Absolute min trades over full window to keep a combo")

    # EV density logging
    p.add_argument("--ev-hist-bins", default="0,0.0005,0.001,0.002,0.003,0.005,0.01",
                   help="Comma-separated EV bin edges for histogram per combo")
    p.add_argument("--sample-ev-density", action="store_true",
                   help="Compute EV density from sim_*.parquet (costs extra I/O)")

    p.add_argument("--tag", default="tuned")
    return p.parse_args()

# ------------------------ Helpers ------------------------

def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_csv_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]

def _parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_bands(csv: str) -> List[Tuple[Optional[float], Optional[float]]]:
    out=[]
    for token in _parse_csv_strs(csv):
        if token.lower()=="none":
            out.append((None,None))
        else:
            lo,hi = token.split(":")
            out.append((float(lo), float(hi)))
    return out

def run_sim(
    a, run_dir: Path, edge_thresh: float,
    stake_mode: str, omin: Optional[float], omax: Optional[float],
    scale: float, cap: float, exit_ticks: int, topk: int, budget: float, liq: int, mff: float
) -> Optional[Dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).with_name("simulate_stream.py")),
        "--curated", a.curated,
        "--asof", a.asof, "--start-date", a.start_date, "--valid-days", str(a.valid_days),
        "--sport", a.sport, "--preoff-max", str(a.preoff_max), "--horizon-secs", "120", "--commission", str(a.commission),
        "--model-path", a.model_path,
        "--edge-thresh", f"{edge_thresh}",
        "--stake-mode", stake_mode, "--bankroll-nom", f"{a.bankroll_nom}", "--kelly-cap", f"{a.kelly_cap}", "--kelly-floor", f"{a.kelly_floor}",
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

    with open(run_dir/"stdout.log","w") as so, open(run_dir/"stderr.log","w") as se:
        subprocess.run(cmd, stdout=so, stderr=se)

    summ_path = run_dir / f"summary_{a.asof}.json"
    if not summ_path.exists():
        return None
    try:
        summ = json.loads(summ_path.read_text())
        return summ
    except Exception:
        return None

def trades_per_day(summary: Dict, valid_days: int) -> float:
    n = float(summary.get("n_trades") or 0.0)
    return n / max(1, valid_days)

def within_band(x: float, lo: float, hi: float) -> bool:
    return (x >= lo) and (x <= hi)

def ev_density(run_dir: Path, bins: List[float]) -> Optional[pl.DataFrame]:
    files = sorted(glob.glob(str(run_dir / "sim_*.parquet")))
    if not files:
        return None
    try:
        lf = pl.scan_parquet(files).select(pl.col("ev_per_1"))
        df = lf.collect()
        if df.height == 0:
            return None
        ev = df["ev_per_1"]
        # Bin
        cuts = pl.cut(ev, bins=bins, labels=None, left_closed=True)
        hist = pl.DataFrame({"bin": cuts}).group_by("bin").len().rename({"len":"count"})
        total = float(hist["count"].sum())
        hist = hist.with_columns((pl.col("count")/total).alias("pct"))
        # Ensure all bins appear
        # labels like "[0, 0.0005)"
        return hist
    except Exception:
        return None

def write_csv(df: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)

# ------------------------ Adaptive edge search ------------------------

def home_in_edge(
    a, sweeps_root: Path, tag_prefix: str,
    stake_mode: str, omin: Optional[float], omax: Optional[float],
    scale: float, cap: float, exit_ticks: int, topk: int, budget: float, liq: int, mff: float,
    target_lo: float, target_hi: float, edge_lo: float, edge_hi: float,
    probe_edges: List[float], max_iters: int, ev_bins: List[float], sample_ev: bool
):
    """
    Returns: (best_summary, best_run_dir, tried_rows[list])
    Picks the best summary inside target band by realised ROI (fallback to expected ROI).
    """
    tried: List[Dict] = []
    best = None  # (score, summary, run_dir)

    def score_from(s):
        r = s.get("overall_roi_real_mtm")
        if r is None:
            r = s.get("overall_roi_exp", -1e12)
        return float(r)

    # Ensure we probe within [edge_lo, edge_hi]
    probes = sorted({max(edge_lo, min(edge_hi, e)) for e in probe_edges})
    lo, hi = edge_lo, edge_hi
    iters = 0

    # Run probes
    for e in probes:
        run_id = f"{tag_prefix}_edge{e:g}"
        run_dir = sweeps_root / run_id
        summ = run_sim(a, run_dir, e, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff)
        if summ is None:
            continue
        tpd = trades_per_day(summ, a.valid_days)
        summ["edge_thresh"] = e
        summ["trades_per_day"] = tpd
        # Optional EV density
        if sample_ev:
            hist = ev_density(run_dir, ev_bins)
            if hist is not None:
                write_csv(hist, run_dir / "ev_hist.csv")
        tried.append({
            "run_id": run_id, "edge_thresh": e,
            "stake_mode": stake_mode, "odds_min": omin, "odds_max": omax,
            "ev_scale_used": scale, "ev_cap": cap,
            "exit_on_move_ticks": exit_ticks, "per_market_topk": topk, "per_market_budget": budget,
            "enforce_liquidity": bool(liq), "min_fill_frac": mff,
            "n_trades": int(summ.get("n_trades", 0)), "trades_per_day": tpd,
            "overall_roi_real_mtm": summ.get("overall_roi_real_mtm"),
            "overall_roi_exp": summ.get("overall_roi_exp"),
            "avg_ev_per_1": summ.get("avg_ev_per_1"),
            "output_dir": str(run_dir),
        })
        if within_band(tpd, target_lo, target_hi):
            sc = score_from(summ)
            if (best is None) or (sc > best[0]):
                best = (sc, summ, run_dir)

    # Bisection to hit band if probes miss
    # We'll push the edge up if tpd is too high, down if too low.
    # Start midpoint from latest probe extremes if available, else [lo,hi]
    bounds = [lo, hi]
    iters = 0
    while iters < max_iters and best is None:
        iters += 1
        mid = (bounds[0] + bounds[1]) / 2.0
        e = max(edge_lo, min(edge_hi, mid))
        run_id = f"{tag_prefix}_edge{e:g}"
        run_dir = sweeps_root / run_id

        # Avoid duplicate run
        if any(abs(r["edge_thresh"] - e) < 1e-12 for r in tried):
            # Adjust bounds a bit to avoid stalling
            mid = (mid + bounds[1]) / 2.0
            e = max(edge_lo, min(edge_hi, mid))
            run_id = f"{tag_prefix}_edge{e:g}"
            run_dir = sweeps_root / run_id
            if any(abs(r["edge_thresh"] - e) < 1e-12 for r in tried):
                break

        summ = run_sim(a, run_dir, e, stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff)
        if summ is None:
            # Nudge bounds to progress
            bounds[0] = (bounds[0] * 1.1)
            continue

        tpd = trades_per_day(summ, a.valid_days)
        summ["edge_thresh"] = e
        summ["trades_per_day"] = tpd

        if sample_ev:
            hist = ev_density(run_dir, ev_bins)
            if hist is not None:
                write_csv(hist, run_dir / "ev_hist.csv")

        tried.append({
            "run_id": run_id, "edge_thresh": e,
            "stake_mode": stake_mode, "odds_min": omin, "odds_max": omax,
            "ev_scale_used": scale, "ev_cap": cap,
            "exit_on_move_ticks": exit_ticks, "per_market_topk": topk, "per_market_budget": budget,
            "enforce_liquidity": bool(liq), "min_fill_frac": mff,
            "n_trades": int(summ.get("n_trades", 0)), "trades_per_day": tpd,
            "overall_roi_real_mtm": summ.get("overall_roi_real_mtm"),
            "overall_roi_exp": summ.get("overall_roi_exp"),
            "avg_ev_per_1": summ.get("avg_ev_per_1"),
            "output_dir": str(run_dir),
        })

        if within_band(tpd, target_lo, target_hi):
            sc = score_from(summ)
            if (best is None) or (sc > best[0]):
                best = (sc, summ, run_dir)

        # Update bounds: if too many trades, increase edge; else decrease
        if tpd > target_hi:
            bounds[0] = max(bounds[0], e)  # raise lower bound
        elif tpd < target_lo:
            bounds[1] = min(bounds[1], e)  # lower upper bound
        else:
            # already handled
            pass

    return (best[1], best[2], tried) if best else (None, None, tried)

# ------------------------ Main ------------------------

def main():
    a = parse_args()

    sweeps_root = Path(a.base_output_dir) / "sweeps" / a.asof / (a.tag or "tuned")
    sweeps_root.mkdir(parents=True, exist_ok=True)

    stakes  = [s.lower() for s in _parse_csv_strs(a.stake_modes)]
    bands   = _parse_bands(a.odds_bands)
    scales  = _parse_csv_floats(a.ev_scale_grid)
    caps    = _parse_csv_floats(a.ev_cap_grid)
    exits   = _parse_csv_ints(a.exit_ticks_grid)
    topks   = _parse_csv_ints(a.topk_grid)
    budgets = _parse_csv_floats(a.budget_grid)
    liqs    = _parse_csv_ints(a.liq_enforce_grid)
    minfills= _parse_csv_floats(a.min_fill_frac_grid)

    probe_edges = _parse_csv_floats(a.edge_init_grid)
    ev_bins = _parse_csv_floats(a.ev_hist_bins)

    all_trials: List[Dict] = []
    global_best = None  # (score, summary, run_dir)

    for stake_mode, (omin, omax), scale, cap, exit_ticks, topk, budget, liq, mff in product(
        stakes, bands, scales, caps, exits, topks, budgets, liqs, minfills
    ):
        tag_prefix = f"{stake_mode}"
        if omin is not None or omax is not None:
            tag_prefix += f"_odds{(omin or 0):g}-{(omax or 999):g}"
        tag_prefix += f"_es{scale:g}_ec{cap:g}_x{exit_ticks}_k{topk}_b{int(budget)}_liq{liq}"

        print(f"[sweep] Tuning edge for combo: {tag_prefix}")
        best_summ, best_dir, tried = home_in_edge(
            a, sweeps_root, tag_prefix,
            stake_mode, omin, omax, scale, cap, exit_ticks, topk, budget, liq, mff,
            target_lo=a.target_trades_per_day_min, target_hi=a.target_trades_per_day_max,
            edge_lo=a.edge_min, edge_hi=a.edge_max,
            probe_edges=probe_edges, max_iters=a.max_edge_iterations,
            ev_bins=ev_bins, sample_ev=a.sample_ev_density
        )

        # Record all tried runs
        all_trials.extend(tried)

        # Update global best (within band only; else we'll rely on min_trades guard)
        if best_summ:
            score = best_summ.get("overall_roi_real_mtm")
            if score is None:
                score = best_summ.get("overall_roi_exp", -1e12)
            sc = float(score)
            if (global_best is None) or (sc > global_best[0]):
                global_best = (sc, best_summ, best_dir)

    if not all_trials:
        print("[sweep] No successful runs to record.")
        return

    trials_df = pl.DataFrame(all_trials)
    trials_df = trials_df.sort(by=["overall_roi_real_mtm","overall_roi_exp","trades_per_day"], descending=[True, True, False])
    trials_path = sweeps_root / "trials.csv"
    write_csv(trials_df, trials_path)
    print(f"[sweep] Wrote {trials_path}")

    # Pick best: prefer within trade band; fallback to highest realised ROI with >= min_trades
    best_cfg = None
    if global_best is not None:
        _, summ, run_dir = global_best
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
                "ev_scale_used": float(summ.get("ev_scale_used", scales[0] if scales else 0.05)),
                "ev_cap": float(summ.get("ev_cap", caps[0] if caps else 0.05)),

                "n_trades": int(summ.get("n_trades", 0)),
                "trades_per_day": float(summ.get("trades_per_day", 0.0)),
                "total_exp_profit": float(summ.get("total_exp_profit", 0.0)),
                "overall_roi_real_mtm": float(summ.get("overall_roi_real_mtm", 0.0)),
                "overall_roi_exp": float(summ.get("overall_roi_exp", 0.0)),
                "avg_ev_per_1": float(summ.get("avg_ev_per_1", 0.0)),

                "output_dir": str(run_dir),
                "trials_csv": str(trials_path),
            }

    if best_cfg:
        (sweeps_root / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
        print(f"[sweep] Best → edge{best_cfg['edge_thresh']}  tpd={best_cfg['trades_per_day']:.0f}  ROI={best_cfg['overall_roi_real_mtm']:.4f}")
    else:
        print("[sweep] No best config met guardrails; see trials.csv for details.")

if __name__ == "__main__":
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(300)
    main()
