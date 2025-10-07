#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
from itertools import product
from datetime import datetime, timedelta
import polars as pl

# --------------------- CLI ---------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep EV thresholds / staking / odds bands using precomputed EV parquet files"
    )
    # Data window (used only to pick which sim_YYYY-MM-DD.parquet files to read)
    p.add_argument("--curated", required=True)                    # kept for compatibility; not used
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--sport", default="horse-racing")             # kept for compatibility; not used
    p.add_argument("--preoff-max", type=int, default=30)          # kept for logs
    p.add_argument("--commission", type=float, default=0.02)      # not applied in EV proxy
    p.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    p.add_argument("--ev-mode", choices=["mtm","settlement"], default="mtm")

    # Model path is unused in this EV-only sweep (predictions already baked into EV files)
    p.add_argument("--model-path", default="/opt/BetfairBotML/train_price_trend/output/models/xgb_trend_reg.json")

    # IO / outputs
    p.add_argument("--base-output-dir", default="/opt/BetfairBotML/train_price_trend/output/stream")

    # Portfolio-ish knobs (applied as EV-only proxies)
    p.add_argument("--bankroll-nom", type=float, default=5000.0)
    p.add_argument("--kelly-cap", type=float, default=0.02)       # used as cap for kelly proxy sizing
    p.add_argument("--kelly-floor", type=float, default=0.001)    # used as min stake for kelly proxy
    p.add_argument("--batch-size", type=int, default=200_000)     # unused here; reserved
    p.add_argument("--min-trades", type=int, default=10_000, help="skip runs with too few trades")

    # Grids (CSV)
    p.add_argument("--edge-thresh", default="0.001")              # single or CSV
    p.add_argument("--stake-modes", default="flat,kelly")         # CSV of {flat, kelly}
    p.add_argument("--odds-bands", default="none,1.5:5.0,2.2:3.6")
    p.add_argument("--tag", default="", help="optional tag to nest sweep outputs")
    return p.parse_args()

# --------------------- Helpers ---------------------

def _parse_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()

def _daterange(start: str, end: str) -> list[str]:
    sd, ed = _parse_date(start), _parse_date(end)
    d = sd
    out = []
    while d <= ed:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out

def _parse_floats(csv: str) -> list[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]

def _parse_strs(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]

def _parse_bands(csv: str) -> list[tuple[float|None,float|None]]:
    out = []
    for token in _parse_strs(csv):
        t = token.lower()
        if t == "none":
            out.append((None, None))
        else:
            lo, hi = token.split(":")
            out.append((float(lo), float(hi)))
    return out

def _ev_file_dates(asof: str, start_date: str, valid_days: int) -> list[str]:
    """
    Pick the last <valid_days> calendar days between start_date..asof (inclusive).
    We expect sim_YYYY-MM-DD.parquet files for exactly these dates in base-output-dir.
    """
    all_days = _daterange(start_date, asof)
    if valid_days > 0 and len(all_days) > valid_days:
        all_days = all_days[-valid_days:]
    return all_days

def _load_day(base_dir: Path, day: str) -> pl.DataFrame:
    """
    Loads a single /stream/sim_YYYY-MM-DD.parquet produced by simulate_stream.py (EV-only).
    Expected columns include:
      marketId, selectionId, publishTimeMs, ltp_f, ev_per_1, mins_to_off, marketStartMs, ...
    """
    p = base_dir / f"sim_{day}.parquet"
    if not p.exists():
        return pl.DataFrame({})
    try:
        df = pl.read_parquet(p)
    except Exception:
        return pl.DataFrame({})
    # Ensure required columns exist
    need = {"marketId","selectionId","publishTimeMs","ltp_f","ev_per_1"}
    if not need.issubset(set(df.columns)):
        # try legacy names
        cols = {c.lower(): c for c in df.columns}
        miss = [c for c in need if c not in df.columns]
        if miss and {"ltp","ev_per_1"}.issubset(set(df.columns)):
            df = df.rename({"ltp":"ltp_f"})
    return df

def _apply_filters(df: pl.DataFrame, edge: float, odds_min: float|None, odds_max: float|None) -> pl.DataFrame:
    if df.is_empty():
        return df
    out = df
    # edge: keep rows with EV >= threshold
    out = out.filter(pl.col("ev_per_1").is_not_null() & (pl.col("ev_per_1") >= float(edge)))
    # odds band on ltp_f if provided
    if "ltp_f" in out.columns:
        if odds_min is not None:
            out = out.filter(pl.col("ltp_f") >= float(odds_min))
        if odds_max is not None:
            out = out.filter(pl.col("ltp_f") <= float(odds_max))
    return out

def _stake_flat(n: int) -> float:
    # 1 unit per trade
    return float(n)

def _stake_kelly_proxy(ev: pl.Series, cap: float, floor: float) -> float:
    """
    Kelly-like proxy using EV directly as fraction (clipped to [floor, cap]).
    This is NOT true Kelly (needs p & odds), but works to rank configs consistently
    when using expected EV per £1 as the signal.
    Returns the total 'effective stake' across trades.
    """
    vals = ev.to_numpy()
    import numpy as np
    fracs = np.clip(vals, floor, cap)  # negative EVs are already filtered out
    return float(fracs.sum())

def _summarize_config(
    base_dir: Path,
    days: list[str],
    edge: float,
    stake_mode: str,
    odds_min: float|None,
    odds_max: float|None,
    bankroll_nom: float,
    kelly_cap: float,
    kelly_floor: float,
) -> dict:
    """
    Loads EV rows per day, applies filters, computes EV-proxy totals.
    Returns a summary dict.
    """
    total_rows = 0
    total_trades = 0
    ev_sum = 0.0
    stake_equiv = 0.0

    for d in days:
        df = _load_day(base_dir, d)
        if df.is_empty():
            continue
        total_rows += int(df.height)
        filt = _apply_filters(df, edge, odds_min, odds_max)
        if filt.is_empty():
            continue
        n = int(filt.height)
        total_trades += n
        ev_sum += float(filt["ev_per_1"].sum())

        if stake_mode == "flat":
            stake_equiv += _stake_flat(n)
        elif stake_mode == "kelly":
            stake_equiv += _stake_kelly_proxy(filt["ev_per_1"], kelly_cap, kelly_floor)
        else:
            stake_equiv += _stake_flat(n)

    avg_ev = (ev_sum / float(total_trades)) if total_trades > 0 else None

    # Expected profit proxy:
    # flat: sum(ev_per_1)   (1 unit per trade)
    # kelly: sum(frac_i * ev_i), approximated by stake_equiv when we treat EV as fraction directly
    if stake_mode == "flat":
        total_exp_profit = ev_sum
    else:
        # Stake-equivalent already sums clipped fractions; multiply by mean EV as a simple proxy
        total_exp_profit = (avg_ev or 0.0) * stake_equiv

    overall_roi = (total_exp_profit / bankroll_nom) if bankroll_nom and bankroll_nom > 0 else None

    return {
        "n_total_rows_seen": total_rows,
        "n_trades": total_trades,
        "avg_ev_per_1": avg_ev,
        "total_exp_profit": total_exp_profit,
        "overall_roi": overall_roi,
        "edge_thresh": edge,
        "stake_mode": stake_mode,
        "odds_min": odds_min,
        "odds_max": odds_max,
    }

# --------------------- Main ---------------------

def main():
    a = parse_args()

    base_dir = Path(a.base_output_dir)
    sweeps_root = base_dir / "sweeps" / a.asof / (a.tag or "untagged")
    sweeps_root.mkdir(parents=True, exist_ok=True)

    # Determine validation days
    all_days = _daterange(a.start_date, a.asof)
    days = all_days[-a.valid_days:] if a.valid_days > 0 else all_days
    if not days:
        print("[sweep] No days selected.")
        return

    # Build grids
    edges  = _parse_floats(a.edge_thresh)
    stakes = [s.lower() for s in _parse_strs(a.stake_modes)]
    bands  = _parse_bands(a.odds_bands)

    trials_rows = []
    best = None  # (score, run_id, summary)

    combo_idx = 0
    for edge, stake, (omin, omax) in product(edges, stakes, bands):
        combo_idx += 1
        run_id = f"edge{edge:g}_{stake}"
        if omin is not None or omax is not None:
            run_id += f"_odds{(omin or 0):g}-{(omax or 999):g}"

        run_dir = sweeps_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[sweep] Running {run_id} …")
        summ = _summarize_config(
            base_dir=base_dir,
            days=days,
            edge=edge,
            stake_mode=stake,
            odds_min=omin,
            odds_max=omax,
            bankroll_nom=a.bankroll_nom,
            kelly_cap=a.kelly_cap,
            kelly_floor=a.kelly_floor,
        )

        # Skip tiny runs
        n_trades = int(summ.get("n_trades") or 0)
        if n_trades < a.min_trades:
            print(f"[sweep] SKIP {run_id} (n_trades={n_trades} < {a.min_trades})")
            continue

        # Write per-run summary
        (run_dir / f"summary_{a.asof}.json").write_text(json.dumps(summ, indent=2))

        # Score by overall_roi if present, else by avg_ev
        score = summ.get("overall_roi")
        if score is None:
            score = summ.get("avg_ev_per_1") or -1e12

        trials_rows.append({
            "run_id": run_id,
            "edge_thresh": edge,
            "stake_mode": stake,
            "odds_min": omin,
            "odds_max": omax,
            "n_trades": n_trades,
            "total_exp_profit": float(summ.get("total_exp_profit") or 0.0),
            "overall_roi": float(summ.get("overall_roi") or 0.0) if summ.get("overall_roi") is not None else None,
            "avg_ev_per_1": float(summ.get("avg_ev_per_1") or 0.0),
            "output_dir": str(run_dir),
        })

        # Track best
        if (best is None) or (float(score) > float(best[0])):
            best = (float(score), run_id, summ, run_dir)

    if not trials_rows:
        print("[sweep] No successful runs to record.")
        return

    # Save trials.csv
    trials = pl.DataFrame(trials_rows)
    trials = trials.sort(by=["overall_roi","total_exp_profit","avg_ev_per_1"], descending=[True, True, True])
    trials_path = sweeps_root / "trials.csv"
    trials.write_csv(trials_path)
    print(f"[sweep] Wrote {trials_path}")

    # Save best_config.json
    if best is not None:
        score, run_id, summ, run_dir = best
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
            "edge_thresh": float(summ["edge_thresh"]),
            "stake_mode": summ["stake_mode"],
            "odds_min": summ["odds_min"],
            "odds_max": summ["odds_max"],
            "n_trades": int(summ["n_trades"]),
            "total_exp_profit": float(summ["total_exp_profit"] or 0.0),
            "overall_roi": float(summ["overall_roi"] or 0.0) if summ["overall_roi"] is not None else None,
            "avg_ev_per_1": float(summ["avg_ev_per_1"] or 0.0),
            "output_dir": str(run_dir),
            "trials_csv": str(trials_path),
        }
        (sweeps_root / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
        print(f"[sweep] Best → {run_id}  score={score:.6g}")

if __name__ == "__main__":
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(300)
    main()
