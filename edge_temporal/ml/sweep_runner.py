#!/usr/bin/env python3
import os, sys, csv, time, json, subprocess
from pathlib import Path
from itertools import product

# ------------ USER SETTINGS (edit if you want) ------------
ASOF = sys.argv[1] if len(sys.argv) > 1 else "2025-09-25"
CURATED_ROOT = os.environ.get("CURATED_ROOT", "/mnt/nvme/betfair-curated")
TRAIN_SH = "/opt/BetfairBotML/edge_temporal/bin/train_edge_temporal.sh"

# Sweep grids (keep modest at first; expand once stable)
PM_CUTOFFS   = [0.90, 0.92, 0.94]
PREOFF_MINSS = [5, 8, 12]
EDGE_THRESHS = [0.06, 0.08, 0.10]
LTP_BANDS    = [(2.2, 2.8), (2.4, 3.1), (2.6, 3.2)]

# Output base
OUT_BASE = Path("/opt/BetfairBotML/edge_temporal/output/sweeps") / ASOF
OUT_BASE.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_BASE / "summary.csv"

# Always evaluate both stake modes (the trainer does both internally).
# We’ll rank by stake_mode='flat' by default (safer).
RANK_STAKE_MODE = "flat"

def run_once(pm_cut, preoff, edge, ltp_min, ltp_max, idx):
    # Unique directory per config
    tag = f"pm{pm_cut:.2f}_pre{preoff}_edge{edge:.3f}_ltp{ltp_min:.1f}-{ltp_max:.1f}"
    run_out_dir = OUT_BASE / tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CURATED_ROOT"] = CURATED_ROOT
    env["PM_CUTOFF"] = f"{pm_cut:.2f}"
    env["PREOFF_MINS"] = str(preoff)
    env["EDGE_THRESH"] = f"{edge:.3f}"
    env["PER_MARKET_TOPK"] = "1"
    env["LTP_MIN"] = f"{ltp_min:.1f}"
    env["LTP_MAX"] = f"{ltp_max:.1f}"
    env["OUTPUT_DIR"] = str(run_out_dir)

    cmd = [TRAIN_SH, ASOF, "--fit-calib"]
    print(f"\n[{idx}] RUN {tag}")
    print("CMD:", " ".join(cmd))
    print("ENV overrides:", json.dumps({
        k: env[k] for k in ["PM_CUTOFF","PREOFF_MINS","EDGE_THRESH","PER_MARKET_TOPK","LTP_MIN","LTP_MAX","OUTPUT_DIR"]
    }, indent=2))

    # Run
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[WARN] Run failed (code={res.returncode}) for {tag}")
        return []

    # Collect sweep CSV
    sweep_path = run_out_dir / f"edge_sweep_{ASOF}.csv"
    if not sweep_path.exists():
        print(f"[WARN] Missing sweep file: {sweep_path}")
        return []

    rows = []
    with open(sweep_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # augment with our config
            r2 = dict(r)
            r2["pm_cutoff"] = pm_cut
            r2["preoff_mins"] = preoff
            r2["ltp_min"] = ltp_min
            r2["ltp_max"] = ltp_max
            r2["run_dir"] = str(run_out_dir)
            # Cast numerics defensively
            for k in ["roi","profit","n_trades","edge_thresh","topk","ltp_min","ltp_max"]:
                if k in r2 and r2[k] != "":
                    try:
                        r2[k] = float(r2[k]) if k not in ("n_trades","topk") else int(float(r2[k]))
                    except Exception:
                        pass
            rows.append(r2)
    return rows

def main():
    all_rows = []
    combos = list(product(PM_CUTOFFS, PREOFF_MINSS, EDGE_THRESHS, LTP_BANDS))
    print(f"Total runs: {len(combos)}")

    for i, (pm, pre, edge, band) in enumerate(combos, start=1):
        lmin, lmax = band
        rows = run_once(pm, pre, edge, lmin, lmax, i)
        all_rows.extend(rows)

    if not all_rows:
        print("No successful runs found.")
        return

    # Keep only stake_mode = RANK_STAKE_MODE for ranking (trainer writes 2 rows)
    ranked = [r for r in all_rows if str(r.get("stake_mode","")).startswith(RANK_STAKE_MODE)]

    # Sort by ROI desc, then by profit desc, then by n_trades asc
    ranked.sort(key=lambda r: (float(r.get("roi",-1e9)), float(r.get("profit",-1e9)), -int(r.get("n_trades", 0))), reverse=True)

    # Write summary
    fields = [
        "roi","profit","n_trades","stake_mode",
        "pm_cutoff","preoff_mins","edge_thresh","topk","ltp_min","ltp_max",
        "run_dir"
    ]
    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ranked:
            w.writerow({k: r.get(k,"") for k in fields})

    print(f"\nWrote summary → {SUMMARY_PATH}")
    print("\nTop 10 (stake_mode = %s):" % RANK_STAKE_MODE)
    for r in ranked[:10]:
        print(f"ROI={r['roi']:.3f}  profit={float(r['profit']):.0f}  n={int(r['n_trades'])}  "
              f"pm={r['pm_cutoff']:.2f} pre={int(r['preoff_mins'])} edge={float(r['edge_thresh']):.3f} "
              f"ltp={float(r['ltp_min']):.1f}-{float(r['ltp_max']):.1f}  dir={r['run_dir']}")

if __name__ == "__main__":
    main()
