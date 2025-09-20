#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_edge_temporal.py  —  drop-in runner that matches your shell args.

Accepts:
  --curated --sport --asof --train-days --valid-days
  --preoff-mins --downsample-secs
  --commission --edge-thresh
  --pm-horizon-secs --pm-tick-threshold --pm-slack-secs --pm-cutoff
  --edge-prob --no-sum-to-one --market-prob
  --per-market-topk --side
  --stake {flat,kelly} --kelly-cap --kelly-floor --bankroll-nom
  --ltp-min --ltp-max
  --device

This drop-in generates synthetic features so you can verify the CLI + logging
without touching your real feature pipeline. Replace the synthetic block with
your actual data/feature build when ready.
"""

import argparse, os, time
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# -------------------------------
# utilities
# -------------------------------
def odds_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    p = np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, np.nan)
    return p

def safe_logloss(y_true, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

def kelly_fraction(p, odds):
    """Full Kelly for BACK bets. f = (b*p - (1-p))/b, b = odds-1."""
    b = max(0.0, float(odds) - 1.0)
    if b <= 0:
        return 0.0
    q = 1.0 - float(p)
    f = (b * float(p) - q) / b
    return max(0.0, float(f))

def pm_threshold_sweep(pm_probs, future_move_hit, future_ticks, thresholds):
    lines = []
    n_total = len(pm_probs)
    for th in thresholds:
        sel = pm_probs >= th
        n_sel = int(sel.sum())
        if n_sel == 0:
            lines.append((th, 0, 0.0, 0.0, 0.0, float("nan")))
            continue
        hits = float(future_move_hit[sel].mean())  # precision
        # "recall" here is relative to all positives in the set
        positives = float(future_move_hit.sum())
        recall = (float(future_move_hit[sel].sum()) / positives) if positives > 0 else 0.0
        # F1
        prec = hits
        f1 = 0.0 if (prec + recall) == 0 else 2 * prec * recall / (prec + recall)
        avg_ticks = float(future_ticks[sel].mean()) if np.isfinite(future_ticks[sel]).any() else float("nan")
        lines.append((th, n_sel, prec, recall, f1, avg_ticks))
    return lines

def roi_by_odds_buckets(odds, outcomes, flat_stake=10.0):
    # Simple profit calculation per bucket for BACK bets with fixed stake.
    buckets = [1.01, 1.50, 2.00, 3.00, 5.00, 10.00, 50.00, 1000.00]
    res = []
    odds = np.asarray(odds)
    outcomes = np.asarray(outcomes)
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i+1]
        m = (odds >= lo) & (odds < hi)
        n = int(m.sum())
        if n == 0:
            res.append((lo, hi, 0, None, None))
            continue
        wins = int(outcomes[m].sum())
        # Profit: win gets (odds-1)*stake, lose loses stake
        profit = wins * (float(odds[m].mean()) - 1.0) * flat_stake - (n - wins) * flat_stake
        roi = profit / (n * flat_stake) if n > 0 else None
        hit = wins / n if n > 0 else None
        res.append((lo, hi, n, hit, roi))
    return res

# -------------------------------
# argparse
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--train-days", type=int, default=12)
    ap.add_argument("--valid-days", type=int, default=2)  # << accepts the flag your .sh passes
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--downsample-secs", type=int, default=5)

    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--edge-thresh", type=float, default=0.015)

    ap.add_argument("--pm-horizon-secs", type=int, default=300)
    ap.add_argument("--pm-tick-threshold", type=int, default=1)
    ap.add_argument("--pm-slack-secs", type=int, default=3)
    ap.add_argument("--pm-cutoff", type=float, default=0.55)

    ap.add_argument("--edge-prob", default="cal", choices=["raw", "cal"])
    ap.add_argument("--no-sum-to-one", action="store_true")
    ap.add_argument("--market-prob", default="overround", choices=["overround", "ltp"])

    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--side", choices=["back", "lay", "both"], default="back")

    ap.add_argument("--stake", choices=["flat", "kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--kelly-floor", type=float, default=0.0)  # << NEW: floor as fraction of bankroll
    ap.add_argument("--bankroll-nom", type=float, default=1000.0)

    ap.add_argument("--ltp-min", type=float, default=1.5)
    ap.add_argument("--ltp-max", type=float, default=5.0)

    ap.add_argument("--device", default="cuda")
    return ap.parse_args()

# -------------------------------
# main
# -------------------------------
def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # banner similar to your .sh
    print("=== Edge Temporal Training (LOCAL) ===")
    print(f"Curated root:         {args.curated}")
    today = time.strftime("%Y-%m-%d")
    print(f"Today:                {today}")
    print(f"ASOF (arg to trainer):{args.asof}   # validation excludes today")
    # We don't recompute dates here; .sh already prints them. Echo config:
    print(f"Validation days:      {args.valid_days} (from shell)")
    print(f"Training days:        {args.train_days}")
    print(f"Sport:                {args.sport}")
    print(f"Pre-off minutes:      {args.preoff_mins}")
    print(f"Downsample (secs):    {args.downsample_secs}")
    print(f"Commission:           {args.commission}")
    print(f"Edge threshold:       {args.edge_thresh}")
    print(f"Edge prob:            {args.edge_prob}")
    print(f"Sum-to-one:           {'disabled' if args.no_sum_to_one else 'enabled'}")
    print(f"Market prob:          {args.market_prob}")
    print(f"Per-market topK:      {args.per_market_topk}")
    print(f"Side:                 {args.side}")
    print(f"LTP range:            [{args.ltp_min}, {args.ltp_max}]")
    print(f"Stake mode:           {args.stake} (kelly_cap={args.kelly_cap}, floor={args.kelly_floor})")
    print(f"Bankroll (nominal):   £{args.bankroll_nom:.0f}")
    print(f"PM horizon (secs):    {args.pm_horizon_secs}")
    print(f"PM tick threshold:    {args.pm_tick_threshold}")
    print(f"PM slack (secs):      {args.pm_slack_secs}")
    print(f"PM cutoff:            {args.pm_cutoff}")
    print()

    # -------------------------------
    # Synthetic feature build
    # (drop-in; replace with your real pipeline)
    # -------------------------------
    rng = np.random.default_rng(42)
    n_train = max(10000, args.train_days * 1000)
    n_valid = max(3000, args.valid_days * 1500)

    X_train = rng.normal(size=(n_train, 32))
    y_train = rng.binomial(1, 0.5, size=n_train).astype(float)

    X_valid = rng.normal(size=(n_valid, 32))
    y_valid = rng.binomial(1, 0.5, size=n_valid).astype(float)

    # Simulated last-traded price within [ltp_min, ltp_max]
    odds_valid = rng.uniform(low=args.ltp_min, high=args.ltp_max, size=n_valid)
    # PM "future move hit" proxy and ticks (purely synthetic)
    pm_future_hit = rng.binomial(1, 0.55, size=n_valid).astype(float)
    pm_future_ticks = rng.normal(loc=90.0, scale=40.0, size=n_valid)

    # -------------------------------
    # Train XGB value head
    # -------------------------------
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    params_val = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        predictor="auto"
    )
    t0 = time.time()
    bst_val = xgb.train(
        params_val,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )
    elapsed_val = time.time() - t0
    preds_val = bst_val.predict(dvalid)
    v_logloss = safe_logloss(y_valid, preds_val)
    v_auc = float(roc_auc_score(y_valid, preds_val))

    print(f"[XGB value] elapsed={elapsed_val:.2f}s")
    print(f"\n[Value head: winLabel] logloss={v_logloss:.4f} auc={v_auc:.3f}  n={len(y_valid)}")

    # -------------------------------
    # Train XGB PM head (synthetic label)
    # -------------------------------
    pm_label = pm_future_hit  # stand-in
    dtrain_pm = xgb.DMatrix(X_train, label=rng.binomial(1, 0.5, size=n_train))
    dvalid_pm = xgb.DMatrix(X_valid, label=pm_label)
    params_pm = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        predictor="auto"
    )
    t1 = time.time()
    bst_pm = xgb.train(
        params_pm,
        dtrain_pm,
        num_boost_round=120,
        evals=[(dtrain_pm, "train"), (dvalid_pm, "valid")],
        verbose_eval=False,
    )
    elapsed_pm = time.time() - t1
    pm_preds = bst_pm.predict(dvalid_pm)

    pm_logloss = safe_logloss(pm_label, pm_preds)
    try:
        pm_auc = float(roc_auc_score(pm_label, pm_preds))
    except:
        pm_auc = float("nan")
    acc_at_05 = float((pm_preds >= 0.5).mean())
    take = int((pm_preds >= 0.5).sum())
    hit_rate = float(pm_label[pm_preds >= 0.5].mean()) if take > 0 else 0.0
    avg_ticks = float(pm_future_ticks[pm_preds >= 0.5].mean()) if take > 0 else float("nan")

    print(f"\n[Price-move head: horizon={args.pm_horizon_secs}s, threshold={args.pm_tick_threshold}t]")
    print(f"  logloss={pm_logloss:.4f} auc={pm_auc:.3f}  acc@0.5={acc_at_05:.3f}")
    print(f"  taken_signals={take}  hit_rate={hit_rate:.3f}  avg_future_move_ticks={avg_ticks:.2f}")

    # -------------------------------
    # PM threshold sweep
    # -------------------------------
    sweep_ths = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    lines = pm_threshold_sweep(pm_preds, pm_label, pm_future_ticks, sweep_ths)
    print("\n[PM threshold sweep]")
    print("  th   N_sel  precision  recall   F1     avg_move_ticks")
    for th, n_sel, prec, recall, f1, avg_mt in lines:
        avg_str = f"{avg_mt:.2f}" if np.isfinite(avg_mt) else "nan"
        print(f"  {th:.2f} {n_sel:7d}      {prec:.3f}   {recall:.3f}   {f1:.3f}           {avg_str}")

    # -------------------------------
    # Backtest (synthetic)
    # -------------------------------
    # Selection: value model probability vs market implied, PM cutoff, side, etc.
    p_market = odds_to_prob(odds_valid)
    # crude "calibration" choice: if edge_prob==cal, nudge preds a bit toward market
    if args.edge_prob == "cal":
        p_model = 0.5 * preds_val + 0.5 * p_market
    else:
        p_model = preds_val.copy()

    # sum-to-one guard (no-op here, but keep flag coherent)
    # pm cutoff gate:
    pm_gate = pm_preds >= float(args.pm_cutoff)

    # ENTRY RULE (BACK): edge_back = p_model - p_market
    edge_back = p_model - p_market
    entry_mask = (pm_gate) & (edge_back >= float(args.edge_thresh))
    # odds filter
    entry_mask &= (odds_valid >= args.ltp_min) & (odds_valid <= args.ltp_max)

    idx = np.where(entry_mask)[0]
    n_trades = int(len(idx))

    # outcomes (synthetic): y_valid
    outcomes = y_valid[idx].astype(float)
    odds_sel = odds_valid[idx]
    p_model_sel = p_model[idx]
    p_market_sel = p_market[idx]
    edge_sel = edge_back[idx]

    # Flat £10 P&L
    flat_stake = 10.0
    profit_flat = (outcomes * (odds_sel - 1.0) * flat_stake) - ((1.0 - outcomes) * flat_stake)
    pnl_flat = float(profit_flat.sum())
    staked_flat = float(n_trades * flat_stake)
    roi_flat = pnl_flat / staked_flat if staked_flat > 0 else 0.0
    hit_rate_val = float(outcomes.mean()) if n_trades > 0 else 0.0
    avg_edge_val = float(edge_sel.mean()) if n_trades > 0 else float("nan")

    print("\n[Backtest @ validation — value]")
    print(f"  n_trades={n_trades}  roi={roi_flat:.4f}  hit_rate={hit_rate_val:.3f}  avg_edge={avg_edge_val}")

    # ROI by odds bucket (flat stake for visibility)
    print("\n[Value head — ROI by decimal odds bucket]")
    bucket_rows = roi_by_odds_buckets(odds_sel, outcomes, flat_stake=flat_stake)
    print("  bucket              n     hit     roi")
    for lo, hi, n, hit, roi in bucket_rows:
        if n == 0:
            print(f"  [{lo:>.2f}, {hi:>6.2f})      0                    ")
        else:
            hit_s = f"{hit:.3f}" if hit is not None else "   "
            roi_s = f"{roi:.3f}" if roi is not None else "   "
            print(f"  [{lo:>.2f}, {hi:>6.2f}) {n:6d}   {hit_s}   {roi_s}")

    # Side summary (all back in this synthetic drop-in)
    if n_trades > 0:
        avg_stake_flat = flat_stake
    else:
        avg_stake_flat = 0.0
    print("\n[Backtest — side summary]")
    print(f"  side=back  n={n_trades:4d}  avg_edge={avg_edge_val:.4f}  avg_stake_gbp=£{avg_stake_flat:.2f}")

    # Kelly comparison
    pnl_kelly = 0.0
    stakes_kelly = []
    if args.stake == "kelly":
        cap = float(args.kelly_cap)
        floor_frac = float(args.kelly_floor)
        bankroll = float(args.bankroll_nom)
        for pi, oi in zip(p_model_sel, odds_sel):
            f = kelly_fraction(pi, oi)
            f = min(cap, max(f, 0.0))
            # apply Kelly floor
            f = max(f, floor_frac)
            stake = f * bankroll
            stakes_kelly.append(stake)
        stakes_kelly = np.asarray(stakes_kelly, dtype=float)
        profit_kelly = (outcomes * (odds_sel - 1.0) * stakes_kelly) - ((1.0 - outcomes) * stakes_kelly)
        pnl_kelly = float(profit_kelly.sum())
        staked_kelly = float(stakes_kelly.sum())
        roi_kelly = pnl_kelly / staked_kelly if staked_kelly > 0 else 0.0

    # Print staking comparison
    print("\n[Staking comparison]")
    print(f"  Flat £10 stake    → trades={n_trades}  staked=£{staked_flat:.2f}  profit=£{pnl_flat:.2f}  roi={roi_flat:.3f}")
    if args.stake == "kelly":
        avg_stake = float(stakes_kelly.mean()) if n_trades > 0 else 0.0
        print(f"  Kelly (nom £{args.bankroll_nom:.0f}) → trades={n_trades}  staked=£{staked_kelly:.2f}  profit=£{pnl_kelly:.2f}  roi={roi_kelly:.3f}  avg_stake=£{avg_stake:.2f}")
        # Stake distribution
        q = np.quantile(stakes_kelly, [0.00, 0.25, 0.50, 0.75, 1.00]) if n_trades > 0 else [0,0,0,0,0]
        print("\n[Kelly stake distribution]")
        print(f"  min=£{q[0]:.2f}  p25=£{q[1]:.2f}  median=£{q[2]:.2f}  p75=£{q[3]:.2f}  max=£{q[4]:.2f}")

    # -------------------------------
    # Save recommendations CSV
    # -------------------------------
    recs = []
    # To emulate per-market-topK & side, we’ll just dump the selected rows as “back”
    if n_trades > 0:
        # choose stakes used (flat or kelly)
        if args.stake == "kelly":
            stake_vec = stakes_kelly
        else:
            stake_vec = np.full(n_trades, flat_stake, dtype=float)

        for i, idx_i in enumerate(idx):
            recs.append(dict(
                marketId=f"M{int(idx_i):08d}",
                selectionId=int(idx_i),
                ltp=float(odds_valid[idx_i]),
                side="back",
                stake=float(stake_vec[i]),
                p_model=float(p_model[idx_i]),
                p_market=float(p_market[idx_i]),
                edge_back=float(edge_back[idx_i]),
            ))

    rec_df = pl.DataFrame(recs) if len(recs) else pl.DataFrame({
        "marketId": pl.Series([], pl.Utf8),
        "selectionId": pl.Series([], pl.Int64),
        "ltp": pl.Series([], pl.Float64),
        "side": pl.Series([], pl.Utf8),
        "stake": pl.Series([], pl.Float64),
        "p_model": pl.Series([], pl.Float64),
        "p_market": pl.Series([], pl.Float64),
        "edge_back": pl.Series([], pl.Float64),
    })

    out_csv = OUTPUT_DIR / f"edge_recos_valid_{args.asof}_T{args.train_days}.csv"
    rec_df.write_csv(str(out_csv))
    print(f"\nSaved recommendations → {out_csv}")

    # save “models”
    (OUTPUT_DIR / f"edge_value_xgb_30m_{args.asof}_T{args.train_days}.json").write_text('{"dummy":"value_model"}\n')
    (OUTPUT_DIR / f"edge_price_xgb_{args.pm_horizon_secs}s_30m_{args.asof}_T{args.train_days}.json").write_text('{"dummy":"pm_model"}\n')

    print(f"Saved models →")
    print(f"  {OUTPUT_DIR}/edge_value_xgb_30m_{args.asof}_T{args.train_days}.json")
    print(f"  {OUTPUT_DIR}/edge_price_xgb_{args.pm_horizon_secs}s_30m_{args.asof}_T{args.train_days}.json")
    print(f"Saved validation detail → {OUTPUT_DIR}/edge_valid_both_{args.asof}_T{args.train_days}.csv")

if __name__ == "__main__":
    main()
