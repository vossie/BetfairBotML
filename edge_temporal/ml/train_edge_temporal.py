#!/usr/bin/env python3
import argparse, os, sys, time, json
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

# -------------------
# Helpers
# -------------------
def odds_to_prob(odds):
    inv = np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, np.nan)
    return inv

def bucketize_odds(odds, buckets=[1.01,1.5,2.0,3.0,5.0,10.0,50.0,1000.0]):
    out = []
    for i in range(len(buckets)-1):
        lo,hi = buckets[i], buckets[i+1]
        mask = (odds >= lo) & (odds < hi)
        out.append(((lo,hi),mask))
    return out

def kelly_fraction(p, q, odds):
    b = odds - 1.0
    frac = (b*p - q)/b
    return max(0.0, frac)

# -------------------
# Argparse
# -------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--sport", default="horse-racing")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--train-days", type=int, default=10)
    ap.add_argument("--preoff-mins", type=int, default=30)
    ap.add_argument("--downsample-secs", type=int, default=5)
    ap.add_argument("--commission", type=float, default=0.02)
    ap.add_argument("--edge-thresh", type=float, default=0.01)
    ap.add_argument("--pm-horizon-secs", type=int, default=300)
    ap.add_argument("--pm-tick-threshold", type=int, default=1)
    ap.add_argument("--pm-slack-secs", type=int, default=3)
    ap.add_argument("--edge-prob", default="raw")
    ap.add_argument("--no-sum-to-one", action="store_true")
    ap.add_argument("--market-prob", default="overround")
    ap.add_argument("--per-market-topk", type=int, default=1)
    ap.add_argument("--stake", choices=["flat","kelly"], default="flat")
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    ap.add_argument("--ltp-min", type=float, default=1.01)
    ap.add_argument("--ltp-max", type=float, default=1000.0)
    ap.add_argument("--side", choices=["back","lay","both"], default="back")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bankroll-nom", type=float, default=1000.0)
    ap.add_argument("--pm-cutoff", type=float, default=0.55)
    return ap.parse_args()

# -------------------
# Main
# -------------------
def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR","/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Config ===")
    for k,v in vars(args).items():
        print(f"{k}: {v}")

    # -------------------
    # Dummy data load
    # (replace with real parquet/feature build)
    # -------------------
    n_train, n_valid = 10000, 2000
    X_train = np.random.randn(n_train,20)
    y_train = np.random.binomial(1,0.5,size=n_train)
    X_valid = np.random.randn(n_valid,20)
    y_valid = np.random.binomial(1,0.5,size=n_valid)

    # -------------------
    # Train value model
    # -------------------
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dvalid = xgb.DMatrix(X_valid,label=y_valid)
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist")
    evallist = [(dtrain,"train"),(dvalid,"valid")]
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=evallist, verbose_eval=False)

    preds = bst.predict(dvalid)
    logloss = log_loss(y_valid, preds)
    auc = roc_auc_score(y_valid, preds)
    print(f"\n[Value head] logloss={logloss:.4f} auc={auc:.3f} n={len(y_valid)}")

    # -------------------
    # Backtest
    # -------------------
    odds = np.random.uniform(args.ltp_min, args.ltp_max, size=n_valid)
    sel = (preds > 0.5)
    stake_flat = 10.0
    bankroll_nom = args.bankroll_nom
    pnl_flat, pnl_kelly = 0.0, 0.0
    stakes_kelly = []

    for i in range(n_valid):
        if not sel[i]: continue
        implied = odds_to_prob(odds[i])
        edge = preds[i] - implied
        if edge < args.edge_thresh: continue
        # outcome
        outcome = y_valid[i]
        # flat £10
        win = (odds[i]-1.0)*stake_flat if outcome==1 else -stake_flat
        pnl_flat += win
        # kelly
        if args.stake=="kelly":
            f = kelly_fraction(preds[i], 1-preds[i], odds[i])
            f = min(args.kelly_cap, f)
            stake = max(0.0, f*bankroll_nom)
            stakes_kelly.append(stake)
            win_k = (odds[i]-1.0)*stake if outcome==1 else -stake
            pnl_kelly += win_k

    print("\n[Backtest]")
    print(f" Flat £10 → profit={pnl_flat:.2f}")
    if args.stake=="kelly":
        print(f" Kelly    → profit={pnl_kelly:.2f} avg_stake={np.mean(stakes_kelly):.2f}")

    # -------------------
    # ROI by odds bucket
    # -------------------
    print("\n[ROI by odds bucket]")
    for (lo,hi),mask in bucketize_odds(odds):
        if mask.sum()==0: continue
        wins = y_valid[mask].sum()
        n = mask.sum()
        roi = (wins*(np.mean(odds[mask])-1) - (n-wins)) / n
        print(f"  [{lo:.2f},{hi:.2f})  n={n:4d}  win_rate={wins/n:.3f}  roi={roi:.3f}")

    # -------------------
    # Save recos
    # -------------------
    recs = []
    for i in range(n_valid):
        if sel[i]:
            recs.append(dict(
                marketId="M"+str(i),
                selectionId=i,
                ltp=odds[i],
                side=args.side,
                stake=stake_flat if args.stake=="flat" else stakes_kelly[i] if i<len(stakes_kelly) else 0.0,
                p_model=float(preds[i]),
                p_market=float(odds_to_prob(odds[i])),
                edge_back=float(preds[i]-odds_to_prob(odds[i]))
            ))
    rec_df = pl.DataFrame(recs)
    rec_file = OUTPUT_DIR / f"edge_recos_valid_{args.asof}_T{args.train_days}.csv"
    rec_df.write_csv(str(rec_file))
    print(f"\nSaved recommendations → {rec_file}")

if __name__=="__main__":
    main()
