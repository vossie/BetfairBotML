#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss

# ----------------------------
# Helpers
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser("train_edge_temporal.py")
    p.add_argument("train_start")
    p.add_argument("train_end")
    p.add_argument("valid_start")
    p.add_argument("valid_end")
    p.add_argument("--preoff-mins", type=int, default=30)
    p.add_argument("--downsample-secs", type=int, default=5)
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--edge-thresh", type=float, default=0.01)
    p.add_argument("--edge-prob", choices=["cal", "raw"], default="cal")
    p.add_argument("--no-sum-to-one", action="store_true")
    p.add_argument("--market-prob", choices=["overround"], default="overround")
    p.add_argument("--per-market-topk", type=int, default=1)
    p.add_argument("--side", choices=["back", "lay"], default="back")
    p.add_argument("--ltp-min", type=float, default=1.01)
    p.add_argument("--ltp-max", type=float, default=1000.0)
    p.add_argument("--stake", choices=["flat", "kelly"], default="flat")
    p.add_argument("--kelly-cap", type=float, default=0.05)
    p.add_argument("--kelly-floor", type=float, default=0.0)
    p.add_argument("--bankroll-nom", type=float, default=1000.0)
    p.add_argument("--pm-horizon-secs", type=int, default=300)
    p.add_argument("--pm-tick-threshold", type=int, default=1)
    p.add_argument("--pm-slack-secs", type=int, default=3)
    p.add_argument("--pm-cutoff", type=float, default=0.55)
    p.add_argument("--one-trade-per-market", action="store_true")
    return p.parse_args()


def safe_logloss(y_true, y_prob):
    p_clip = np.clip(y_prob, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))


def odds_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, np.nan)


def load_curated(train_start, train_end, valid_start, valid_end, preoff, downsample):
    # Placeholder: swap with your curated feature loader
    # Must return dict {split: DataFrame} with columns:
    #   sport, marketId, selectionId, publishTimeMs, features..., winLabel, odds
    rng = pd.date_range(train_start, valid_end, freq="5min")
    df = pd.DataFrame({
        "sport": "horse-racing",
        "marketId": np.random.choice(["m1", "m2"], size=len(rng)),
        "selectionId": np.random.choice([11, 22, 33], size=len(rng)),
        "publishTimeMs": rng.view(np.int64) // 1_000_000,
        "feature1": np.random.randn(len(rng)),
        "feature2": np.random.randn(len(rng)),
        "winLabel": np.random.binomial(1, 0.5, size=len(rng)),
        "odds": np.random.uniform(1.5, 8.0, size=len(rng)),
    })
    df_train = df[(df.index >= train_start) & (df.index <= train_end)]
    df_valid = df[(df.index >= valid_start) & (df.index <= valid_end)]
    return {"train": df_train, "valid": df_valid}


def train_xgb(X, y, evals):
    dtrain = xgb.DMatrix(X, label=y)
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda" if xgb.context().gpu_id is not None else "cpu",
    )
    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=10,
    )
    return bst


def backtest(df, args, probs):
    # compute edges
    inv_odds = odds_to_prob(df["odds"].values)
    edge = probs - inv_odds
    sel = (edge > args.edge_thresh) & (df["odds"] >= args.ltp_min) & (df["odds"] <= args.ltp_max)

    if args.one_trade_per_market:
        df_sel = df.loc[sel].copy()
        df_sel["edge"] = edge[sel]
        df_sel = df_sel.sort_values("edge", ascending=False).groupby("marketId").head(1)
        sel = df.index.isin(df_sel.index)

    trades = df.loc[sel].copy()
    trades["edge"] = edge[sel]
    trades["stake"] = 10.0
    if args.stake == "kelly":
        f = trades["edge"] / (trades["odds"] - 1.0)
        f = np.clip(f, args.kelly_floor, args.kelly_cap)
        trades["stake"] = f * args.bankroll_nom
    trades["profit"] = trades["stake"] * (trades["winLabel"] * (trades["odds"] - 1) - (1 - trades["winLabel"]))

    roi = trades["profit"].sum() / trades["stake"].sum() if len(trades) > 0 else 0.0

    # bucket ROI
    bins = [1.01, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 1000.0]
    trades["bucket"] = pd.cut(trades["odds"], bins)
    bucket_roi = trades.groupby("bucket").apply(
        lambda g: pd.Series(dict(
            n=len(g),
            hit=g["winLabel"].mean() if len(g) > 0 else np.nan,
            roi=g["profit"].sum() / g["stake"].sum() if g["stake"].sum() > 0 else np.nan,
        ))
    )

    return trades, roi, bucket_roi


def main():
    args = parse_args()

    # load
    data = load_curated(args.train_start, args.train_end,
                        args.valid_start, args.valid_end,
                        args.preoff_mins, args.downsample_secs)
    df_train, df_valid = data["train"], data["valid"]

    X_train = df_train[["feature1", "feature2"]].values
    y_train = df_train["winLabel"].values
    X_valid = df_valid[["feature1", "feature2"]].values
    y_valid = df_valid["winLabel"].values

    bst = train_xgb(X_train, y_train,
                    [(xgb.DMatrix(X_train, label=y_train), "train"),
                     (xgb.DMatrix(X_valid, label=y_valid), "valid")])

    # predict
    prob_valid = bst.predict(xgb.DMatrix(X_valid))
    logloss = safe_logloss(y_valid, prob_valid)
    auc = roc_auc_score(y_valid, prob_valid)
    print(f"[Value head] logloss={logloss:.4f} auc={auc:.3f} n={len(y_valid)}")

    # backtest
    trades, roi, bucket_roi = backtest(df_valid, args, prob_valid)
    print(f"[Backtest @ validation] n_trades={len(trades)} roi={roi:.4f} hit_rate={trades['winLabel'].mean():.3f}")

    print("\n[Value head — ROI by decimal odds bucket]")
    print(bucket_roi)

    # staking summary
    if len(trades) > 0:
        print("\n[Backtest — side summary]")
        print(f" side={args.side} n={len(trades)} avg_edge={trades['edge'].mean():.4f} avg_stake_gbp=£{trades['stake'].mean():.2f}")

        print("\n[Staking comparison]")
        flat_profit = trades.assign(stake=10.0).eval("stake * (winLabel*(odds-1)-(1-winLabel))").sum()
        print(f" Flat £10 stake → trades={len(trades)} staked=£{len(trades)*10:.2f} profit=£{flat_profit:.2f} roi={flat_profit/(len(trades)*10):.3f}")
        if args.stake == "kelly":
            print(f" Kelly (nom £{args.bankroll_nom}) → trades={len(trades)} staked=£{trades['stake'].sum():.2f} profit=£{trades['profit'].sum():.2f} roi={roi:.3f} avg_stake=£{trades['stake'].mean():.2f}")

    # save
    outdir = Path(os.environ.get("OUTPUT_DIR", "./output"))
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    recos_path = outdir / f"edge_recos_valid_{args.valid_end}_{ts}.csv"
    trades.to_csv(recos_path, index=False)
    print(f"Saved recommendations → {recos_path}")


if __name__ == "__main__":
    main()
