#!/usr/bin/env python3
import os, itertools
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score


# -------------------
# Helpers
# -------------------
def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--curated", required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--valid-days", type=int, default=3)
    p.add_argument("--sport", default="horse-racing")
    p.add_argument("--device", default="cuda")  # GPU default
    p.add_argument("--commission", type=float, default=0.02)
    p.add_argument("--bankroll-nom", type=float, default=1000.0)
    p.add_argument("--kelly-cap", type=float, default=0.05)
    p.add_argument("--kelly-floor", type=float, default=0.0)
    return p.parse_args()


def safe_logloss(y_true, y_pred):
    return log_loss(y_true, y_pred)


def make_params(device: str):
    return dict(
        max_depth=8,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        device=device,
    )


# -------------------
# Main
# -------------------
def main():
    args = parse_args()
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/opt/BetfairBotML/edge_temporal/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    asof_dt = parse_date(args.asof)
    start_dt = parse_date(args.start_date)

    train_end   = asof_dt - timedelta(days=args.valid_days)
    train_start = start_dt
    valid_start = train_end + timedelta(days=1)
    valid_end   = asof_dt

    print(f"Training window:   {train_start.date()} .. {train_end.date()}")
    print(f"Validation window: {valid_start.date()} .. {valid_end.date()}")

    curated = Path(args.curated)

    # -------------------
    # Load data
    # -------------------
    snap_lf = load_snapshots(curated, train_start, valid_end, args.sport)
    defs_lf = load_defs(curated, train_start, valid_end, args.sport)
    res_lf  = load_results(curated, train_start, valid_end, args.sport)

    res_lf = res_lf.select(["marketId", "selectionId", "winLabel"])
    df_all = join_snapshots_results(snap_lf, res_lf).join(defs_lf, on=["marketId", "selectionId"], how="left")

    # -------------------
    # Drop leakage columns
    # -------------------
    drop_cols = {"runnerStatus", "settledTimeMs", "eventId", "marketType"}
    df_all = df_all.drop([c for c in drop_cols if c in df_all.columns])

    # -------------------
    # Train / valid split
    # -------------------
    df_train = df_all.filter(
        (pl.col("publishTimeMs") >= int(train_start.timestamp() * 1000)) &
        (pl.col("publishTimeMs") <  int((train_end + timedelta(days=1)).timestamp() * 1000))
    )
    df_valid = df_all.filter(
        (pl.col("publishTimeMs") >= int(valid_start.timestamp() * 1000)) &
        (pl.col("publishTimeMs") <  int((valid_end + timedelta(days=1)).timestamp() * 1000))
    )

    feats = [c for c in df_all.columns if c not in ("winLabel", "sport", "marketId", "selectionId")]
    print(f"[rows] train={df_train.height:,}  valid={df_valid.height:,}  features={len(feats)}")
    print("Features:", feats)

    # -------------------
    # Train model
    # -------------------
    dtrain = xgb.DMatrix(df_train.select(feats).to_arrow(),
                         label=df_train["winLabel"].to_numpy().astype("float32"))
    dvalid = xgb.DMatrix(df_valid.select(feats).to_arrow(),
                         label=df_valid["winLabel"].to_numpy().astype("float32"))

    booster = xgb.train(make_params(args.device), dtrain, num_boost_round=500,
                        evals=[(dtrain, "train"), (dvalid, "valid")],
                        early_stopping_rounds=20)

    preds = booster.predict(dvalid)

    # -------------------
    # Metrics
    # -------------------
    y_valid = df_valid["winLabel"].to_numpy().astype("float32")
    odds    = df_valid["ltp"].to_numpy().astype("float32")
    p_model = preds.astype("float32")
    p_market = (1.0 / pl.Series(odds).clip_min(1e-12)).to_numpy().astype("float32")

    print(f"[Value] logloss={safe_logloss(y_valid, preds):.4f}  auc={roc_auc_score(y_valid, preds):.3f}")

    # -------------------
    # Sweep
    # -------------------
    EDGE_T = [float(os.environ.get("EDGE_THRESH", 0.015))]
    TOPK   = [int(os.environ.get("PER_MARKET_TOPK", 1))]
    LTP_W  = [(float(os.environ.get("LTP_MIN", 1.5)), float(os.environ.get("LTP_MAX", 5.0)))]
    STAKE  = [
        ("flat", None, None, args.bankroll_nom),
        ("kelly", args.kelly_cap, args.kelly_floor, args.bankroll_nom),
    ]

    recs = []
    for e, t, (lo, hi), (sm, cap, floor_, bank) in itertools.product(EDGE_T, TOPK, LTP_W, STAKE):
        m = evaluate(df_valid, odds, y_valid, p_model, p_market,
                     e, t, lo, hi, sm, cap or 0.0, floor_ or 0.0, bank or 1000.0, args.commission)
        m.update(dict(edge_thresh=e, topk=t, ltp_min=lo, ltp_max=hi, stake_mode=sm))
        recs.append(m)

    sweep = pl.DataFrame(recs).sort(["roi", "n_trades"], descending=[True, True])
    out = OUTPUT_DIR / f"edge_sweep_{args.asof}.csv"
    sweep.write_csv(str(out))
    print(f"sweep saved → {out}")
    print(sweep.select(["roi", "profit", "n_trades", "edge_thresh", "stake_mode"]).head(10))


if __name__ == "__main__":
    main()
