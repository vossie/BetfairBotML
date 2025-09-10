# ml/run_train.py
from __future__ import annotations
import os, sys, argparse
from datetime import datetime, timedelta, timezone
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(HERE)
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from ml.features import build_features_for_date
from ml import dataio

def daterange_ending_at(end_iso: str | None, days: int) -> list[str]:
    end = datetime.now(timezone.utc).date() if not end_iso else datetime.fromisoformat(end_iso).date()
    return [(end - timedelta(days=i)).isoformat() for i in reversed(range(days))]

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True, help="Use 's3://betfair-curated' (NOT s3a://)")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--outdir", default=os.path.join(HERE, "artifacts"))
    args = ap.parse_args()

    # Print available dates to help debug empties
    avail = {
        "orderbook": dataio.list_dates(args.curated, args.sport, "orderbook_snapshots_5s"),
        "defs":      dataio.list_dates(args.curated, args.sport, "market_definitions"),
        "results":   dataio.list_dates(args.curated, args.sport, "results"),
    }
    print("Available dates:", avail)

    dates = daterange_ending_at(args.date, args.days)
    print(f"Loading sport={args.sport}, dates={dates[0]}..{dates[-1]}")
    feats_list, labels_list = [], []

    for d in dates:
        print(f"  probing {d}")
        f, l = build_features_for_date(args.curated, args.sport, d)
        if f.height and l.height:
            feats_list.append(f); labels_list.append(l)
            print(f"    + feats={f.height}, labels={l.height}")
        else:
            print(f"    - empty")

    if not feats_list:
        raise RuntimeError("No features (check curated path scheme and S3 env vars; see README below).")

    feats = pl.concat(feats_list, how="diagonal_relaxed")
    labels = pl.concat(labels_list, how="diagonal_relaxed")
    df = feats.join(labels, on=["marketId","selectionId"], how="inner").drop_nulls(["winLabel"])
    print(f"Training set size: {df.height}")

    # Minimal model for now (keep your richer trainer if desired)
    from xgboost import XGBClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import log_loss
    import numpy as np

    FEATURES = ["ltp_odds","best_back_odds","best_lay_odds","best_back_size","best_lay_size",
                "back_overround","lay_overround","field_size"]

    X = df.select(FEATURES).fill_null(0.0).to_numpy()
    y = df["winLabel"].to_numpy().astype(int)
    groups = df["marketId"].to_numpy()

    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X,y,groups))
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    model = XGBClassifier(tree_method="hist", n_estimators=200, max_depth=6, learning_rate=0.05)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:,1]
    print("logloss:", log_loss(yva, p))

    # Simple EV (taker) and dump recs
    bb = df["best_back_odds"].to_numpy()[va]
    bl = df["best_lay_odds"].to_numpy()[va]
    ev_back = p*(bb-1) + (1-p)*(-1)
    ev_lay  = (1-p) + p*(-(bl-1))
    side = np.where(ev_back >= ev_lay, "BACK", "LAY")

    recs = pl.DataFrame({
        "marketId": df["marketId"][va],
        "selectionId": df["selectionId"][va],
        "side": side,
        "p": p,
        "ev_back": ev_back,
        "ev_lay": ev_lay
    })
    ensure_dir(args.outdir)
    out_csv = os.path.join(args.outdir, f"recs-{args.sport}-{dates[0]}_{dates[-1]}.csv")
    recs.write_csv(out_csv)
    print("Wrote", out_csv)

if __name__ == "__main__":
    main()
