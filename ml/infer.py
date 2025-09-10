# ml/infer.py
from __future__ import annotations
import numpy as np
import polars as pl
import joblib

def kelly_back(p, odds, frac_cap=0.1):
    # Full Kelly for back: f* = (p*(odds-1) - (1-p)) / (odds-1)
    b = odds - 1.0
    f = (p*b - (1.0 - p)) / b
    return float(max(0.0, min(frac_cap, f)))

def kelly_lay(p, odds, frac_cap=0.1):
    # For lay: edge on the "not win" side
    # Approximate stake fraction on unit bankroll (risk measured in liability terms)
    # You can refine for your bankroll/liability constraints.
    q = 1.0 - p
    b = (1.0 / (1.0 - 1.0/odds)) - 1.0  # notional
    f = (q*b - (1.0 - q)) / b
    return float(max(0.0, min(frac_cap, f)))

def decide_actions(df_feats: pl.DataFrame, p_win: np.ndarray, commission=0.05, bankroll=1000.0):
    back_odds = df_feats["best_back_odds"].to_numpy()
    lay_odds  = df_feats["best_lay_odds"].to_numpy()

    ev_back = p_win * (back_odds - 1.0) * (1.0 - commission) + (1.0 - p_win) * (-1.0)
    ev_lay  = (1.0 - p_win) * (1.0 - commission) + p_win * (-(lay_odds - 1.0))
    decisions = []
    for i in range(len(p_win)):
        if ev_back[i] <= 0.0 and ev_lay[i] <= 0.0:
            decisions.append({"action":"no-bet","stake":0.0,"ev":0.0})
            continue
        if ev_back[i] >= ev_lay[i]:
            f = kelly_back(p_win[i], back_odds[i])
            decisions.append({"action":"back","stake":round(f*bankroll, 2),"ev":float(ev_back[i])})
        else:
            f = kelly_lay(p_win[i], lay_odds[i])
            decisions.append({"action":"lay","stake":round(f*bankroll, 2),"ev":float(ev_lay[i])})
    return decisions

def load_and_infer(artifact_path: str, feats: pl.DataFrame):
    art = joblib.load(artifact_path)
    model, feature_cols = art["model"], art["features"]
    X = feats.select(feature_cols).to_numpy()
    p = model.predict_proba(X)[:,1]
    return p
