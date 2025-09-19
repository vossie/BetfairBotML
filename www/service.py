# ml/service.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import xgboost as xgb
from fastapi import FastAPI, Query
from pydantic import BaseModel

# ----------------------------
# Load model + features
# ----------------------------

MODEL_PATH = Path(__file__).resolve().parent.parent / "output" / "xgb_model_country.json"
FEATS_PATH = Path(__file__).resolve().parent.parent / "output" / "xgb_model_country.features.txt"

bst = xgb.Booster()
bst.load_model(str(MODEL_PATH))

FEATURES: List[str] = FEATS_PATH.read_text().splitlines()

app = FastAPI(title="BetfairBotML Service (with stake)")


# ----------------------------
# Request/response models
# ----------------------------

class RunnerFeatures(BaseModel):
    marketId: str
    selectionId: int
    tto_minutes: float
    ltp: float | None = None
    tradedVolume: float | None = None
    spreadTicks: int | None = None
    imbalanceBest1: float | None = None
    implied_prob: float | None = None
    mom_10s: float | None = None
    mom_60s: float | None = None
    vol_10s: float | None = None
    vol_60s: float | None = None
    countryCode: str | None = None


class DecisionResponse(BaseModel):
    marketId: str
    selectionId: int
    prob: float
    side: str
    price: float
    stake: float


# ----------------------------
# Helper functions
# ----------------------------

def _kelly_stake(prob: float, price: float, kelly_frac: float = 0.25, bankroll: float = 1.0) -> float:
    """
    Kelly formula:
      f* = (bp - q) / b
      stake = bankroll * f* * kelly_frac
    with b = price - 1
    """
    b = price - 1.0
    q = 1.0 - prob
    edge = b * prob - q
    if b <= 0 or edge <= 0:
        return 0.0
    f_star = edge / b
    stake = bankroll * f_star * kelly_frac
    return max(stake, 0.0)


def _map_country(df: pl.DataFrame) -> pl.DataFrame:
    if "countryCode" in df.columns:
        return df.with_columns(
            pl.col("countryCode").fill_null("UNK").cast(pl.Categorical).to_physical().alias("country_feat")
        )
    return df


# ----------------------------
# API
# ----------------------------

@app.post("/decide", response_model=DecisionResponse)
def decide(r: RunnerFeatures,
           min_edge: float = Query(0.02, description="Minimum edge to bet"),
           kelly: float = Query(0.25, description="Kelly fraction"),
           commission: float = Query(0.02, description="Betfair commission")):
    # Build single-row dataframe
    df = pl.DataFrame([r.dict()])
    df = _map_country(df)

    # Select feature array
    X = df.select(FEATURES).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)

    # Predict probability
    p = float(bst.predict(xgb.DMatrix(X))[0])

    # Implied fair price
    fair_price = 1.0 / max(p, 1e-8)
    side = "back" if fair_price < (r.ltp or fair_price) else "lay"
    price = float(r.ltp or fair_price)

    # Compute stake using Kelly
    stake = _kelly_stake(p, price, kelly_frac=kelly, bankroll=1.0)

    return DecisionResponse(
        marketId=r.marketId,
        selectionId=r.selectionId,
        prob=p,
        side=side,
        price=price,
        stake=round(stake, 2)  # 2dp for clarity
    )
