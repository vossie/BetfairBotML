# ml/service.py
from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

# --- Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/xgb_preoff.pkl")
FEATURE_LIST_ENV = os.getenv("FEATURE_LIST", "")
API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="Betfair ML Inference API", version="1.1.1")

_model = None
_feature_cols: Optional[List[str]] = None


def _load_model():
    global _model, _feature_cols
    if _model is not None:
        return
    try:
        import joblib
        art = joblib.load(MODEL_PATH)
        _model = art["model"]
        _feature_cols = art.get("features")
        return
    except Exception:
        pass
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        _model = model
        if FEATURE_LIST_ENV:
            _feature_cols = [c.strip() for c in FEATURE_LIST_ENV.split(",") if c.strip()]
        else:
            raise RuntimeError("FEATURE_LIST not set for xgb JSON model")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


def _require_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ---------- Schemas ----------

class FeatureRow(BaseModel):
    marketId: str
    selectionId: int
    ltp: Optional[float] = None
    mom_10s: Optional[float] = None
    mom_60s: Optional[float] = None
    vol_10s: Optional[float] = None
    vol_60s: Optional[float] = None
    spread_ticks: Optional[int] = Field(None, alias="spreadTicks")
    imbalanceBest1: Optional[float] = None
    tradedVolume: Optional[float] = None
    overround: Optional[float] = None
    norm_prob: Optional[float] = None
    rank_price: Optional[float] = None

    class Config:
        allow_population_by_field_name = True


class PredictRequest(BaseModel):
    rows: List[FeatureRow]
    feature_order: Optional[List[str]] = None
    kelly: Optional[float] = 0.125
    cap_ratio: Optional[float] = 0.02
    bankroll: Optional[float] = 1000.0
    market_cap: Optional[float] = 50.0
    commission: Optional[float] = 0.05


class Prediction(BaseModel):
    marketId: str
    selectionId: int
    p_win: float
    side: Optional[str] = None
    stake: Optional[float] = None
    ev_back: Optional[float] = None
    ev_lay: Optional[float] = None


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    features_used: List[str]


class DecideRequest(BaseModel):
    rows: List[FeatureRow]
    feature_order: Optional[List[str]] = None
    kelly: float = 0.125
    bankroll: float = 1000.0
    market_cap: float = 50.0
    commission: float = 0.05
    min_stake: float = 1.0
    min_edge: float = 0.0
    top_n_per_market: int = 1
    prefer_back: bool = True


class Decision(BaseModel):
    marketId: str
    selectionId: int
    action: str
    stake: float
    p_win: float
    odds: Optional[float] = None
    ev: float
    reason: Optional[str] = None


class DecideResponse(BaseModel):
    decisions: List[Decision]
    features_used: List[str]


# ---------- Helpers ----------

def _kelly_fraction(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    f = (o * p - 1.0) / np.maximum(o - 1.0, 1e-9)
    return np.clip(f, 0.0, 1.0)


def _pick_features(df: pl.DataFrame, explicit: Optional[List[str]], learned: Optional[List[str]]) -> List[str]:
    if explicit:
        return explicit
    if learned:
        return learned
    exclude = {"marketId", "selectionId"}
    return [c for c, dt in zip(df.columns, df.dtypes) if c not in exclude and getattr(dt, "is_numeric", lambda: False)()]


def _predict_proba_matrix(X: np.ndarray) -> np.ndarray:
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    pred = _model.predict(X)
    return 1.0 / (1.0 + np.exp(-pred))


# ---------- Endpoints ----------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: bool = Depends(_require_api_key)):
    _load_model()
    if not req.rows:
        raise HTTPException(400, "Empty request")
    df = pl.from_dicts([r.dict(by_alias=True) for r in req.rows])
    feat_cols = _pick_features(df, req.feature_order, _feature_cols)
    if not feat_cols:
        raise HTTPException(400, "No feature columns provided or discoverable")
    try:
        X = df.select(feat_cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)
    except Exception as e:
        raise HTTPException(400, f"Failed to build feature matrix: {e}")
    p = _predict_proba_matrix(X)
    preds = []
    for i, row in enumerate(req.rows):
        preds.append(Prediction(
            marketId=row.marketId,
            selectionId=row.selectionId,
            p_win=float(p[i])
        ))
    return PredictResponse(predictions=preds, features_used=feat_cols)


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest, _: bool = Depends(_require_api_key)):
    _load_model()
    if not req.rows:
        raise HTTPException(400, "Empty request")
    df = pl.from_dicts([r.dict(by_alias=True) for r in req.rows])
    feat_cols = _pick_features(df, req.feature_order, _feature_cols)
    if not feat_cols:
        raise HTTPException(400, "No feature columns provided or discoverable")
    try:
        X = df.select(feat_cols).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)
    except Exception as e:
        raise HTTPException(400, f"Failed to build feature matrix: {e}")
    p = _predict_proba_matrix(X)

    # For demo: trivial action (PLACE_BACK if p>0.5 else HOLD)
    decisions: List[Decision] = []
    for i, row in enumerate(req.rows):
        action = "PLACE_BACK" if p[i] > 0.5 else "HOLD"
        stake = req.min_stake if action != "HOLD" else 0.0
        decisions.append(Decision(
            marketId=row.marketId,
            selectionId=row.selectionId,
            action=action,
            stake=stake,
            p_win=float(p[i]),
            odds=row.ltp,
            ev=p[i] - 0.5,
            reason=None if action != "HOLD" else "below threshold"
        ))
    return DecideResponse(decisions=decisions, features_used=feat_cols)
