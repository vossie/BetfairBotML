# ml/service.py
from __future__ import annotations
import os
from typing import List, Optional, Tuple
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

# -------- Config (env) --------
MODEL_PATH = os.getenv("MODEL_PATH", "output/xgb_model.json")
FEATURE_LIST_ENV = os.getenv("FEATURE_LIST", "")  # comma-separated, in TRAIN order (if Booster)
API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="Betfair ML Inference API", version="1.3.0")

_model = None                # sklearn estimator OR xgboost Booster
_model_is_booster = False    # True if _model is xgboost.Booster
_feature_cols: Optional[List[str]] = None  # training feature order if known


def _load_model():
    """Load the model once. Supports joblib artifacts or XGBoost Booster JSON."""
    global _model, _feature_cols, _model_is_booster
    if _model is not None:
        return

    # Try joblib artifact first (sklearn-style)
    try:
        import joblib
        art = joblib.load(MODEL_PATH)
        _model = art["model"]
        _feature_cols = art.get("features")
        # Allow override by env if explicitly provided
        if FEATURE_LIST_ENV:
            _feature_cols = [c.strip() for c in FEATURE_LIST_ENV.split(",") if c.strip()]
        _model_is_booster = False
        return
    except Exception:
        pass

    # Fallback: raw XGBoost Booster JSON (what your trainer saves)
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        _model = booster
        _model_is_booster = True
        # For Booster, we usually need the exact feature order used at train:
        if FEATURE_LIST_ENV:
            _feature_cols = [c.strip() for c in FEATURE_LIST_ENV.split(",") if c.strip()]
        # If not set, we will accept request-provided feature_order; otherwise we must error.
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


def _require_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# -------- Schemas --------

class FeatureRow(BaseModel):
    marketId: str
    selectionId: int
    # Extend with whatever features your model uses:
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
    feature_order: Optional[List[str]] = None  # optional override order


class Prediction(BaseModel):
    marketId: str
    selectionId: int
    p_win: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    features_used: List[str]


class DecideRequest(BaseModel):
    rows: List[FeatureRow]
    feature_order: Optional[List[str]] = None  # optional override
    threshold: float = 0.5  # demo policy: PLACE_BACK if p >= threshold else HOLD


class Decision(BaseModel):
    marketId: str
    selectionId: int
    action: str           # PLACE_BACK | HOLD
    stake: float
    p_win: float
    odds: Optional[float] = None
    ev: float
    reason: Optional[str] = None


class DecideResponse(BaseModel):
    decisions: List[Decision]
    features_used: List[str]


# -------- Feature helpers --------

def _resolve_feature_order(
    df: pl.DataFrame,
    request_order: Optional[List[str]],
    learned_order: Optional[List[str]],
) -> List[str]:
    """
    Pick feature order:
      1) learned_order (FEATURE_LIST env or artifact)
      2) else request_order (from payload)
      3) else auto-detect numeric columns (excluding ids)
    """
    if learned_order:
        return learned_order
    if request_order:
        return request_order
    exclude = {"marketId", "selectionId"}
    return [c for c, dt in zip(df.columns, df.dtypes) if c not in exclude and getattr(dt, "is_numeric", lambda: False)()]


def _align_features(
    df: pl.DataFrame,
    ordered_features: List[str],
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    Ensure df has all ordered_features; add missing as zeros; ignore extras.
    Returns (df_aligned, used_features, missing_features).
    """
    missing = [c for c in ordered_features if c not in df.columns]
    for m in missing:
        df = df.with_columns(pl.lit(0.0).alias(m))
    used = ordered_features[:]
    # Keep id cols at hand if needed (not used in X input)
    return df.select(used + ["marketId", "selectionId"]), used, missing


# -------- Prediction helpers --------

def _predict_proba_matrix(X: np.ndarray) -> np.ndarray:
    """
    Predict P(win) for:
      - sklearn-like models with predict_proba
      - XGBoost Booster (requires DMatrix)
      - sklearn API XGBClassifier (predict_proba)
    """
    global _model, _model_is_booster
    if _model_is_booster:
        import xgboost as xgb
        dm = xgb.DMatrix(X)
        # objective=binary:logistic returns probabilities
        return _model.predict(dm)
    # sklearn-style
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    # raw margin -> logistic
    pred = _model.predict(X)
    return 1.0 / (1.0 + np.exp(-pred))


# -------- Endpoints --------

@app.get("/health")
def health(_: bool = Depends(_require_api_key)):
    try:
        _load_model()
        return {
            "status": "ok",
            "model_path": MODEL_PATH,
            "model_type": "xgboost.Booster" if _model_is_booster else type(_model).__name__,
            "features": _feature_cols or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model_load_failed: {e}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: bool = Depends(_require_api_key)):
    _load_model()
    if not req.rows:
        raise HTTPException(400, "Empty request")

    df = pl.from_dicts([r.dict(by_alias=True) for r in req.rows])

    order = _resolve_feature_order(df, req.feature_order, _feature_cols)
    # For Booster models: we must have either FEATURE_LIST or request feature_order.
    if _model_is_booster and not order:
        raise HTTPException(400, "Booster model requires FEATURE_LIST env or feature_order in request")

    try:
        df_aligned, used_features, _ = _align_features(df, order)
        X = df_aligned.select(used_features).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)
    except Exception as e:
        raise HTTPException(400, f"Failed to build feature matrix: {e}")

    p = _predict_proba_matrix(X)
    preds = [
        Prediction(marketId=r.marketId, selectionId=r.selectionId, p_win=float(p[i]))
        for i, r in enumerate(req.rows)
    ]
    return PredictResponse(predictions=preds, features_used=used_features)


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest, _: bool = Depends(_require_api_key)):
    _load_model()
    if not req.rows:
        raise HTTPException(400, "Empty request")

    df = pl.from_dicts([r.dict(by_alias=True) for r in req.rows])

    order = _resolve_feature_order(df, req.feature_order, _feature_cols)
    if _model_is_booster and not order:
        raise HTTPException(400, "Booster model requires FEATURE_LIST env or feature_order in request")

    try:
        df_aligned, used_features, _ = _align_features(df, order)
        X = df_aligned.select(used_features).fill_null(strategy="mean").to_numpy().astype(np.float32, copy=False)
    except Exception as e:
        raise HTTPException(400, f"Failed to build feature matrix: {e}")

    p = _predict_proba_matrix(X)

    # Simple decision policy: PLACE_BACK if p >= threshold, else HOLD; stake = £1
    decisions: List[Decision] = []
    thr = float(req.threshold)
    for i, row in enumerate(req.rows):
        pi = float(p[i])
        action = "PLACE_BACK" if pi >= thr else "HOLD"
        stake = 1.0 if action == "PLACE_BACK" else 0.0
        odds = row.ltp
        ev = pi - thr
        reason = None if action == "PLACE_BACK" else "below threshold"
        decisions.append(Decision(
            marketId=row.marketId,
            selectionId=row.selectionId,
            action=action,
            stake=stake,
            p_win=pi,
            odds=odds,
            ev=float(ev),
            reason=reason
        ))
    return DecideResponse(decisions=decisions, features_used=used_features)
