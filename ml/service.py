# ml/service.py
from __future__ import annotations
import os, json
from typing import List, Optional, Dict, Any
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

# --- Model loading: supports (A) joblib {"model", "features"} or (B) xgb JSON + FEATURE_LIST env ---
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/xgb_preoff.pkl")  # joblib by default
FEATURE_LIST_ENV = os.getenv("FEATURE_LIST", "")  # comma-separated list if using xgb json
API_KEY = os.getenv("API_KEY", "")                # simple auth (optional)

app = FastAPI(title="Betfair ML Inference API", version="1.0.0")

# Lazy imports to reduce cold start time
_model = None
_feature_cols: Optional[List[str]] = None
_use_proba_index: Optional[int] = 1  # binary positive class index


def _load_model():
    global _model, _feature_cols, _use_proba_index
    if _model is not None:
        return
    # Try joblib artifact first
    try:
        import joblib
        art = joblib.load(MODEL_PATH)
        _model = art["model"]
        _feature_cols = art.get("features")
        return
    except Exception:
        pass

    # Fallback: XGBoost JSON
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)  # MODEL_PATH should be xgb_model.json
        _model = model
        if FEATURE_LIST_ENV:
            _feature_cols = [c.strip() for c in FEATURE_LIST_ENV.split(",") if c.strip()]
        else:
            raise RuntimeError("FEATURE_LIST env not set; cannot infer features for xgb JSON model")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


def _require_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ---------- Request / Response schemas ----------

class FeatureRow(BaseModel):
    # identifiers (echoed back)
    marketId: str
    selectionId: int

    # minimal features; you can send more – we’ll pick intersection with model’s features
    # Include all numeric columns your model expects; examples:
    ltp: Optional[float] = None
    mom_10s: Optional[float] = None
    mom_60s: Optional[float] = None
    vol_10s: Optional[float] = None
    vol_60s: Optional[float] = None
    spread_ticks: Optional[int] = Field(None, alias="spreadTicks")
    imb1: Optional[float] = Field(None, alias="imbalanceBest1")
    traded_vol: Optional[float] = Field(None, alias="tradedVolume")
    overround: Optional[float] = None
    norm_prob: Optional[float] = None
    rank_price: Optional[float] = None

    class Config:
        allow_population_by_field_name = True  # accept either spreadTicks or spread_ticks


class PredictRequest(BaseModel):
    rows: List[FeatureRow]
    # if you prefer to send feature names explicitly, include this:
    feature_order: Optional[List[str]] = None
    # optional Kelly config
    kelly: Optional[float] = 0.125
    cap_ratio: Optional[float] = 0.02
    bankroll: Optional[float] = 1000.0
    market_cap: Optional[float] = 50.0
    commission: Optional[float] = 0.05  # exchange commission


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


# ---------- Simple Kelly helpers (same math you use offline) ----------

def _kelly_fraction(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    # Kelly for BACK @ decimal odds o: (o*p - 1)/(o - 1), clipped
    f = (o * p - 1.0) / (o - 1.0)
    return np.clip(f, 0.0, 1.0)

def _recommend(df: pl.DataFrame, p: np.ndarray,
               kelly: float, cap_ratio: float, bankroll: float, market_cap: float, commission: float):
    # Odds from ltp; clip to plausible range
    o = np.clip(df["ltp"].fill_null(1.01).to_numpy(), 1.01, 1000.0)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    f_back = _kelly_fraction(p, o) * kelly
    o_lay = o / (o - 1.0)
    p_lay = 1.0 - p
    f_lay = _kelly_fraction(p_lay, o_lay) * kelly

    max_stake = bankroll * cap_ratio
    stake_back = np.minimum(f_back * bankroll, max_stake)
    stake_lay = np.minimum(f_lay * bankroll, max_stake)

    ev_back = p * (o - 1) * (1.0 - commission) - (1 - p)
    ev_lay = (1 - p) * (1.0 - commission) - p * (o - 1)  # rough liability-based EV

    choose_back = ev_back >= ev_lay
    side = np.where(choose_back, "BACK", "LAY")
    stake = np.where(choose_back, stake_back, stake_lay)

    # per-market cap
    pdf = df.select(["marketId", "selectionId"]).to_pandas()
    pdf["side"] = side
    pdf["stake"] = stake
    pdf["ev_back"] = ev_back
    pdf["ev_lay"] = ev_lay

    for mid, idx in pdf.groupby("marketId").groups.items():
        s = float(pdf.loc[idx, "stake"].sum())
        if s > market_cap:
            scale = market_cap / s
            pdf.loc[idx, "stake"] *= scale

    return pdf["side"].to_numpy(), pdf["stake"].to_numpy(), ev_back, ev_lay


# ---------- Inference ----------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: bool = Depends(_require_api_key)):
    _load_model()

    if not req.rows:
        raise HTTPException(400, "Empty request")

    # Convert payload to Polars
    df = pl.from_dicts([r.dict(by_alias=True) for r in req.rows])

    # Figure out features to use
    if req.feature_order:
        feat_cols = req.feature_order
    elif _feature_cols:
        feat_cols = _feature_cols
    else:
        # fallback: use numeric intersection (avoid IDs)
        exclude = {"marketId", "selectionId"}
        feat_cols = [c for c, dt in zip(df.columns, df.dtypes) if c not in exclude and dt.is_numeric()]

    if not feat_cols:
        raise HTTPException(400, "No feature columns provided or discoverable")

    # Build X
    try:
        X = df.select(feat_cols).fill_null(strategy="mean").to_numpy(dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"Failed to build feature matrix: {e}")

    # Predict probabilities
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    else:
        # xgb Booster-like: outputs margin; apply logistic
        pred = _model.predict(X)
        p = 1.0 / (1.0 + np.exp(-pred))

    # Optional recommendations if 'ltp' is available
    side = stake = ev_back = ev_lay = [None] * len(p)
    if "ltp" in df.columns:
        s, st, eb, el = _recommend(
            df,
            p,
            kelly=req.kelly or 0.125,
            cap_ratio=req.cap_ratio or 0.02,
            bankroll=req.bankroll or 1000.0,
            market_cap=req.market_cap or 50.0,
            commission=req.commission or 0.05,
        )
        side = s.tolist()
        stake = [float(x) for x in st]
        ev_back = [float(x) for x in eb]
        ev_lay = [float(x) for x in el]

    preds = []
    for i, row in enumerate(req.rows):
        preds.append(Prediction(
            marketId=row.marketId,
            selectionId=row.selectionId,
            p_win=float(p[i]),
            side=side[i] if side else None,
            stake=stake[i] if stake else None,
            ev_back=ev_back[i] if ev_back else None,
            ev_lay=ev_lay[i] if ev_lay else None,
        ))

    return PredictResponse(predictions=preds, features_used=feat_cols)
