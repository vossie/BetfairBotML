# ml/service.py
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

# --- Config (env) ---
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/xgb_preoff.pkl")  # joblib dict {"model","features"} or xgb json
FEATURE_LIST_ENV = os.getenv("FEATURE_LIST", "")  # required if using xgb json
API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="Betfair ML Inference API", version="1.1.0")

# Lazy globals
_model = None
_feature_cols: Optional[List[str]] = None


def _load_model():
    global _model, _feature_cols
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
        model.load_model(MODEL_PATH)
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


# ---------- Schemas ----------

class FeatureRow(BaseModel):
    marketId: str
    selectionId: int

    # model features (send all numeric features your model expects)
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
        allow_population_by_field_name = True  # accept spreadTicks or spread_ticks


class PredictRequest(BaseModel):
    rows: List[FeatureRow]
    feature_order: Optional[List[str]] = None
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


# ---- Decision-oriented DTOs ----

class DecideRequest(BaseModel):
    rows: List[FeatureRow]
    # How to build features
    feature_order: Optional[List[str]] = None

    # Decision policy (defaults are sensible for exchange)
    kelly: float = 0.125             # Kelly fraction multiplier
    bankroll: float = 1000.0         # account size
    market_cap: float = 50.0         # max total stake per market
    commission: float = 0.05         # Betfair commission on wins
    min_stake: float = 1.0           # £1 floor
    min_edge: float = 0.0            # require EV >= min_edge * stake to PLACE
    top_n_per_market: int = 1        # choose at most N selections per market
    prefer_back: bool = True         # tie-breaker if EV back == EV lay


class Decision(BaseModel):
    marketId: str
    selectionId: int
    action: str             # PLACE_BACK | PLACE_LAY | HOLD
    stake: float            # rounded to 2dp
    p_win: float
    odds: Optional[float] = None
    ev: float               # EV of chosen action
    reason: Optional[str] = None     # e.g. "below min edge", "market cap exhausted"


class DecideResponse(BaseModel):
    decisions: List[Decision]
    features_used: List[str]


# ---------- Math helpers ----------

def _kelly_fraction(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    # Kelly for BACK @ decimal odds o: (o*p - 1)/(o - 1)
    f = (o * p - 1.0) / np.maximum(o - 1.0, 1e-9)
    return np.clip(f, 0.0, 1.0)


def _pick_features(df: pl.DataFrame, explicit: Optional[List[str]], learned: Optional[List[str]]) -> List[str]:
    if explicit:
        return explicit
    if learned:
        return learned
    exclude = {"marketId", "selectionId"}
    # numeric intersection fallback
    return [c for c, dt in zip(df.columns, df.dtypes) if c not in exclude and getattr(dt, "is_numeric", lambda: False)()]


def _predict_proba_matrix(X: np.ndarray) -> np.ndarray:
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    # Booster/raw margin -> logistic
    pred = _model.predict(X)
    return 1.0 / (1.0 + np.exp(-pred))


def _decide_core(
    df: pl.DataFrame,
    p: np.ndarray,
    *,
    kelly: float,
    bankroll: float,
    market_cap: float,
    commission: float,
    min_stake: float,
    min_edge: float,
    top_n_per_market: int,
    prefer_back: bool,
) -> List[Decision]:
    # sanity
    if "ltp" not in df.columns:
        raise HTTPException(400, "ltp required to decide")

    # odds, probs
    o = np.clip(df["ltp"].fill_null(1.01).to_numpy(), 1.01, 1000.0)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # Kelly stakes (fraction of bankroll), limited by market cap later
    f_back = _kelly_fraction(p, o) * kelly
    o_lay = o / (o - 1.0)
    p_lay = 1.0 - p
    f_lay = _kelly_fraction(p_lay, o_lay) * kelly
    stake_back = f_back * bankroll
    stake_lay = f_lay * bankroll

    # EV including commission (profit taxed only)
    ev_back = p * (o - 1.0) * (1.0 - commission) - (1.0 - p)  # per £ staked
    ev_lay = (1.0 - p) * (1.0 - commission) - p * (o - 1.0)   # per £ staked (rough lay EV)

    # Choose side with higher EV; break ties by prefer_back
    choose_back = ev_back > ev_lay if not prefer_back else ev_back >= ev_lay
    side = np.where(choose_back, "PLACE_BACK", "PLACE_LAY")
    ev_per_pound = np.where(choose_back, ev_back, ev_lay)
    raw_stake = np.where(choose_back, stake_back, stake_lay)

    # Apply per-market cap with £1 min rule and top-N
    # Build a working table
    tbl = pl.DataFrame({
        "marketId": df["marketId"],
        "selectionId": df["selectionId"],
        "side": side,
        "p_win": p,
        "odds": o,
        "ev_per_pound": ev_per_pound,
        "stake_suggested": raw_stake,
    })

    # Filter by min_edge (EV threshold per £ staked)
    tbl = tbl.with_columns([
        (pl.col("ev_per_pound")).alias("edge"),
    ]).with_columns([
        pl.when(pl.col("edge") >= min_edge).then(pl.col("stake_suggested")).otherwise(0.0).alias("stake_suggested")
    ])

    # pick top-N by edge within each market
    # (works on older Polars via arange-within-group)
    length_expr = getattr(pl, "len", None) or pl.count
    tbl = (
        tbl.sort(["marketId", "edge"], descending=[False, True])
        .with_columns(
            pl.arange(0, length_expr()).over("marketId").alias("_rank")
        )
        .filter(pl.col("_rank") < top_n_per_market)
        .drop(["_rank"])
    )

    # floor to min_stake for any positive stake
    tbl = tbl.with_columns([
        pl.when(pl.col("stake_suggested") <= 0.0).then(0.0)
        .when(pl.col("stake_suggested") < min_stake).then(min_stake)
        .otherwise(pl.col("stake_suggested"))
        .alias("stake_floor")
    ])

    # Re-normalize under market caps while keeping the floor
    def _renorm_market(g: pl.DataFrame) -> pl.DataFrame:
        # sort by edge desc, then by stake_suggested desc (priority)
        g = g.sort(by=["edge", "stake_suggested"], descending=[True, True])
        stake = g["stake_floor"].to_numpy().astype(float)
        pos = stake > 0
        n_pos = int(pos.sum())
        if n_pos == 0 or market_cap <= 0:
            stake[:] = 0.0
            return g.with_columns(pl.Series("stake_final", stake))

        if market_cap < n_pos * min_stake:
            # not enough room to give everyone min_stake → keep top k
            k = int(market_cap // min_stake)
            stake_new = np.zeros_like(stake)
            if k > 0:
                idx_pos = np.flatnonzero(pos)[:k]
                stake_new[idx_pos] = min_stake
            return g.with_columns(pl.Series("stake_final", stake_new))

        # everyone gets at least min_stake; distribute remaining budget proportionally
        base = n_pos * min_stake
        budget = market_cap - base
        # current "excess" over min_stake
        excess = np.where(pos, np.maximum(stake - min_stake, 0.0), 0.0)
        sum_excess = float(excess.sum())
        if sum_excess <= budget + 1e-12:
            return g.with_columns(pl.Series("stake_final", stake))
        scale = budget / sum_excess if sum_excess > 0 else 0.0
        stake_new = np.where(pos, min_stake + excess * scale, 0.0)
        return g.with_columns(pl.Series("stake_final", stake_new))

    # group-wise renorm (map_groups if available; else python loop)
    gb = tbl.group_by("marketId")
    if hasattr(gb, "map_groups"):
        tbl = gb.map_groups(_renorm_market)
    else:
        out_parts = []
        for mkt in tbl.select("marketId").unique().to_series().to_list():
            out_parts.append(_renorm_market(tbl.filter(pl.col("marketId") == mkt)))
        tbl = pl.concat(out_parts, how="vertical")

    # final EV (ev_per_pound * stake_final)
    tbl = tbl.with_columns((pl.col("ev_per_pound") * pl.col("stake_final")).alias("ev_total"))

    # Convert to decisions; HOLD if stake_final == 0
    decisions: List[Decision] = []
    for r in tbl.iter_rows(named=True):
        action = r["side"] if r["stake_final"] >= min_stake else "HOLD"
        reason = None if action != "HOLD" else ("below min edge or capped to zero")
        decisions.append(Decision(
            marketId=r["marketId"],
            selectionId=int(r["selectionId"]),
            action=action,
            stake=round(float(r["stake_final"]), 2),
            p_win=float(r["p_win"]),
            odds=float(r["odds"]),
            ev=round(float(r["ev_total"]), 4),
            reason=reason
        ))
    return decisions


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

    # Build X
    try:
        X = df.select(feat_cols).fill_null(strategy="mean").to_numpy(dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"Failed to build feature matrix: {e}")
    p = _predict_proba_matrix(X)

    # Optional simple recommendation if ltp present
    side = stake = ev_back = ev_lay = [None] * len(p)
    if "ltp" in df.columns:
        # reuse simple policy: Kelly+cap_ratio (legacy)
        kelly = req.kelly or 0.125
        cap_ratio = req.cap_ratio or 0.02
        bankroll = req.bankroll or 1000.0
        market_cap = req.market_cap or 50.0
        commission = req.commission or 0.05

        o = np.clip(df["ltp"].fill_null(1.01).to_numpy(), 1.01, 1000.0)
        p_clip = np.clip(p, 1e-6, 1 - 1e-6)
        f_back = _kelly_fraction(p_clip, o) * kelly
        o_lay = o / (o - 1.0)
        f_lay = _kelly_fraction(1.0 - p_clip, o_lay) * kelly
        max_stake = bankroll * cap_ratio
        stake_back = np.minimum(f_back * bankroll, max_stake)
        stake_lay = np.minimum(f_lay * bankroll, max_stake)
        ev_b = p_clip * (o - 1) * (1.0 - commission) - (1 - p_clip)
        ev_l = (1 - p_clip) * (1.0 - commission) - p_clip * (o - 1)
        choose_back = ev_b >= ev_l
        side_np = np.where(choose_back, "BACK", "LAY")
        stake_np = np.where(choose_back, stake_back, stake_lay)

        # per-market cap
        pdf = df.select(["marketId"]).to_pandas()
        pdf["stake"] = stake_np
        for mid, idx in pdf.groupby("marketId").groups.items():
            s = float(pdf.loc[idx, "stake"].sum())
            if s > market_cap:
                scale = market_cap / s
                pdf.loc[idx, "stake"] *= scale

        side = side_np.tolist()
        stake = [float(x) for x in pdf["stake"].to_numpy()]
        ev_back = [float(x) for x in ev_b]
        ev_lay = [float(x) for x in ev_l]

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
        X = df.select(feat_cols).fill_null(strategy="mean").to_numpy(dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"Failed to build feature matrix: {e}")

    p = _predict_proba_matrix(X)

    decisions = _decide_core(
        df, p,
        kelly=req.kelly,
        bankroll=req.bankroll,
        market_cap=req.market_cap,
        commission=req.commission,
        min_stake=req.min_stake,
        min_edge=req.min_edge,
        top_n_per_market=req.top_n_per_market,
        prefer_back=req.prefer_back,
    )
    return DecideResponse(decisions=decisions, features_used=feat_cols)
