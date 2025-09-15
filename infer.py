# ml/infer.py
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import xgboost as xgb


def load_booster(path: str) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(path)
    return bst


def predict_unified(bst_path: str, X: np.ndarray) -> np.ndarray:
    bst = load_booster(bst_path)
    d = xgb.DMatrix(X)
    p = bst.predict(d)
    if p.ndim == 2 and p.shape[1] > 1:
        p = p[:, 1]
    return p


def predict_gated(
    model_early_path: str,
    model_late_path: str,
    X: np.ndarray,
    tto_minutes: np.ndarray,
    gate_minutes: float = 45.0,
    blend_width: float = 30.0,
) -> np.ndarray:
    """
    Two-model gating: if tto<=gate use late model; else early model.
    Soft blend across a window of size blend_width centered at gate_minutes.
    """
    bst_e = load_booster(model_early_path)  # trained up to 180 minutes
    bst_l = load_booster(model_late_path)   # trained on <= 45 minutes
    de = xgb.DMatrix(X)
    pe = bst_e.predict(de)
    dl = xgb.DMatrix(X)
    pl = bst_l.predict(dl)
    if pe.ndim == 2 and pe.shape[1] > 1:
        pe = pe[:, 1]
    if pl.ndim == 2 and pl.shape[1] > 1:
        pl = pl[:, 1]

    # soft blend weight w in [0,1]: w=1 use late-model, w=0 use early-model
    half = max(1e-6, blend_width / 2.0)
    w = np.clip(1.0 - (tto_minutes - (gate_minutes - half)) / (2 * half), 0.0, 1.0)
    return w * pl + (1.0 - w) * pe
