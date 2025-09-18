import argparse
import numpy as np
import polars as pl

# --- helper to normalize executed rows schema ---
def _exec_rows_df(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "marketId": pl.Utf8,
        "selectionId": pl.Int64,
        "publishTimeMs": pl.Int64,
        "tto_minutes": pl.Float64,
        "side": pl.Utf8,
        "stake": pl.Float64,
        "odds": pl.Float64,
        "odds_exec": pl.Float64,
        "winLabel": pl.Int64,
        "p_hat": pl.Float64,
        "edge": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)

    norm = []
    for r in rows:
        norm.append({
            "marketId": str(r.get("marketId", "")),
            "selectionId": int(r.get("selectionId", 0)) if r.get("selectionId") is not None else 0,
            "publishTimeMs": int(r.get("publishTimeMs", 0)) if r.get("publishTimeMs") is not None else 0,
            "tto_minutes": float(r.get("tto_minutes")) if r.get("tto_minutes") is not None else np.nan,
            "side": str(r.get("side", "back")),
            "stake": float(r.get("stake", 0.0)) if r.get("stake") is not None else 0.0,
            "odds": float(r.get("odds")) if r.get("odds") is not None else np.nan,
            "odds_exec": float(r.get("odds_exec")) if r.get("odds_exec") is not None else np.nan,
            "winLabel": (int(r.get("winLabel")) if r.get("winLabel") is not None else None),
            "p_hat": float(r.get("p_hat")) if r.get("p_hat") is not None else np.nan,
            "edge": float(r.get("edge")) if r.get("edge") is not None else np.nan,
        })
    return pl.DataFrame(norm, schema=schema)

# --- placeholder main so module is importable ---
def main():
    print("sim2.py drop-in loaded with _exec_rows_df schema guard")

if __name__ == "__main__":
    main()
