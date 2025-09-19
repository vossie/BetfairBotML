# ml/pm_gate.py
import polars as pl


def apply_pm_cutoff(trades: pl.DataFrame, pm_cutoff: float | None = None) -> pl.DataFrame:
    """
    Filter trades by price-move model probability cutoff.

    trades must include 'p_pm_up' column (probability of upward move).
    """
    if pm_cutoff is None:
        return trades

    if "p_pm_up" not in trades.columns:
        print("[PM gate] WARNING: 'p_pm_up' not found in trades → skipping cutoff")
        return trades

    before = trades.height
    trades = trades.filter(pl.col("p_pm_up") >= pm_cutoff)
    after = trades.height

    print(f"[PM gate] Applied cutoff {pm_cutoff:.2f} → kept {after}/{before} trades")
    return trades
