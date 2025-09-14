# ml/features.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any

import numpy as np
import polars as pl
import pyarrow.dataset as ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from . import dataio


# --------- helpers (version compatible) ---------

def _with_time(df: pl.DataFrame) -> pl.DataFrame:
    need = ["marketId", "selectionId", "publishTimeMs", "ltp",
            "spreadTicks", "imbalanceBest1", "tradedVolume"]
    for c in need:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return (
        df
        .with_columns([
            pl.col("marketId").cast(pl.Utf8),
            pl.col("selectionId").cast(pl.Int64),
            pl.col("publishTimeMs").cast(pl.Int64),
            pl.col("ltp").cast(pl.Float64),
            pl.col("spreadTicks").cast(pl.Int32).alias("spread_ticks"),
            pl.col("imbalanceBest1").cast(pl.Float64).alias("imb1"),
            pl.col("tradedVolume").cast(pl.Float64).alias("traded_vol"),
            # datetime from ms since epoch w/out from_epoch(unit=...)
            pl.col("publishTimeMs").cast(pl.Datetime("ms")).alias("ts"),
        ])
        .sort(["marketId", "selectionId", "publishTimeMs"])
        .with_columns(pl.col("ltp").forward_fill().alias("ltp_ff"))
    )


def _lag_join(df: pl.DataFrame, secs: int, value_col: str, out_name: str) -> pl.DataFrame:
    left = (
        df.select("marketId", "selectionId", "publishTimeMs", "ts", value_col)
          .with_columns((pl.col("ts") - pl.duration(seconds=secs)).alias("ts_lag"))
          .sort(["marketId", "selectionId", "ts_lag"])
    )
    right = (
        df.select("marketId", "selectionId", "ts", pl.col(value_col).alias(out_name))
          .sort(["marketId", "selectionId", "ts"])
    )
    j = left.join_asof(
        right,
        left_on="ts_lag",
        right_on="ts",
        by=["marketId", "selectionId"],
        strategy="backward",
    ).select("marketId", "selectionId", "publishTimeMs", out_name)

    return df.join(j, on=["marketId", "selectionId", "publishTimeMs"], how="left")


def _rolling_std(df: pl.DataFrame, window: str, out_name: str) -> pl.DataFrame:
    # robust rolling std via group_by_dynamic on ts
    g = (
        df
        .group_by_dynamic(index_column="ts", every=window, period=window, by=["marketId", "selectionId"], closed="right")
        .agg(pl.col("ltp_ff").std().alias(out_name))
    )
    return df.join(g, on=["marketId", "selectionId", "ts"], how="left")


# --------- memory-safe streaming builder ---------

def build_features_streaming(
    curated_root: str,
    sport: str,
    dates: List[str],
    preoff_minutes: int = 30,
    batch_markets: int = 100,
    downsample_secs: int | None = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Memory-safe feature builder:
      - identifies markets with results
      - narrows snapshots to last `preoff_minutes` before off
      - lazy scans parquet and collects per-market batches (streaming)
    Returns (features_df, total_raw_rows_scanned_approx).
    """
    tables = dataio.load_curated_multi(curated_root, sport, dates)
    df_defs = pl.from_arrow(tables["defs"])
    df_res  = pl.from_arrow(tables["results"])
    total_raw = int(pl.from_arrow(tables["snapshots"]).height) if tables["snapshots"] else 0

    if df_defs.is_empty() or df_res.is_empty():
        return pl.DataFrame(), 0

    # Markets with labels
    markets = (
        df_res.select("marketId")
        .unique()
        .to_series()
        .to_list()
    )
    if not markets:
        return pl.DataFrame(), 0

    # marketId -> startMs
    mkt_times = (
        df_defs
        .select(["marketId", "marketStartMs"])
        .unique(subset=["marketId"], keep="last")
        .drop_nulls(["marketId", "marketStartMs"])
        .with_columns([
            (pl.col("marketStartMs") - preoff_minutes * 60_000).alias("fromMs"),
            pl.col("marketStartMs").alias("toMs"),
        ])
    )
    if mkt_times.is_empty():
        return pl.DataFrame(), 0

    # lazy scan snapshots for all dates with column pruning
    snap_dirs = [dataio.ds_orderbook(curated_root, sport, d) for d in dates]

    fs, _ = dataio._fs_and_root(curated_root)

    file_list: List[str] = []
    for p in snap_dirs:
        file_list.extend(dataio._list_parquet_files(fs, p))
    if not file_list:
        return pl.DataFrame(), 0

    file_list = [dataio._to_fs_path(fs, f) for f in file_list]
    pa_ds = ds.dataset(file_list, format="parquet", filesystem=fs)
    lf_scan = pl.scan_pyarrow_dataset(pa_ds)

    need_cols = [c for c in ["marketId", "selectionId", "publishTimeMs", "ltp", "spreadTicks", "imbalanceBest1", "tradedVolume"]
                 if c in lf_scan.columns]
    lf_snap = lf_scan.select(need_cols)

    # optional downsample to reduce rows
    if downsample_secs and downsample_secs > 1:
        lf_snap = (
            lf_snap
            .with_columns((pl.col("publishTimeMs") // (downsample_secs * 1000)).alias("_bin"))
            .group_by(["marketId", "selectionId", "_bin"])
            .agg([
                pl.last("publishTimeMs").alias("publishTimeMs"),
                pl.last("ltp").alias("ltp"),
                pl.last("spreadTicks").alias("spreadTicks"),
                pl.last("imbalanceBest1").alias("imbalanceBest1"),
                pl.last("tradedVolume").alias("tradedVolume"),
            ])
            .drop("_bin")
        )

    out_batches: List[pl.DataFrame] = []

    # process markets in batches
    for i in range(0, len(markets), batch_markets):
        batch = markets[i:i+batch_markets]
        mkt_times_b = mkt_times.filter(pl.col("marketId").is_in(batch))

        lf_b = (
            lf_snap
            .join(mkt_times_b.lazy(), on="marketId", how="inner")
            .filter((pl.col("publishTimeMs") >= pl.col("fromMs")) & (pl.col("publishTimeMs") <= pl.col("toMs")))
            .filter(pl.col("marketId").is_in(batch))
        )

        # `collect(streaming=True)` can trigger a "not yet implemented" panic
        # in polars for some query plans (e.g. joins with filters).  Instead of
        # crashing the whole training run, try to collect using the streaming
        # engine and fall back to a normal in-memory collect if it fails.  This
        # keeps memory usage low when streaming works, but still allows the
        # function to succeed when polars lacks streaming support for a given
        # operation.
        try:
            df_b = lf_b.collect(streaming=True)
        except Exception:
            df_b = lf_b.collect()
        if df_b.is_empty():
            continue

        # labels for batch
        df_res_b = (
            df_res
            .filter(pl.col("marketId").is_in(batch))
            .select(["marketId", "selectionId", "winLabel", "runnerStatus"])
            .unique(subset=["marketId", "selectionId"], keep="first")
        )

        # feature engineering (on small batch)
        df_b = _with_time(df_b)
        df_b = _lag_join(df_b, 10, "ltp_ff", "ltp_10s_ago")
        df_b = _lag_join(df_b, 60, "ltp_ff", "ltp_60s_ago")
        df_b = df_b.with_columns([
            (pl.col("ltp_ff") - pl.col("ltp_10s_ago")).alias("mom_10s"),
            (pl.col("ltp_ff") - pl.col("ltp_60s_ago")).alias("mom_60s"),
        ])
        df_b = _rolling_std(df_b, "10s", "vol_10s")
        df_b = _rolling_std(df_b, "60s", "vol_60s")

        # overround / normalized probs / rank
        df_b = df_b.with_columns([
            pl.when(pl.col("ltp_ff") > 1.0).then(1.0 / pl.col("ltp_ff")).otherwise(None).alias("imp_prob")
        ]).with_columns([
            pl.sum("imp_prob").over(["marketId", "publishTimeMs"]).alias("overround")
        ]).with_columns([
            (pl.col("imp_prob") / pl.col("overround")).alias("norm_prob"),
            pl.col("ltp_ff").rank("ordinal").over(["marketId", "publishTimeMs"]).alias("rank_price"),
        ])

        # ensure `ltp` is unique
        if "ltp" in df_b.columns:
            df_b = df_b.drop("ltp")
        df_b = df_b.with_columns(pl.col("ltp_ff").alias("ltp"))
        df_b = df_b.drop([c for c in ["ltp_ff", "ltp_10s_ago", "ltp_60s_ago"] if c in df_b.columns])

        # attach labels; keep labeled rows
        df_b = df_b.join(df_res_b, on=["marketId", "selectionId"], how="left")
        df_b = df_b.filter(pl.col("winLabel").is_not_null())

        # per-market soft target (handles dead-heats / places if present)
        df_b = (
            df_b
            .with_columns(pl.sum("winLabel").over("marketId").alias("sum_win_in_mkt"))
            .with_columns(
                pl.when(pl.col("sum_win_in_mkt") > 0)
                .then(pl.col("winLabel") / pl.col("sum_win_in_mkt"))
                .otherwise(0.0)
                .alias("soft_target")
            )
            .drop("sum_win_in_mkt")
        )

        out_batches.append(df_b)

    if not out_batches:
        return pl.DataFrame(), 0

    return pl.concat(out_batches, how="diagonal_relaxed"), total_raw


# --------- train + recommend ---------

FEATURE_COLS = [
    "ltp", "mom_10s", "mom_60s", "vol_10s", "vol_60s",
    "spread_ticks", "imb1", "traded_vol", "overround", "norm_prob", "rank_price",
]

def _prep_xy(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # target: binary win (fallback from soft_target)
    y = df.get_column("winLabel").cast(pl.Int32).to_numpy()
    X = df.select([c for c in FEATURE_COLS if c in df.columns]).fill_null(strategy="mean").to_numpy()
    return X, y

def train_model(df_feat: pl.DataFrame) -> Dict[str, Any]:
    X, y = _prep_xy(df_feat)

    # downsample (optional) if very large
    if X.shape[0] > 2_000_000:
        idx = np.random.choice(X.shape[0], size=2_000_000, replace=False)
        X, y = X[idx], y[idx]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(Xtr, ytr)

    p = np.clip(clf.predict_proba(Xte)[:, 1], 1e-6, 1-1e-6)
    metrics = {
        "logloss": float(log_loss(yte, p)),
        "auc": float(roc_auc_score(yte, p)),
        "n_test": int(len(yte)),
    }
    return {"model": clf, "metrics": metrics}


def _kelly_fraction(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    """
    Kelly for BACK (decimal odds o): f* = (o*p - 1)/(o - 1), clipped to [0,1].
    For LAY, we invert below in recommend().
    """
    f = (o * p - 1.0) / (o - 1.0)
    return np.clip(f, 0.0, 1.0)


def recommend(
    df_feat: pl.DataFrame,
    artifacts: Dict[str, Any],
    bankroll: float = 1000.0,
    kelly: float = 0.125,          # fractional Kelly
    cap_ratio: float = 0.02,       # max % bank per bet
    market_cap: float = 50.0,      # max stake per market
) -> pl.DataFrame:
    clf = artifacts["model"]

    # Ensure required cols exist
    cols = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat.select(cols).fill_null(strategy="mean").to_numpy()
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1-1e-6)

    # odds from ltp (decimal)
    o = df_feat.get_column("ltp").fill_null(strategy="forward").cast(pl.Float64).to_numpy()
    o = np.clip(o, 1.01, 1000.0)

    # Back Kelly
    f_back = _kelly_fraction(p, o) * kelly

    # Lay Kelly (mirror via p_lay = 1-p, o_lay ~ o/(o-1) -> implied price of lay liability)
    # Simpler proxy: stake_back vs stake_lay; choose higher EV after caps.
    # Compute suggested stakes with caps:
    max_stake = bankroll * cap_ratio
    stake_back = np.minimum(f_back * bankroll, max_stake)

    # naive lay fraction: use back on (1-p) @ lay-odds = o/(o-1)
    o_lay = o / (o - 1.0)
    p_lay = 1.0 - p
    f_lay = _kelly_fraction(p_lay, o_lay) * kelly
    stake_lay = np.minimum(f_lay * bankroll, max_stake)

    # EV proxy (gross expectation per unit stake)
    ev_back = p * (o - 1) - (1 - p)
    ev_lay  = (1 - p_lay) * (1.0 / (o_lay - 1)) - p_lay  # rough; treat unit liability

    # choose side with higher EV, set stake
    choose_back = ev_back >= ev_lay
    side = np.where(choose_back, "BACK", "LAY")
    stake = np.where(choose_back, stake_back, stake_lay)

    # enforce per-market cap by grouping (approx in pandas)
    pdf = df_feat.select(["marketId", "selectionId", "ltp"]).to_pandas()
    pdf["side"] = side
    pdf["stake"] = stake
    pdf["ev_back"] = ev_back
    pdf["ev_lay"] = ev_lay

    # per-market cap: cap total stakes within a market
    for mid, idx in pdf.groupby("marketId").groups.items():
        s = pdf.loc[idx, "stake"].sum()
        if s > market_cap:
            scale = market_cap / s
            pdf.loc[idx, "stake"] *= scale

    # filter very small stakes
    pdf = pdf[pdf["stake"] >= 0.50].copy()

    return pl.from_pandas(pdf[["marketId", "selectionId", "ltp", "side", "stake", "ev_back", "ev_lay"]])
