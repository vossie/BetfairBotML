# ml/dataio.py
from __future__ import annotations
import pyarrow as pa
import pyarrow.dataset as ds
from typing import List, Dict

# ---------- Paths ----------
def ds_orderbook(curated_root: str, sport: str, date: str) -> str:
    return f"{curated_root.rstrip('/')}/orderbook_snapshots_5s/sport={sport}/date={date}"

def ds_market_defs(curated_root: str, sport: str, date: str) -> str:
    return f"{curated_root.rstrip('/')}/market_definitions/sport={sport}/date={date}"

def ds_results(curated_root: str, sport: str, date: str) -> str:
    return f"{curated_root.rstrip('/')}/results/sport={sport}/date={date}"

# ---------- Readers ----------
def read_table(path_or_paths) -> pa.Table:
    if isinstance(path_or_paths, str):
        paths = [path_or_paths]
    else:
        paths = list(path_or_paths)

    for p in paths:
        try:
            _ = ds.dataset(p, format="parquet")
            break
        except Exception:
            continue
    else:
        return pa.table({})

    dataset = ds.dataset(paths, format="parquet")
    return dataset.to_table()

# ---------- Single-day ----------
def load_curated(curated_root: str, sport: str, date: str):
    snaps = read_table(ds_orderbook(curated_root, sport, date))
    defs  = read_table(ds_market_defs(curated_root, sport, date))
    res   = read_table(ds_results(curated_root, sport, date))
    return snaps, defs, res

# ---------- Multi-day ----------
def load_curated_multi(curated_root: str, sport: str, dates: List[str]) -> Dict[str, pa.Table]:
    if not dates:
        empty = pa.table({})
        return {"snapshots": empty, "defs": empty, "results": empty}

    snap_paths = [ds_orderbook(curated_root, sport, d) for d in dates]
    def_paths  = [ds_market_defs(curated_root, sport, d) for d in dates]
    res_paths  = [ds_results(curated_root, sport, d) for d in dates]

    snaps = read_table(snap_paths)
    defs  = read_table(def_paths)
    res   = read_table(res_paths)
    return {"snapshots": snaps, "defs": defs, "results": res}
