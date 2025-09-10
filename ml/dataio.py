# ml/dataio.py
from __future__ import annotations
import os
from typing import List, Dict, Tuple
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from urllib.parse import urlparse

# ---------- FS helpers ----------

def _mk_s3_fs_from_env() -> pafs.S3FileSystem:
    """
    Build an S3FileSystem for MinIO using env vars:

      S3_ENDPOINT (or AWS_ENDPOINT_URL)  e.g. http://ziqni-minio.bronzegate.local:9000
      AWS_ACCESS_KEY_ID
      AWS_SECRET_ACCESS_KEY
      AWS_REGION (optional, default: us-east-1)
      S3_USE_SSL (optional "true"/"false")
    """
    endpoint = os.getenv("S3_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL")
    if not endpoint:
        raise RuntimeError("S3_ENDPOINT or AWS_ENDPOINT_URL must be set for MinIO access")

    access = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")

    region = os.getenv("AWS_REGION", "us-east-1")
    use_ssl = (os.getenv("S3_USE_SSL", "").lower() == "true") or endpoint.startswith("https://")

    # endpoint_override must not include scheme in some builds; pyarrow ≥12 handles full URL.
    return pafs.S3FileSystem(
        access_key=access,
        secret_key=secret,
        region=region,
        endpoint_override=endpoint,
        scheme="https" if use_ssl else "http"
    )

def _fs_and_root(curated_root: str) -> Tuple[pafs.FileSystem, str]:
    """
    Return (filesystem, normalized_root).
    Requires curated_root like: 's3://betfair-curated' or 'file:///path' or '/path'.
    """
    if curated_root.startswith("s3a://"):
        # Switch to s3:// for Arrow
        curated_root = "s3://" + curated_root[len("s3a://"):]
    parsed = urlparse(curated_root)

    if parsed.scheme == "s3":
        fs = _mk_s3_fs_from_env()
        # normalized root (bucket + optional prefix)
        root = curated_root.rstrip("/")
        return fs, root
    elif parsed.scheme in ("file", ""):
        fs = pafs.LocalFileSystem()
        # strip file:// if present
        root = (parsed.path if parsed.scheme == "file" else curated_root).rstrip("/")
        return fs, root
    else:
        raise RuntimeError(f"Unsupported curated_root scheme: {parsed.scheme}")

def _exists_dir(fs: pafs.FileSystem, path: str) -> bool:
    info = fs.get_file_info([path])[0]
    if info.type == pafs.FileType.Directory:
        return True
    if info.type == pafs.FileType.File:
        return True
    # Some S3 backends report NotFound for empty prefixes; try listing
    try:
        for _ in fs.get_file_info_selector(pafs.FileSelector(path, allow_not_found=True, recursive=False)):
            return True
    except Exception:
        return False
    return False

# ---------- Paths (partitioned layout) ----------

def _p(curated_root: str, dataset: str, sport: str, date: str) -> str:
    # Always build s3://… or local path, no trailing slash needed
    return f"{curated_root.rstrip('/')}/{dataset}/sport={sport}/date={date}"

def ds_orderbook(curated_root: str, sport: str, date: str) -> str:
    return _p(curated_root, "orderbook_snapshots_5s", sport, date)

def ds_market_defs(curated_root: str, sport: str, date: str) -> str:
    return _p(curated_root, "market_definitions", sport, date)

def ds_results(curated_root: str, sport: str, date: str) -> str:
    return _p(curated_root, "results", sport, date)

# ---------- Readers ----------

def read_table(path_or_paths, filesystem: pafs.FileSystem | None = None) -> pa.Table:
    """
    Read a parquet dataset (directory of .parquet files) into a single Arrow Table.
    """
    paths = [path_or_paths] if isinstance(path_or_paths, str) else list(path_or_paths)
    if not paths:
        return pa.table({})

    # Infer FS from first path if not supplied
    if filesystem is None:
        fs, _ = _fs_and_root(paths[0].split("/")[0] + "//") if "://" in paths[0] else (pafs.LocalFileSystem(), "")
        filesystem = fs

    # Check existence
    any_present = False
    for p in paths:
        if _exists_dir(filesystem, p):
            any_present = True
            break
    if not any_present:
        return pa.table({})

    dataset = ds.dataset(paths, format="parquet", filesystem=filesystem)
    return dataset.to_table()

# ---------- Single-day (backwards compatible) ----------

def load_curated(curated_root: str, sport: str, date: str):
    fs, root = _fs_and_root(curated_root)
    snaps = read_table(ds_orderbook(root, sport, date), filesystem=fs)
    defs  = read_table(ds_market_defs(root, sport, date), filesystem=fs)
    res   = read_table(ds_results(root, sport, date), filesystem=fs)
    return snaps, defs, res

# ---------- Multi-day ----------

def load_curated_multi(curated_root: str, sport: str, dates: List[str]) -> Dict[str, pa.Table]:
    fs, root = _fs_and_root(curated_root)
    if not dates:
        empty = pa.table({})
        return {"snapshots": empty, "defs": empty, "results": empty}

    snap_paths = [ds_orderbook(root, sport, d) for d in dates]
    def_paths  = [ds_market_defs(root, sport, d) for d in dates]
    res_paths  = [ds_results(root, sport, d) for d in dates]

    snaps = read_table(snap_paths, filesystem=fs)
    defs  = read_table(def_paths,  filesystem=fs)
    res   = read_table(res_paths,  filesystem=fs)
    return {"snapshots": snaps, "defs": defs, "results": res}

# ---------- Diagnostics ----------

def list_dates(curated_root: str, sport: str, dataset: str) -> List[str]:
    """
    List available dates under <root>/<dataset>/sport=<sport>/date=YYYY-MM-DD
    """
    fs, root = _fs_and_root(curated_root)
    base = f"{root.rstrip('/')}/{dataset}/sport={sport}"
    if not _exists_dir(fs, base):
        return []
    out: List[str] = []
    for info in fs.get_file_info_selector(pafs.FileSelector(base, recursive=False)):
        name = info.path.split("/")[-1]
        if name.startswith("date="):
            out.append(name.split("=", 1)[1])
    return sorted(out)
