# ml/dataio.py
from __future__ import annotations
import os
from typing import List, Tuple, Dict
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs


# ============================ S3 / FS HELPERS ============================

def _mk_s3_fs_from_env() -> pafs.S3FileSystem:
    """
    Build an S3FileSystem for MinIO/S3 using env vars:

      S3_ENDPOINT or AWS_ENDPOINT_URL or S3_ENDPOINT_URL   (e.g. http://minio:9000)
      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
      AWS_REGION or AWS_DEFAULT_REGION (default 'us-east-1')

    Notes:
      - We do NOT set path-style options here; PyArrow expects bucket/key paths when using S3FileSystem.
      - For HTTP MinIO, endpoint must start with http://; for HTTPS, https://
    """
    endpoint = (
        os.getenv("S3_ENDPOINT")
        or os.getenv("AWS_ENDPOINT_URL")
        or os.getenv("S3_ENDPOINT_URL")
    )
    if not endpoint:
        raise RuntimeError(
            "Set S3_ENDPOINT or AWS_ENDPOINT_URL or S3_ENDPOINT_URL (e.g. http://host:9000)"
        )

    access = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    use_ssl = endpoint.startswith("https://")

    return pafs.S3FileSystem(
        access_key=access,
        secret_key=secret,
        region=region,
        endpoint_override=endpoint,         # e.g. http://minio:9000
        scheme="https" if use_ssl else "http",
    )


def _fs_and_root(curated_root: str) -> Tuple[pafs.FileSystem, str]:
    """
    Return (filesystem, normalized_root).
    Accepts:
      - 's3://bucket[/prefix]'  (or 's3a://' which is normalized to 's3://')
      - local '/abs/path' or 'file:///abs/path'
    """
    if curated_root.startswith("s3a://"):
        curated_root = "s3://" + curated_root[len("s3a://") :]

    parsed = urlparse(curated_root)
    scheme = parsed.scheme

    if scheme == "s3":
        fs = _mk_s3_fs_from_env()
        # keep as s3://bucket/prefix for outward-facing paths; we normalize only when calling FS
        return fs, curated_root.rstrip("/")
    elif scheme in ("file", ""):
        fs = pafs.LocalFileSystem()
        root = (parsed.path if scheme == "file" else curated_root).rstrip("/")
        return fs, root
    else:
        raise RuntimeError(f"Unsupported curated_root scheme: {scheme}")


def _is_s3(fs: pafs.FileSystem) -> bool:
    return isinstance(fs, pafs.S3FileSystem)


def _to_fs_path(fs: pafs.FileSystem, path: str) -> str:
    """
    For S3FileSystem, convert 's3://bucket/key...' -> 'bucket/key...'
    For LocalFileSystem, return as-is.
    """
    if _is_s3(fs) and path.startswith("s3://"):
        return path[len("s3://"):]
    return path


# ============================ DIRECTORY / FILE ENUM ============================

def _list_children(fs: pafs.FileSystem, path: str, recursive: bool) -> List[pafs.FileInfo]:
    """
    Return FileInfo list under 'path' (dir) using FileSelector.
    Works for S3 & local.
    """
    p = _to_fs_path(fs, path)
    sel = pafs.FileSelector(p, allow_not_found=True, recursive=recursive)
    try:
        return fs.get_file_info(sel)
    except Exception:
        return []


def _list_parquet_files(fs: pafs.FileSystem, path: str) -> List[str]:
    """
    Return normalized paths (bucket/key for S3, absolute for local) to all '.parquet' files
    under 'path' recursively. Returns [] if none.
    """
    out: List[str] = []
    for info in _list_children(fs, path, recursive=True):
        # robust check for file:
        is_file = (
            getattr(info, "is_file", None) is True
            or info.type == pafs.FileType.File
        )
        if is_file and info.path.lower().endswith(".parquet"):
            # info.path is already normalized for the FS (bucket/key for S3)
            out.append(info.path)
    return out


def _has_any(fs: pafs.FileSystem, path: str) -> bool:
    """True if directory exists and has any entries (file or subdir)."""
    return len(_list_children(fs, path, recursive=False)) > 0


# ============================ PARTITIONED PATH HELPERS ============================

def _p(curated_root: str, dataset: str, sport: str, date: str) -> str:
    # <root>/<dataset>/sport=<sport>/date=<YYYY-MM-DD>
    return f"{curated_root.rstrip('/')}/{dataset}/sport={sport}/date={date}/"

def ds_orderbook(curated_root: str, sport: str, date: str) -> str:
    return _p(curated_root, "orderbook_snapshots_5s", sport, date)

def ds_market_defs(curated_root: str, sport: str, date: str) -> str:
    return _p(curated_root, "market_definitions", sport, date)

def ds_results(curated_root: str, sport: str, date: str) -> str:
    return _p(curated_root, "results", sport, date)


# ============================ READERS ============================

def read_table(path_or_paths, filesystem: pafs.FileSystem | None = None) -> pa.Table:
    """
    Read one or many *partition directories* (or files) of parquet into a pyarrow.Table.
    - Expands each directory into its explicit .parquet file list.
    - Skips empty directories silently (returns empty table if no files).
    """
    # Normalize input list
    paths = [path_or_paths] if isinstance(path_or_paths, str) else list(path_or_paths)
    if not paths:
        return pa.table({})

    # Infer FS if not supplied
    if filesystem is None:
        first = paths[0]
        if first.startswith("s3://") or first.startswith("s3a://"):
            # Just construct an S3 FS; we’ll normalize names later
            filesystem, _ = _fs_and_root("s3://dummy")  # endpoint pulled from env
        else:
            filesystem = pafs.LocalFileSystem()

    # Expand directories to explicit parquet file list
    file_list: List[str] = []
    for p in paths:
        # If user passes a file path directly, accept it; else expand dir
        if p.lower().endswith(".parquet"):
            # direct file path supplied
            file_list.append(_to_fs_path(filesystem, p))
        else:
            file_list.extend(_list_parquet_files(filesystem, p))

    if not file_list:
        return pa.table({})

    # dataset() accepts a list of files relative to the FS (bucket/key for S3)
    dataset = ds.dataset(file_list, format="parquet", filesystem=filesystem)
    return dataset.to_table()


# ============================ SINGLE-DAY (compat) ============================

def load_curated(curated_root: str, sport: str, date: str):
    """
    Return (snapshots_table, defs_table, results_table) for a single date.
    Empty tables are returned if a partition has no files.
    """
    fs, root = _fs_and_root(curated_root)
    snaps = read_table(ds_orderbook(root, sport, date), filesystem=fs)
    defs  = read_table(ds_market_defs(root, sport, date), filesystem=fs)
    res   = read_table(ds_results(root, sport, date), filesystem=fs)
    return snaps, defs, res


# ============================ MULTI-DAY ============================

def load_curated_multi(curated_root: str, sport: str, dates: List[str]) -> Dict[str, pa.Table]:
    """
    Return dict of pyarrow.Tables for multiple dates:
      {"snapshots": ..., "defs": ..., "results": ...}
    """
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


# ============================ DIAGNOSTICS ============================

def list_dates(curated_root: str, sport: str, dataset: str) -> List[str]:
    """
    List available dates under <root>/<dataset>/sport=<sport>/date=YYYY-MM-DD
    Works for S3 and local.
    """
    fs, root = _fs_and_root(curated_root)
    base = f"{root.rstrip('/')}/{dataset}/sport={sport}"
    if not _has_any(fs, base):
        return []
    out: List[str] = []
    for info in _list_children(fs, base, recursive=False):
        # guard: FileInfo may return full path including bucket/key or absolute path
        name = os.path.basename(info.path)
        if name.startswith("date="):
            # only include dates that have at least one parquet file
            full_path = info.path
            if _list_parquet_files(fs, full_path):
                out.append(name.split("=", 1)[1])
    return sorted(out)


__all__ = [
    "_mk_s3_fs_from_env", "_fs_and_root", "_is_s3", "_to_fs_path",
    "_list_parquet_files", "read_table",
    "ds_orderbook", "ds_market_defs", "ds_results",
    "load_curated", "load_curated_multi",
    "list_dates",
]


# ============================ CLI (optional) ============================

if __name__ == "__main__":
    # Usage:
    #   python -m ml.dataio s3://betfair-curated horse_racing orderbook_snapshots_5s
    import sys, json
    if len(sys.argv) != 4:
        print("Usage: python -m ml.dataio <curated_root> <sport> <dataset>")
        sys.exit(2)
    curated_root, sport, dataset = sys.argv[1:]
    dates = list_dates(curated_root, sport, dataset)
    print(json.dumps(dates, indent=2))
