#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema loader for curated parquet datasets.

- Pulls JSON schema docs from environment-configured URLs (or CLI args).
- Caches them locally so runs don't block on network.
- Provides a small API to fetch field lists per dataset and validate presence.

ENV (optional):
  SCHEMA_ORDERBOOK_URL          default: http://192.168.40.200:8888/schema/parquet/orderbook
  SCHEMA_MARKETDEF_URL          default: http://192.168.40.200:8888/schema/parquet/market-definition
  SCHEMA_RESULTS_URL            default: http://192.168.40.200:8888/schema/parquet/results
  SCHEMA_CACHE_DIR              default: /opt/BetfairBotML/train_price_trend/cache/schemas
  SCHEMA_CACHE_TTL_SECS         default: 86400 (1 day)
"""

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import urllib.request

DEFAULTS = {
    "orderbook": os.environ.get("SCHEMA_ORDERBOOK_URL", "http://192.168.40.200:8888/schema/parquet/orderbook"),
    "marketdef": os.environ.get("SCHEMA_MARKETDEF_URL", "http://192.168.40.200:8888/schema/parquet/market-definition"),
    "results":   os.environ.get("SCHEMA_RESULTS_URL",   "http://192.168.40.200:8888/schema/parquet/results"),
}
CACHE_DIR = Path(os.environ.get("SCHEMA_CACHE_DIR", "/opt/BetfairBotML/train_price_trend/cache/schemas"))
CACHE_TTL = int(os.environ.get("SCHEMA_CACHE_TTL_SECS", "86400"))

# Fallback "must-have" columns if remote isn’t reachable
FALLBACK = {
    "orderbook": [
        "sport","marketId","selectionId","publishTimeMs","ltp","ltpTick",
        "tradedVolume","spreadTicks","imbalanceBest1","backTicks","backSizes","layTicks","laySizes"
    ],
    "marketdef": ["sport","marketId","marketStartMs","countryCode"],
    "results":   ["sport","marketId","selectionId","runnerStatus","winLabel"],
}

def _cache_path(kind: str) -> Path:
    return CACHE_DIR / f"{kind}.json"

def _stale(p: Path) -> bool:
    if not p.exists(): return True
    return (time.time() - p.stat().st_mtime) > CACHE_TTL

def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _ensure_cache(kind: str, url: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(kind)
    if _stale(cp):
        try:
            doc = _fetch(url)
            cp.write_text(json.dumps(doc, indent=2))
        except Exception:
            # leave old cache if any; otherwise write fallback
            if not cp.exists():
                cp.write_text(json.dumps({"fields": FALLBACK.get(kind, [])}, indent=2))
    return json.loads(cp.read_text())

def load_schema(kind: str, url: Optional[str] = None) -> List[str]:
    url_eff = url or DEFAULTS[kind]
    doc = _ensure_cache(kind, url_eff)
    # Accept both {"fields":[...]} and {"columns":[...]} shapes
    if isinstance(doc, dict):
        if "fields" in doc and isinstance(doc["fields"], list):
            return list(map(str, doc["fields"]))
        if "columns" in doc and isinstance(doc["columns"], list):
            return list(map(str, doc["columns"]))
    # If doc is a bare list
    if isinstance(doc, list):
        return list(map(str, doc))
    # Fallback
    return FALLBACK.get(kind, [])

def intersect_existing(requested: List[str], available: List[str]) -> List[str]:
    aset = set(available)
    return [c for c in requested if c in aset]

def cli():
    ap = argparse.ArgumentParser("schema_loader")
    ap.add_argument("--orderbook-url", default=DEFAULTS["orderbook"])
    ap.add_argument("--marketdef-url", default=DEFAULTS["marketdef"])
    ap.add_argument("--results-url",   default=DEFAULTS["results"])
    ap.add_argument("--print", choices=["orderbook","marketdef","results","all"], default="all")
    args = ap.parse_args()

    out = {}
    if args.print in ("orderbook","all"):
        out["orderbook"] = load_schema("orderbook", args.orderbook_url)
    if args.print in ("marketdef","all"):
        out["marketdef"] = load_schema("marketdef", args.marketdef_url)
    if args.print in ("results","all"):
        out["results"]   = load_schema("results",   args.results_url)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    cli()
