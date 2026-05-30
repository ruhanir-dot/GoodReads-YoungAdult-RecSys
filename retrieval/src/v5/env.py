"""Bootstrap for this step's source: resolve the project root, put the canonical
path map on sys.path, and expose this step's own dirs.

Robust to nesting depth: works whether this file lives at <root>/retrieval/src/env.py
or one level deeper at <root>/retrieval/src/v4/env.py — STEP_DIR is resolved as the
ancestor that is a direct child of ROOT (i.e. <root>/retrieval), not by a fixed
parents[] index.

Usage from any sibling module in this version's src/:

    from env import P, ROOT, STEP_DIR, STEP_DATA
    parquet = P.PARQUET_DIR            # upstream (ingest) output
    out     = STEP_DATA / "v1"         # shared base arrays (prepare_data output)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("RECSYS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / ".recsys-root").exists():
            return cand
    return here.parents[2]


ROOT = _find_root()
sys.path.insert(0, str(ROOT))
import paths as P  # noqa: E402

# STEP_DIR = <root>/retrieval — the ancestor of this file that sits directly under ROOT,
# so STEP_DATA stays <root>/retrieval/data regardless of how deep this file is nested.
_here = Path(__file__).resolve()
STEP_DIR = next((p for p in _here.parents if p.parent == ROOT), _here.parents[1])
STEP_DATA = STEP_DIR / "data"
