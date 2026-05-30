"""Canonical path map for the Goodreads YA recsys pipeline — the single source of truth.

Every pipeline step (ingest / profiling / fusion / retrieval / ranking) resolves its
input/output locations from HERE, so the folder layout can change in exactly one place
and nothing downstream breaks.

Layout (each step owns a `src/` for code and a `data/` for its outputs):

    recsys/
    ├── .recsys-root            sentinel that marks the project root
    ├── paths.py                this file
    ├── data/                   shared raw NDJSON input (read-only)
    ├── eda/        {src,data}   notebook EDA
    ├── ingest/     {src,data}   raw NDJSON -> typed parquet + id_maps
    ├── profiling/  {src,data}   LLM item/user profiles + unified vocab
    ├── fusion/     {src,data}   model-ready bundle + packed model_inputs
    ├── retrieval/  {src,data}   recall models
    └── ranking/    {src,data}   ranking model

Root resolution order:
    1. $RECSYS_ROOT if set,
    2. else walk up from this file until a `.recsys-root` sentinel is found,
    3. else fall back to this file's directory.

A step that lives at  <root>/<step>/src/...  imports this module with:

    import os, sys
    from pathlib import Path
    def _root():
        env = os.environ.get("RECSYS_ROOT")
        if env:
            return Path(env).resolve()
        here = Path(__file__).resolve()
        for c in (here, *here.parents):
            if (c / ".recsys-root").exists():
                return c
        return here.parents[2]
    sys.path.insert(0, str(_root()))
    import paths as P
"""
from __future__ import annotations

import os
from pathlib import Path


def find_root() -> Path:
    env = os.environ.get("RECSYS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / ".recsys-root").exists():
            return cand
    return here.parent


ROOT = find_root()

# --------------------------------------------------------------------------- shared raw input
DATA_DIR         = ROOT / "data"
RAW_BOOKS        = DATA_DIR / "goodreads_books_young_adult.json"
RAW_REVIEWS      = DATA_DIR / "goodreads_reviews_young_adult.json"
RAW_INTERACTIONS = DATA_DIR / "goodreads_interactions_young_adult.json"

# --------------------------------------------------------------------------- step roots
EDA       = ROOT / "eda"
INGEST    = ROOT / "ingest"
PROFILING = ROOT / "profiling"
FUSION    = ROOT / "fusion"
RETRIEVAL = ROOT / "retrieval"
RANKING   = ROOT / "ranking"


def src(step: Path) -> Path:
    return step / "src"


def data(step: Path) -> Path:
    return step / "data"


# --------------------------------------------------------------------------- ingest outputs
INGEST_DATA    = INGEST / "data"
PARQUET_DIR    = INGEST_DATA / "parquet"      # 6 typed parquets
ID_MAPS_DIR    = INGEST_DATA / "id_maps"      # uid_map, book_iid_map (shared prereq)
EDA_REPORT_DIR = INGEST_DATA / "reports"      # CSV summaries from duckdb_preprocess
EDA_PLOT_DIR   = INGEST_DATA / "plots"        # PNG plots from duckdb_preprocess
RUNTIME_DIR    = INGEST_DATA / "runtime"      # DuckDB temp + persistent .duckdb
DB_PATH        = RUNTIME_DIR / "goodreads_eda.duckdb"
TMP_DIR        = RUNTIME_DIR / "tmp"

# --------------------------------------------------------------------------- profiling outputs
PROFILING_DATA = PROFILING / "data"
PROFILES_DIR   = PROFILING_DATA / "profiles"  # item.jsonl / user.jsonl (+ raw + shards)
PROF_INPUTS_DIR = PROFILING_DATA / "inputs"   # item_inputs.parquet / user_inputs.parquet
PROF_LOGS_DIR  = PROFILING_DATA / "logs"
VOCAB_JSON     = PROFILING_DATA / "vocab.json"
PROF_MANIFEST  = PROFILING_DATA / "manifest.json"

# --------------------------------------------------------------------------- fusion outputs
FUSION_DATA      = FUSION / "data"
BUNDLE_DIR       = FUSION_DATA / "bundle"
MODEL_INPUTS_DIR = FUSION_DATA / "bundle" / "model_inputs"

# --------------------------------------------------------------------------- modeling outputs
RETRIEVAL_DATA = RETRIEVAL / "data"
RANKING_DATA   = RANKING / "data"


if __name__ == "__main__":
    # Quick self-check: print the resolved map.
    for name in (
        "ROOT", "DATA_DIR", "PARQUET_DIR", "ID_MAPS_DIR", "RUNTIME_DIR",
        "PROFILES_DIR", "PROF_INPUTS_DIR", "VOCAB_JSON", "BUNDLE_DIR",
    ):
        val = globals()[name]
        exists = "ok " if Path(val).exists() else "MISSING"
        print(f"  [{exists}] {name:16s} = {val}")
