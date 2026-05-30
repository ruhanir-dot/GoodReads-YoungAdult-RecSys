"""Config for the v3 profiling pipeline (User + Book), leakage-free, bge-oriented.

User profile  : from TRAIN-split reviews only; recent positive+negative sampling;
                tokenizer-truncated concat; LLM -> {like, dislike}; language dict
                computed in CODE (langdetect freq), never by the LLM.
Book profile  : from title + description + language_code ONLY (no reviews); LLM ->
                {language (inferred from text if the field is missing), tags}.
Vocab         : built AFTER full profiling (top-3000 phrases; long tail dropped).
"""
from __future__ import annotations
import os, sys
from pathlib import Path


def _root() -> Path:
    env = os.environ.get("RECSYS_ROOT")
    if env: return Path(env).resolve()
    here = Path(__file__).resolve()
    for c in (here, *here.parents):
        if (c / ".recsys-root").exists(): return c
    return here.parents[3]

ROOT = _root(); sys.path.insert(0, str(ROOT))
import paths as P  # noqa: E402

# ---- paths ----
PARQUET   = P.PARQUET_DIR
ID_MAPS   = P.ID_MAPS_DIR
RAW_REVIEWS = P.RAW_REVIEWS
SPLIT     = P.RETRIEVAL / "data" / "v1" / "split.parquet"     # v2/v1 leakage-free split (reused)
OUT       = P.PROFILING_DATA / "v3"
USER_INPUTS = OUT / "user_inputs.parquet"
BOOK_INPUTS = OUT / "book_inputs.parquet"
USER_PROFILES = OUT / "user_profiles.jsonl"
BOOK_PROFILES = OUT / "book_profiles.jsonl"
VOCAB = OUT / "vocab.json"
OUT.mkdir(parents=True, exist_ok=True)

# ---- sampling (user) ----
U_MIN_REV = 3            # min train reviews-with-text to profile a user
K_POS = 20               # most-recent liked reviews (rating>=4)
K_NEG = 10               # most-recent disliked reviews (rating<=2)
REVIEW_CHAR_CAP = 800    # per-review soft char cap before concat
MIN_REVIEW_LEN = 20      # skip junk-short reviews
# ---- truncation ----
REVIEW_TOKEN_BUDGET = 2048   # tokenizer-truncate the CONCATENATED user reviews to this many tokens
TITLE_CAP = 120
DESC_CAP = 3000          # keep book description near-complete (p99 review<5k; desc usually shorter)

# ---- model / inference ----
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-4B")
LLM_DTYPE = "bfloat16"
BATCH_USER = int(os.environ.get("LLM_BATCH_USER", "48"))
BATCH_BOOK = int(os.environ.get("LLM_BATCH_BOOK", "64"))
MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW", "256"))
MAX_INPUT_TOKENS = int(os.environ.get("LLM_MAX_IN", "2700"))   # system + truncated reviews
NSHARDS = 2              # one process per idle GPU (0,1)

# ---- vocab ----
VOCAB_TOP = 3000
