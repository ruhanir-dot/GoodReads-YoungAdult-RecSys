"""v5 config — semantic ID (RQ-VAE) + dislike hard-negatives + language, on the two-tower.

Self-contained v5 layout (per user directive 2026-05-30):
  * BASE_DIR = data/v1   READ-ONLY shared base arrays from prepare_data (machine-independent,
                         CPU/duckdb-deterministic -> reused as-is, NOT regenerated on a40).
  * V5_DIR   = data/v5   ALL v5 artifacts WRITTEN here: a40 bge encodes, RQ-VAE codes,
                         language features, dislike pool, ckpts, evals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from env import P, STEP_DATA  # noqa: E402

# --------------------------------------------------------------------------- paths
# v5 base REBUILT with positives = rating>=4 ONLY (clean pos/neg separation for lever B;
# breaks the 0.486 comparability to v1-v4 which used is_read OR rating>=4 -- intentional).
# Built by prepare_data_v5.py; item-content arrays (bge/sem_codes/lang) live in V5_DIR and are
# positive-rule-INDEPENDENT so they're reused across both bases.
BASE_DIR: Path = STEP_DATA / "v5" / "base"   # v5-owned collaborative base (rating>=4)
V5_DIR:   Path = STEP_DATA / "v5"        # v5 content/codes/lang/dislike + outputs
SWEEP_DIR: Path = V5_DIR / "sweep"       # ckpts + eval_*.json + logs
COLD_DIR:  Path = V5_DIR / "cold"        # cold-item holdout artifacts + ckpts
# back-compat alias so copied modules (dataset.py / evaluate.py) that read C.OUT_DIR keep working:
OUT_DIR:  Path = BASE_DIR

# positive / split rule (NEW for v5 base)
POSITIVE_RULE = "rating>=4"
MIN_POS_FOR_SPLIT = 3

# upstream profiling v3 (THE version v5 uses — NOT v2) + ingest parquet for title/desc
PROFILING_V3: Path = P.PROFILING_DATA / "v3"
BOOK_PROFILES = PROFILING_V3 / "book_profiles.jsonl"   # {iid, language, tags}
USER_PROFILES = PROFILING_V3 / "user_profiles.jsonl"   # {uid, language{code:freq}, like[], dislike[]}
V3_VOCAB      = PROFILING_V3 / "vocab.json"
PARQUET = P.PARQUET_DIR
ID_MAPS = P.ID_MAPS_DIR

# --------------------------------------------------------------------------- data / split (mirror v4)
SEED = 42
HIST_LEN = 50
N_TAGS = 4000                 # base tag vocab (kept all-pad/inert in v5, like v4)
TAG_PAD = N_TAGS
LANG_TOPK = 20                # base-metadata lang categorical cap (item_cat[:,0], from prepare_data)
FORMAT_TOPK = 20
POP_TIERS = 8

# --------------------------------------------------------------------------- bge encodes (a40, -> V5_DIR)
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_DIM = 384
MAX_CHARS = 1200

# --------------------------------------------------------------------------- RQ-VAE (lever A: semantic ID)
RQ_L = 3                      # number of residual codebooks (codes per item)
RQ_K = 256                    # entries per codebook
RQ_DIM = 128                  # RQ-VAE latent dim
RQ_EPOCHS = 60
RQ_BATCH = 1024
RQ_LR = 1e-3
RQ_BETA = 0.25               # commitment loss weight
RQ_EMA = 0.99                # codebook EMA decay (None -> gradient codebooks)

# --------------------------------------------------------------------------- v5 levers (model)
@dataclass
class ModelCfg:
    d_out: int = 64
    d_id: int = 64            # atomic item-ID dim AND history-pool dim
    d_code: int = 64          # learned per-level semantic-code emb dim (lever A)
    d_tag: int = 32
    d_lang: int = 8
    d_format: int = 8
    d_poptier: int = 8
    d_ulang: int = 16         # user-language feature proj dim (lever C1)
    mlp_hidden: tuple = (256, 128)
    dropout: float = 0.1

# lever B: explicit dislike hard-negatives
DISLIKE_M = 8                 # max dislike negatives gathered per user/example
DISLIKE_LAMBDA = 1.0          # weight (logit bias log(lambda)) on dislike negatives

# lever C2: retrieval-time language-match penalty (applied at scoring)
LANG_PENALTY = 0.5           # subtract LANG_PENALTY * (1 - user_lang_weight[book_lang]) from score
LANG_MIN_W = 0.05            # ignore user langs with weight below this when building the match table

# --------------------------------------------------------------------------- training (mirror v4)
@dataclass
class TrainCfg:
    epochs: int = 16
    batch_size: int = 4096
    lr: float = 2e-3
    weight_decay: float = 1e-5
    temperature: float = 0.05
    max_pairs_per_epoch: int = 3_000_000
    grad_clip: float = 5.0
    num_workers: int = 8
    early_stop_patience: int = 4

# --------------------------------------------------------------------------- eval
EVAL_KS = (10, 20, 50, 100, 200)
EVAL_USER_BATCH = 4096
TIERS = {"cold": (3, 5), "warm": (6, 20), "hot": (21, 10**9)}
PER_LANG = ("en", "es", "de", "id", "it", "fr", "pt", "nl")  # per-language eval slices (rest -> 'other')

# --------------------------------------------------------------------------- cold-item holdout (lever A)
COLD_FRAC = 0.15
COLD_SEED = 1234

def ensure_dirs():
    for d in (V5_DIR, BASE_DIR, SWEEP_DIR, COLD_DIR):
        d.mkdir(parents=True, exist_ok=True)

MODEL = ModelCfg()
TRAIN = TrainCfg()
