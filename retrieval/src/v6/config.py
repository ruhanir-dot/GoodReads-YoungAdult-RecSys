"""v6 config — leave-k-out split + soft/hard graded negatives + semantic-ID variations.

Self-contained v6 layout:
  * BASE_DIR    = data/v6/base   collaborative base (leave-k-out, rating>=4), built by prepare_data_v6.py
  * V6_DIR      = data/v6        v6 outputs (negatives, sem_codes_*, collab emb, sweep, cold)
  * CONTENT_DIR = data/v5        REUSE bge content/language (split-independent; not re-encoded)
v6 is NOT comparable to v1-v5 (new split + eval); all v6 models share THIS split + pos/neg rules.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from env import P, STEP_DATA  # noqa: E402

BASE_DIR: Path = STEP_DATA / "v6" / "base"
V6_DIR:   Path = STEP_DATA / "v6"
CONTENT_DIR: Path = STEP_DATA / "v5"          # reuse item_desc/tags/content_emb, book_lang, user_lang_w
SWEEP_DIR: Path = V6_DIR / "sweep"
COLD_DIR:  Path = V6_DIR / "cold"
OUT_DIR:  Path = BASE_DIR                      # back-compat alias for copied dataset.py/evaluate.py

PROFILING_V3: Path = P.PROFILING_DATA / "v3"
PARQUET = P.PARQUET_DIR
ID_MAPS = P.ID_MAPS_DIR

# --------------------------------------------------------------------------- split (leave-k-out, temporal)
SEED = 42
POSITIVE_RULE = "rating>=4"
TEST_FRAC = 0.10
VAL_FRAC = 0.10
MIN_POS_FOR_EVAL = 10        # users with >=10 positives get held-out test/val; else all-train

# --------------------------------------------------------------------------- features (mirror)
HIST_LEN = 50
N_TAGS = 4000
TAG_PAD = N_TAGS
LANG_TOPK = 20
FORMAT_TOPK = 20
POP_TIERS = 8
BGE_DIM = 384

# --------------------------------------------------------------------------- graded negatives (lever B)
NEG_M_HARD = 8              # max explicit hard (rating<=2) negatives per example
NEG_M_SOFT = 8             # max explicit soft (rating==3) negatives per example
LAMBDA_HARD = 2.0          # logit weight on hard negs (push harder): logit += log(LAMBDA_HARD)
LAMBDA_SOFT = 0.5          # logit weight on soft negs (push gently)

# --------------------------------------------------------------------------- RQ-VAE (semantic ID sources)
# content (M1): bge(title+desc+tags); collab (M2): co-occurrence SVD; big (M3): content K=1024/L=4
RQ_EPOCHS = 60; RQ_BATCH = 1024; RQ_LR = 1e-3; RQ_BETA = 0.25; RQ_EMA = 0.99
RQ_CONTENT = {"L": 3, "K": 256, "latent": 128, "src": "content"}
RQ_COLLAB  = {"L": 3, "K": 256, "latent": 128, "src": "collab"}
RQ_BIG     = {"L": 4, "K": 1024, "latent": 128, "src": "content"}
COLLAB_DIM = 64            # truncated-SVD dim for the collaborative item embedding

@dataclass
class ModelCfg:
    d_out: int = 64
    d_id: int = 64
    d_code: int = 64
    d_tag: int = 32
    d_lang: int = 8
    d_format: int = 8
    d_poptier: int = 8
    mlp_hidden: tuple = (256, 128)
    dropout: float = 0.1

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

EVAL_KS = (10, 20, 50, 100, 200)
EVAL_USER_BATCH = 2048
TIERS = {"cold": (10, 20), "warm": (21, 60), "hot": (61, 10**9)}   # user activity tiers (n_pos)
PER_LANG = ("en", "es", "de", "id", "it", "fr", "pt", "nl")

COLD_FRAC = 0.15
COLD_SEED = 1234

def ensure_dirs():
    for d in (V6_DIR, BASE_DIR, SWEEP_DIR, COLD_DIR):
        d.mkdir(parents=True, exist_ok=True)

MODEL = ModelCfg()
TRAIN = TrainCfg()
