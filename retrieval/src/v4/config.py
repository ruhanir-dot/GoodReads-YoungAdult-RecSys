"""Central config for the two-tower retrieval pipeline (shared across versions).

All paths resolve from the canonical `paths.py` via env.py. All hyper-parameters
and feature dimensions live here so experiments are config-driven (no magic
numbers scattered across modules).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from env import P, STEP_DATA  # noqa: E402  (env.py puts repo root on sys.path)

# --------------------------------------------------------------------------- paths
# data/v1 is the SHARED base dir: prepare_data builds the aligned arrays here and ALL
# versions (v1/v3/v4) read them. (Name kept for backward-compat with existing artifacts.)
VERSION = "v1"
OUT_DIR: Path = STEP_DATA / VERSION
CKPT_DIR: Path = OUT_DIR / "ckpt"            # v1 ckpts (v3 -> OUT_DIR, v4 -> v4/sweep)

# --------------------------------------------------------------------------- cold-item holdout (v4_cold)
COLD_DIR: Path = STEP_DATA / "v4" / "cold"   # holdout artifacts + ckpts (prepare_coldstart.py / v4_cold.py)
COLD_FRAC = 0.15                             # fraction of items held fully out of training
COLD_SEED = 1234
D_CONTENT = 128                              # projection dim for bge text emb branch (prepare_coldstart)

# upstream inputs (typed parquet + id maps)
PARQUET = P.PARQUET_DIR
ID_MAPS = P.ID_MAPS_DIR

# --------------------------------------------------------------------------- data / split
SEED = 42
MIN_POS_FOR_SPLIT = 3        # users with >=3 positives get LOO test+val; else all-train
POSITIVE_RULE = "is_read OR rating>=4"
HIST_LEN = 50                # max history items pooled in the user tower
N_TAGS = 4000                # vocab size (verified)
TAG_PAD = N_TAGS             # sentinel id for empty tag bags -> embedding row N_TAGS
ITEM_PAD = None              # filled at runtime = n_items (pad row for history pooling)

# categorical cardinality caps (top-k by frequency, rest -> <unk>=0)
LANG_TOPK = 20
FORMAT_TOPK = 20
POP_TIERS = 8                # popularity quantile buckets

# --------------------------------------------------------------------------- model
@dataclass
class ModelCfg:
    d_out: int = 64           # final user/item embedding dim (retrieval space)
    d_id: int = 64            # item-ID embedding dim (also history-pool dim)
    d_tag: int = 32           # tag embedding dim (shared item<->user)
    d_lang: int = 8
    d_format: int = 8
    d_poptier: int = 8
    mlp_hidden: tuple = (256, 128)
    dropout: float = 0.1

# --------------------------------------------------------------------------- training
@dataclass
class TrainCfg:
    epochs: int = 16
    batch_size: int = 4096
    lr: float = 2e-3
    weight_decay: float = 1e-5
    temperature: float = 0.05
    max_pairs_per_epoch: int = 3_000_000   # subsample train pairs/epoch for speed (None = all)
    grad_clip: float = 5.0
    num_workers: int = 8
    early_stop_patience: int = 4            # stop if val Recall@100 doesn't improve

# --------------------------------------------------------------------------- eval
EVAL_KS = (10, 20, 50, 100, 200)
EVAL_MAX_USERS = None        # None = all eval users; set int to subsample for speed
EVAL_USER_BATCH = 4096       # users scored per GPU batch
TIERS = {"cold": (3, 5), "warm": (6, 20), "hot": (21, 10**9)}

# --------------------------------------------------------------------------- runtime
def ensure_dirs():
    for d in (OUT_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


MODEL = ModelCfg()
TRAIN = TrainCfg()
