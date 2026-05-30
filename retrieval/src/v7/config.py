"""v7 config — multi-channel recall over the v6 leave-k-out base.

Reuses: data/v6/base (collaborative base + leave-k-out split), data/v6/sweep/ckpt_M0_N2_s42.pt
(two_tower channel), data/v5 bge content (content channel). Writes channel artifacts + results to data/v7.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from env import P, STEP_DATA  # noqa

V6_BASE: Path = STEP_DATA / "v6" / "base"
V5_DIR:  Path = STEP_DATA / "v5"
V7_DIR:  Path = STEP_DATA / "v7"
OUT_DIR: Path = V6_BASE                       # alias for copied dataset.py (load_meta etc.)
TWO_TOWER_CKPT: Path = STEP_DATA / "v6" / "sweep" / "ckpt_M0_N2_s42.pt"
PARQUET = P.PARQUET_DIR
ID_MAPS = P.ID_MAPS_DIR

# channel params
MF_DIM = 128            # SVD-MF latent dim
KNN_N = 200             # neighbors stored per item (itemknn / content)
CHANNEL_TOPK = 1000     # candidates each channel emits per user
RRF_C = 60              # RRF constant
EVAL_KS = (100, 200, 500, 1000)
EVAL_SAMPLE = 50000     # random eval users for the multi-channel eval (set-Recall is stable at this N)
EVAL_USER_BATCH = 2048
SEED = 42
PER_LANG = ("en", "es", "de", "id", "it", "fr", "pt", "nl")
HIST_LEN = 50

# two_tower model architecture (must match the v6 M0_N2 ckpt)
@dataclass
class ModelCfg:
    d_out: int = 64; d_id: int = 64; d_code: int = 64; d_tag: int = 32
    d_lang: int = 8; d_format: int = 8; d_poptier: int = 8
    mlp_hidden: tuple = (256, 128); dropout: float = 0.1

MODEL = ModelCfg()
N_TAGS = 4000; TAG_PAD = 4000

def ensure_dirs():
    V7_DIR.mkdir(parents=True, exist_ok=True)
