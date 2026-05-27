"""
Split selectors every model filters through these so the eval user set
is the same everywhere. `split.parquet` is the single source of truth.

All DataFrames returned here use the in-memory `user_id`/`item_id` schema
(the on-disk `uid`/`iid` are renamed in `load_bundle`).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .data import Bundle


def _select(bundle: Bundle, name: str) -> pd.DataFrame:
    s = bundle.split
    return s.loc[s["split"] == name, ["user_id", "item_id"]].reset_index(drop=True)


def train_positives(bundle: Bundle) -> pd.DataFrame:
    """
    All (user_id, item_id)  with split == 'train'.
    """
    return _select(bundle, "train")


def val_pairs(bundle: Bundle) -> pd.DataFrame:
    """
    Held-out (user_id, item_id) pairs with split == 'val'.
    """
    return _select(bundle, "val")


def test_pairs(bundle: Bundle) -> pd.DataFrame:
    """
    Held-out (user_id, item_id) pairs with split == 'test'.
    """
    return _select(bundle, "test")


def eval_users(bundle: Bundle, split: str = "test") -> np.ndarray:
    """
    Sorted unique user_ids that actually have a held-out item in the given split.
    Ranking metrics computed over this set only 
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split!r}")
    s = bundle.split
    return np.sort(s.loc[s["split"] == split, "user_id"].unique())


def train_user_items(bundle: Bundle) -> Dict[int, np.ndarray]:
    """
    Build {user_id: np.ndarray(item_ids)} from split=='train'. 
    Used to mask already-seen items at recommendation time
    """
    tr = train_positives(bundle)
    # groupby is fast on already-sorted data; sort once.
    tr = tr.sort_values("user_id", kind="stable")
    return {
        user_id: grp["item_id"].to_numpy()
        for user_id, grp in tr.groupby("user_id", sort=False)
    }
