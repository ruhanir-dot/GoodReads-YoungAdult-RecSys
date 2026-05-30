"""Load the shared base arrays (data/v1) into tensors.

Tag bags are stored on disk as (flat, offsets); we densify them once into padded
[N, Lmax] int matrices (pad = tag_pad) so a batch is a plain GPU index-gather.
(v4 / v4_cold build their own DataLoaders + logQ inline; this module only loads.)
"""
from __future__ import annotations

import json

import numpy as np
import torch

import config as C


def _pad_bags(npz_path, pad_id, n_rows):
    z = np.load(npz_path)
    flat, off = z["flat"], z["offsets"]                       # off[i] = start of row i
    ends = np.append(off[1:], len(flat))
    lens = ends - off
    Lmax = int(lens.max()) if len(lens) else 1
    out = np.full((n_rows, Lmax), pad_id, dtype=np.int64)
    for i in range(n_rows):
        s, e = off[i], ends[i]
        out[i, : e - s] = flat[s:e]
    return out


def load_meta():
    return json.load(open(C.OUT_DIR / "meta.json"))


def load_features(meta):
    n_items, n_users = meta["n_items"], meta["n_users"]
    feats = {
        "item_num": torch.from_numpy(np.load(C.OUT_DIR / "item_num.npy")),
        "item_cat": torch.from_numpy(np.load(C.OUT_DIR / "item_cat.npy")),
        "item_tags": torch.from_numpy(_pad_bags(C.OUT_DIR / "item_tags.npz", meta["tag_pad"], n_items)),
        "user_num": torch.from_numpy(np.load(C.OUT_DIR / "user_num.npy")),
        "user_liked": torch.from_numpy(_pad_bags(C.OUT_DIR / "user_liked.npz", meta["tag_pad"], n_users)),
        "user_disliked": torch.from_numpy(_pad_bags(C.OUT_DIR / "user_disliked.npz", meta["tag_pad"], n_users)),
        "user_hist": torch.from_numpy(np.load(C.OUT_DIR / "user_hist.npy").astype(np.int64)),
    }
    return feats


def load_eval(name):
    return np.load(C.OUT_DIR / f"{name}.npy")                # [n,2] (uid, target)
