"""Shared evaluation helpers (imported by v4.py / v4_cold.py).

  build_train_mask : per-user consumed-item mask as CSR (offsets, flat), to exclude
                     already-seen items from scoring.
  user_npos        : per-user positive count (for cold/warm/hot activity tiers).
  tier_of          : map a positive-count to a tier name (config.TIERS).

(The standalone full-corpus eval + popularity baseline that used to live here was the v1
driver; each version now runs its own eval loop and just reuses these helpers.)
"""
from __future__ import annotations

import duckdb
import numpy as np

import config as C


def build_train_mask(n_users):
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    df = con.execute(f"SELECT uid, iid FROM read_parquet('{C.OUT_DIR}/split.parquet') "
                     f"WHERE split='train' ORDER BY uid").fetchdf()
    uid = df["uid"].to_numpy(np.int64); iid = df["iid"].to_numpy(np.int64)
    off = np.zeros(n_users + 1, np.int64)
    np.add.at(off, uid + 1, 1)
    off = np.cumsum(off)
    return off, iid       # flat already grouped by uid because ORDER BY uid


def user_npos(n_users):
    con = duckdb.connect()
    df = con.execute(f"SELECT uid, max(np) AS np FROM read_parquet('{C.OUT_DIR}/split.parquet') "
                     f"GROUP BY uid").fetchdf()
    arr = np.zeros(n_users, np.int64)
    arr[df["uid"].to_numpy()] = df["np"].to_numpy()
    return arr


def tier_of(npos):
    for name, (lo, hi) in C.TIERS.items():
        if lo <= npos <= hi:
            return name
    return "hot"
