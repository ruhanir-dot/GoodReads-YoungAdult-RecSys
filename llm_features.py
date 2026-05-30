"""LLM profile embedding features for the multi-stage ranker.

Two features are produced per (user, candidate) pool row:

  score_llm_user_item     cos(precomputed user profile emb, item profile emb).
                          Strong signal where it exists; 0 for users without
                          a precomputed profile (only ~10% of our k-core users).

  score_llm_history_item  cos(mean of user's read-item embs, candidate item emb).
                          Works for any user with >=1 read book that has an
                          item profile emb (~96% item coverage), so this gives
                          near-universal signal.

The bundle's embeddings live under
  recsys_data_v1/preprocessed_v1/embeddings/
and key off the bundle's own integer `uid`/`iid` space. `multi_stage.py`
builds its own dense IDs from string user_id/book_id, so this module bridges
through the bundle's id_maps to find each dense ID's embedding row (or -1
if missing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


class LLMFeatures:
    """Pre-loads LLM embeddings and builds dense_id -> embedding_row lookups
    aligned with multi_stage.py's local ID space."""

    def __init__(
        self,
        project_root: Path,
        user_id_to_dense: Dict[str, int],   # raw string user_id -> our dense_user_id
        book_id_to_dense: Dict[str, int],   # raw string book_id -> our dense_book_id
        n_users: int,
        n_books: int,
    ):
        emb_dir = project_root / 'recsys_data_v1' / 'preprocessed_v1' / 'embeddings'
        id_dir  = project_root / 'recsys_data_v1' / 'preprocessed_v1' / 'id_maps'

        # Embeddings are already unit-normalized, so cosine == dot product.
        self.user_emb = np.load(emb_dir / 'user_profile_emb.npy')   # (40314, 384)
        self.item_emb = np.load(emb_dir / 'item_profile_emb.npy')   # (32243, 384)
        self.dim = self.user_emb.shape[1]

        # Bundle's row indexes: row -> bundle uid / iid
        up_idx = pd.read_parquet(emb_dir / 'user_profile_emb_index.parquet')
        ip_idx = pd.read_parquet(emb_dir / 'item_profile_emb_index.parquet')
        bundle_uid_to_emb_row = dict(zip(up_idx['uid'].astype(int), up_idx['row'].astype(int)))
        bundle_iid_to_emb_row = dict(zip(ip_idx['iid'].astype(int), ip_idx['row'].astype(int)))

        # Bundle's id maps: raw string id -> bundle int id
        uid_map = pd.read_parquet(id_dir / 'uid_map.parquet')
        bid_map = pd.read_parquet(id_dir / 'book_iid_map.parquet')
        raw_uid_to_bundle = dict(zip(uid_map['user_id'], uid_map['uid'].astype(int)))
        raw_bid_to_bundle = dict(zip(bid_map['book_id'], bid_map['iid'].astype(int)))

        # Build our-dense-id -> embedding-row arrays (-1 means "no embedding")
        self.user_emb_row = np.full(n_users, -1, dtype=np.int32)
        for raw_uid, dense_uid in user_id_to_dense.items():
            bundle_uid = raw_uid_to_bundle.get(raw_uid)
            if bundle_uid is not None:
                row = bundle_uid_to_emb_row.get(bundle_uid)
                if row is not None:
                    self.user_emb_row[dense_uid] = row

        self.item_emb_row = np.full(n_books, -1, dtype=np.int32)
        for raw_bid, dense_bid in book_id_to_dense.items():
            bundle_iid = raw_bid_to_bundle.get(raw_bid)
            if bundle_iid is not None:
                row = bundle_iid_to_emb_row.get(bundle_iid)
                if row is not None:
                    self.item_emb_row[dense_bid] = row

        # Coverage stats (printed in the script's loading section)
        self.user_coverage = float((self.user_emb_row >= 0).mean())
        self.item_coverage = float((self.item_emb_row >= 0).mean())

        # Lazily-built per-user history-pooled embedding (set by precompute_history_embs)
        self._history_emb = None   # shape (n_users, dim); rows are zero for users with no embeddable history

    def precompute_history_embs(self, train_user_items: Dict[int, np.ndarray]) -> None:
        """Build a (n_users, dim) array where row[u] = unit-normalized mean of
        item_profile_embs over the items in user u's train history. Users with
        no embeddable history get a zero vector (which produces score=0)."""
        n_users = len(self.user_emb_row)
        hist = np.zeros((n_users, self.dim), dtype=np.float32)
        for user_id, history in train_user_items.items():
            if len(history) == 0:
                continue
            rows = self.item_emb_row[history]
            rows = rows[rows >= 0]
            if len(rows) == 0:
                continue
            mean = self.item_emb[rows].mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                hist[user_id] = mean / norm
        self._history_emb = hist

    def attach(self, pool_df: pd.DataFrame) -> pd.DataFrame:
        """Adds two columns to a copy of pool_df:
            score_llm_user_item, score_llm_history_item
        Requires precompute_history_embs() to have been called first."""
        if self._history_emb is None:
            raise RuntimeError("Call precompute_history_embs(train_user_items) first.")

        u_arr = pool_df['dense_user_id'].to_numpy()
        i_arr = pool_df['dense_book_id'].to_numpy()
        u_rows = self.user_emb_row[u_arr]
        i_rows = self.item_emb_row[i_arr]

        # Feature 1: precomputed user profile cosine
        # 0 for any row where either user or item lacks an embedding.
        score_user_item = np.zeros(len(pool_df), dtype=np.float32)
        valid = (u_rows >= 0) & (i_rows >= 0)
        if valid.any():
            uv = self.user_emb[u_rows[valid]]
            iv = self.item_emb[i_rows[valid]]
            score_user_item[valid] = np.einsum('ij,ij->i', uv, iv).astype(np.float32)

        # Feature 2: user-history-mean cosine
        # 0 only when item has no embedding OR user had no embeddable history.
        score_history_item = np.zeros(len(pool_df), dtype=np.float32)
        valid_item = i_rows >= 0
        if valid_item.any():
            uv_hist = self._history_emb[u_arr[valid_item]]    # (m, dim)
            iv = self.item_emb[i_rows[valid_item]]            # (m, dim)
            score_history_item[valid_item] = np.einsum('ij,ij->i', uv_hist, iv).astype(np.float32)

        out = pool_df.copy()
        out['score_llm_user_item'] = score_user_item
        out['score_llm_history_item'] = score_history_item
        return out