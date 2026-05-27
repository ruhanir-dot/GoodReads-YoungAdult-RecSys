"""
Bundle loader — single entry point for the preprocessed_v1 deliverable.

`load_bundle()` returns a `Bundle` dataclass with every table, embedding,
FAISS index, vocab, and row<->id lookup the modeling notebooks need.
Embeddings are memory-mapped so loading is cheap (<10s).

Naming convention (in-memory):
- Every DataFrame uses `user_id` (int32) and `item_id` (int32) as join keys.
- The on-disk parquets still use the original `uid` / `iid` column names —
  the loader renames them at load time so downstream code never sees the
  old names. (Special case: `uid_map.parquet` has both a string `user_id`
  and an int `uid`; we relabel the string column to `raw_user_id` to free
  up `user_id` for the dense int.)

LLM-derived artifacts (item_tags, user_tags, item_profile_emb,
user_profile_emb, tag_vocab) are **optional** — if any are missing the
loader emits a warning and the corresponding field is `None`. Per-artifact
path overrides let you point a single LLM artifact at an alternate file
(e.g. an experiment directory) without copying the rest of the bundle.
The `extras=` slot accepts arbitrary new artifacts.
"""

from __future__ import annotations
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd


def _default_bundle_path() -> Path:
    # src/ lives at the repo root; the bundle is under recsys_data_v1/preprocessed_v1/
    return Path(__file__).resolve().parent.parent / "recsys_data_v1" / "preprocessed_v1"


# On-disk column names → in-memory column names. Applied once, at load.
_UID_RENAME = {"uid": "user_id"}
_IID_RENAME = {"iid": "item_id"}
_BOTH_RENAME = {"uid": "user_id", "iid": "item_id"}


@dataclass
class Bundle:

    # defining dataclass
    path: Path

    # id maps (always required)
    user_id_map: pd.DataFrame
    book_itemid_map: pd.DataFrame

    # interactions / labels (always required; interactions_all may be empty
    # when load_interactions=False)
    interactions_all: pd.DataFrame
    split: pd.DataFrame
    popularity: pd.DataFrame

    # structured features (always required)
    book_features: pd.DataFrame
    user_features: pd.DataFrame

    # LLM-derived structured features (optional — None when missing)
    item_tags: Optional[pd.DataFrame]
    user_tags: Optional[pd.DataFrame]

    # bge-derived dense embeddings (always required)
    desc_emb: np.ndarray
    desc_emb_index: pd.DataFrame
    user_content_emb: np.ndarray
    user_content_index: pd.DataFrame

    # LLM-derived dense embeddings (optional pairs — both None when either missing)
    item_profile_emb: Optional[np.ndarray]
    item_profile_emb_index: Optional[pd.DataFrame]
    user_profile_emb: Optional[np.ndarray]
    user_profile_emb_index: Optional[pd.DataFrame]

    # faiss (always required unless load_faiss=False)
    faiss_desc: object

    # vocabularies
    tag_vocab: Optional[dict]   # LLM-derived (optional)
    cat_vocab: dict              # always required

    # row<->id lookups built once at load time
    itemid_to_row_desc: dict = field(default_factory=dict)
    userid_to_row_ucontentemb: dict = field(default_factory=dict)
    itemid_to_row_itemprofileemb: dict = field(default_factory=dict)
    userid_to_row_userprofileemb: dict = field(default_factory=dict)

    # convenience counts (recomputed from loaded data, not stale manifest)
    n_users_kcore: int = 0
    n_items: int = 0
    n_users_with_profile: int = 0
    n_items_with_profile: int = 0

    # generic extension slot — pass new artifacts here without changing the dataclass
    extras: dict = field(default_factory=dict)

    def has_user_content_emb(self, user_id: int) -> bool:
        return user_id in self.userid_to_row_ucontentemb

    def has_user_profile_emb(self, user_id: int) -> bool:
        return self.user_profile_emb is not None and user_id in self.userid_to_row_userprofileemb

    def has_item_profile_emb(self, item_id: int) -> bool:
        return self.item_profile_emb is not None and item_id in self.itemid_to_row_itemprofileemb


####### optional-artifact loaders #######


def _warn_missing(label: str, path: Path) -> None:
    warnings.warn(
        f"[load_bundle] optional artifact missing: {label} at {path} — proceeding with None",
        stacklevel=3,
    )



def _try_load_parquet(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        _warn_missing(label, path)
        return None
    return pd.read_parquet(path)


def _try_load_json(path: Path, label: str) -> Optional[dict]:
    if not path.exists():
        _warn_missing(label, path)
        return None
    return json.loads(path.read_text())


def _try_load_emb_pair( emb_path: Path, index_path: Path, label: str, ) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Embedding + index parquet load together. If either is missing, both -> None.
    """
    missing = [pth for pth in (emb_path, index_path) if not pth.exists()]
    if missing:
        _warn_missing(label, missing[0])
        return None, None
    return np.load(emb_path, mmap_mode="r"), pd.read_parquet(index_path)


####### main entry point #######


def load_bundle(
    path: Optional[str | Path] = None,
    *,
    load_interactions: bool = True, # skip if not needed
    load_faiss: bool = True,
    # per-artifact LLM overrides
    item_tags_path: Optional[str | Path] = None,
    user_tags_path: Optional[str | Path] = None,
    item_profile_emb_path: Optional[str | Path] = None,
    item_profile_emb_index_path: Optional[str | Path] = None,
    user_profile_emb_path: Optional[str | Path] = None,
    user_profile_emb_index_path: Optional[str | Path] = None,
    tag_vocab_path: Optional[str | Path] = None,
    # generic extension slot
    extras: Optional[dict[str, Any]] = None,
) -> Bundle:
    """Load the preprocessed_v1 bundle.

    Returns a `Bundle` whose DataFrames use `user_id`/`item_id` as join keys
    (the on-disk parquets still use `uid`/`iid` — they're renamed at load).

    Parameters
    ----------
    path : path to the bundle root. Defaults to the sibling of src/.
    load_interactions : skip the 34M-row interactions_all (~1.5 GB) when False.
    load_faiss : skip reading the FAISS index when False.

    LLM-artifact overrides (all optional): point a single LLM artifact at an
    alternate file without copying the rest of the bundle. If omitted, falls
    back to the path inside `path`. Missing files produce a warning, not an
    error — the corresponding Bundle field is set to None.

    extras : `{name: artifact}` dict attached to `Bundle.extras`. Use this to
    slot in new artifacts (e.g. a `profile_confidence` table or a
    `dealbreaker_emb`) without modifying this module.
    """
    p = Path(path) if path is not None else _default_bundle_path()
    if not p.exists():
        raise FileNotFoundError(f"Bundle path does not exist: {p}")

    manifest_path = p / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    ####### required artfacts #######
    # uid_map has a string user_id and an int uid, relabel the string
    # column before renaming uid -> user_id to avoid a collision.
    user_id_map = (
        pd.read_parquet(p / "id_maps/uid_map.parquet")
        .rename(columns={"user_id": "raw_user_id", "uid": "user_id"})
    )
    book_itemid_map = pd.read_parquet(p / "id_maps/book_iid_map.parquet").rename(columns=_IID_RENAME)

    split = pd.read_parquet(p / "interactions/split.parquet").rename(columns=_BOTH_RENAME)
    popularity = pd.read_parquet(p / "interactions/popularity.parquet").rename(columns=_IID_RENAME)
    if load_interactions:
        interactions_all = pd.read_parquet(p / "interactions/interactions_all.parquet").rename(columns=_BOTH_RENAME)
    else:
        interactions_all = pd.DataFrame(
            columns=["user_id", "item_id", "rating", "is_read", "is_positive", "ts"]
        )

    book_features = pd.read_parquet(p / "features/book_features.parquet").rename(columns=_IID_RENAME)
    user_features = pd.read_parquet(p / "features/user_features.parquet").rename(columns=_UID_RENAME)

    desc_emb = np.load(p / "embeddings/desc_emb.npy", mmap_mode="r")
    desc_emb_index = pd.read_parquet(p / "embeddings/desc_emb_index.parquet").rename(columns=_IID_RENAME)
    user_content_emb = np.load(p / "embeddings/user_content_emb.npy", mmap_mode="r")
    user_content_index = pd.read_parquet(p / "embeddings/user_content_index.parquet").rename(columns=_UID_RENAME)

    cat_vocab = json.loads((p / "vocab/categorical_vocabs.json").read_text())

    if load_faiss:
        import faiss
        faiss_desc = faiss.read_index(str(p / "embeddings/faiss_desc.index"))
    else:
        faiss_desc = None

    # LLM-derived optional artifacts 
    it_path = Path(item_tags_path) if item_tags_path else p / "features/item_tags.parquet"
    ut_path = Path(user_tags_path) if user_tags_path else p / "features/user_tags.parquet"
    item_tags = _try_load_parquet(it_path, "item_tags")
    user_tags = _try_load_parquet(ut_path, "user_tags")
    if item_tags is not None:
        item_tags = item_tags.rename(columns=_IID_RENAME)
    if user_tags is not None:
        user_tags = user_tags.rename(columns=_UID_RENAME)

    ipe_path = Path(item_profile_emb_path) if item_profile_emb_path \
        else p / "embeddings/item_profile_emb.npy"
    ipe_idx_path = Path(item_profile_emb_index_path) if item_profile_emb_index_path \
        else p / "embeddings/item_profile_emb_index.parquet"
    item_profile_emb, item_profile_emb_index = _try_load_emb_pair(
        ipe_path, ipe_idx_path, "item_profile_emb",
    )
    if item_profile_emb_index is not None:
        item_profile_emb_index = item_profile_emb_index.rename(columns=_IID_RENAME)

    upe_path = Path(user_profile_emb_path) if user_profile_emb_path \
        else p / "embeddings/user_profile_emb.npy"
    upe_idx_path = Path(user_profile_emb_index_path) if user_profile_emb_index_path \
        else p / "embeddings/user_profile_emb_index.parquet"
    user_profile_emb, user_profile_emb_index = _try_load_emb_pair(
        upe_path, upe_idx_path, "user_profile_emb",
    )
    if user_profile_emb_index is not None:
        user_profile_emb_index = user_profile_emb_index.rename(columns=_UID_RENAME)

    tv_path = Path(tag_vocab_path) if tag_vocab_path else p / "vocab/tag_vocab.json"
    tag_vocab = _try_load_json(tv_path, "tag_vocab")

    # row<->id lookups
    itemid_to_row_desc = dict(zip(
        desc_emb_index["item_id"].to_numpy(),
        desc_emb_index["row"].to_numpy(),
    ))
    userid_to_row_ucontentemb = dict(zip(
        user_content_index["user_id"].to_numpy(),
        user_content_index["row"].to_numpy(),
    ))
    itemid_to_row_itemprofileemb: dict = {}
    userid_to_row_userprofileemb: dict = {}
    if item_profile_emb_index is not None:
        itemid_to_row_itemprofileemb = dict(zip(
            item_profile_emb_index["item_id"].to_numpy(),
            item_profile_emb_index["row"].to_numpy(),
        ))
    if user_profile_emb_index is not None:
        userid_to_row_userprofileemb = dict(zip(
            user_profile_emb_index["user_id"].to_numpy(),
            user_profile_emb_index["row"].to_numpy(),
        ))

    # coverage counts
    n_items = len(book_features)
    n_users_kcore = manifest.get("coverage", {}).get("n_users_kcore", len(user_features))
    n_items_with_profile = len(item_profile_emb_index) if item_profile_emb_index is not None else 0
    n_users_with_profile = len(user_profile_emb_index) if user_profile_emb_index is not None else 0

    return Bundle(
        path=p,
        user_id_map=user_id_map,
        book_itemid_map=book_itemid_map,
        interactions_all=interactions_all,
        split=split,
        popularity=popularity,
        book_features=book_features,
        user_features=user_features,
        item_tags=item_tags,
        user_tags=user_tags,
        desc_emb=desc_emb,
        desc_emb_index=desc_emb_index,
        user_content_emb=user_content_emb,
        user_content_index=user_content_index,
        item_profile_emb=item_profile_emb,
        item_profile_emb_index=item_profile_emb_index,
        user_profile_emb=user_profile_emb,
        user_profile_emb_index=user_profile_emb_index,
        faiss_desc=faiss_desc,
        tag_vocab=tag_vocab,
        cat_vocab=cat_vocab,
        itemid_to_row_desc=itemid_to_row_desc,
        userid_to_row_ucontentemb=userid_to_row_ucontentemb,
        itemid_to_row_itemprofileemb=itemid_to_row_itemprofileemb,
        userid_to_row_userprofileemb=userid_to_row_userprofileemb,
        n_users_kcore=n_users_kcore,
        n_items=n_items,
        n_users_with_profile=n_users_with_profile,
        n_items_with_profile=n_items_with_profile,
        extras=dict(extras) if extras else {},
    )