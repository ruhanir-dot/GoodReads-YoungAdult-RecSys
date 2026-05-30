"""a40 bge-encode of ALL v5 content sources -> data/v5/*.npy  (one GPU session, bge loaded once).

WHY re-encode on a40 (not reuse h200's data/v1/item_text_emb.npy): the v3 LLM profile (tags/
like/dislike) is BRAND NEW and must be encoded here anyway; encoding title+desc on the SAME
a40 / sentence-transformers stack keeps every v5 content vector numerically consistent (one
machine, one ST version) and self-contained in data/v5. The CPU/duckdb base arrays in data/v1
(splits, pairs, history, numeric/categorical feats, popularity) are machine-independent and are
REUSED as-is, not regenerated.

Outputs (data/v5/, all [N,384] float32 L2-normalized, aligned by iid/uid):
  item_desc_emb.npy     bge(title + desc)                      -> v4-equivalent "desc" content branch
  item_tags_emb.npy     bge(", ".join(book.tags))             -> v3 tag content (fallback desc if no tags)
  item_content_emb.npy  bge(title + desc + tags)              -> RQ-VAE input (lever A)
  user_like_emb.npy     bge(", ".join(user.like))            -> taste vector (zeros if empty/no profile)
  user_dislike_emb.npy  bge(", ".join(user.dislike))         -> aversion vector (zeros if empty/no profile)

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> HF_HUB_OFFLINE=1 python encode_v3.py
"""
from __future__ import annotations
import json, time
import duckdb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config as C
import dataset as D

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def load_titles(n_items):
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    pq = lambda n: f"read_parquet('{C.PARQUET}/{n}.parquet')"
    idm = lambda n: f"read_parquet('{C.ID_MAPS}/{n}.parquet')"
    df = con.execute(f"""
        SELECT iid, title, description FROM (
            SELECT m.iid AS iid, b.title, b.description,
                   row_number() OVER (PARTITION BY m.iid ORDER BY b.ratings_count DESC NULLS LAST) AS rk
            FROM {pq('books_core')} b JOIN {idm('book_iid_map')} m ON b.book_id = m.book_id
        ) WHERE rk = 1 ORDER BY iid
    """).fetchdf()
    assert len(df) == n_items and (df["iid"].values == np.arange(n_items)).all()
    return [(t or "").strip() for t in df["title"]], [(d or "").strip() for d in df["description"]]


def main():
    C.ensure_dirs()
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    titles, descs = load_titles(n_items)
    log(f"loaded {n_items:,} titles/descs from parquet")

    # book tags (v3) aligned by iid
    tags = [None] * n_items
    for l in open(C.BOOK_PROFILES):
        try: r = json.loads(l)
        except Exception: continue
        i = r.get("iid")
        if isinstance(i, int) and 0 <= i < n_items and r.get("tags"):
            tags[i] = ", ".join(r["tags"])
    n_tagged = sum(t is not None for t in tags)
    log(f"book tags: {n_tagged:,}/{n_items:,} have v3 tags")

    # user like/dislike (v3) aligned by uid
    likes = [None] * n_users; dislikes = [None] * n_users
    for l in open(C.USER_PROFILES):
        try: r = json.loads(l)
        except Exception: continue
        u = r.get("uid")
        if not (isinstance(u, int) and 0 <= u < n_users): continue
        if r.get("like"):    likes[u] = ", ".join(r["like"])
        if r.get("dislike"): dislikes[u] = ", ".join(r["dislike"])
    log(f"user like: {sum(x is not None for x in likes):,}/{n_users:,} ; "
        f"dislike: {sum(x is not None for x in dislikes):,}/{n_users:,}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(C.BGE_MODEL, device=dev)
    log(f"loaded {C.BGE_MODEL} on {dev}")

    def encode(texts, bs=512):
        return enc.encode(texts, batch_size=bs, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def clip(s): return s[:C.MAX_CHARS] if s else ""

    # 1) item desc = title + ". " + desc
    desc_text = [clip((t + ". " + d) if d else t) or "unknown book" for t, d in zip(titles, descs)]
    item_desc = encode(desc_text)
    np.save(C.V5_DIR / "item_desc_emb.npy", item_desc); log("saved item_desc_emb.npy")

    # 2) item tags = bge(tags); fallback to desc emb where no tags
    tag_text = [tags[i] if tags[i] else "" for i in range(n_items)]
    have = np.array([tags[i] is not None for i in range(n_items)])
    item_tags = item_desc.copy()
    if have.any():
        enc_tags = encode([tag_text[i] for i in np.where(have)[0]])
        item_tags[have] = enc_tags
    np.save(C.V5_DIR / "item_tags_emb.npy", item_tags); log("saved item_tags_emb.npy (desc fallback for untagged)")

    # 3) item content for RQ-VAE = title + desc + tags
    content_text = [clip(desc_text[i] + ((" Themes: " + tags[i]) if tags[i] else "")) for i in range(n_items)]
    item_content = encode(content_text)
    np.save(C.V5_DIR / "item_content_emb.npy", item_content); log("saved item_content_emb.npy (RQ-VAE input)")

    # 4) user like / 5) user dislike (zeros where empty)
    for name, arr in (("user_like_emb", likes), ("user_dislike_emb", dislikes)):
        out = np.zeros((n_users, C.BGE_DIM), np.float32)
        idx = [u for u in range(n_users) if arr[u] is not None]
        if idx:
            e = encode([clip(arr[u]) for u in idx])
            out[np.array(idx)] = e
        np.save(C.V5_DIR / f"{name}.npy", out)
        log(f"saved {name}.npy (nonzero rows: {len(idx):,})")

    log("encode_v3 DONE")


if __name__ == "__main__":
    main()
