"""Encode each work's (title + description) with BAAI/bge-small-en-v1.5 -> 384d,
L2-normalized. Representative edition per work = the one with max ratings_count
(same rule as prepare_data). Output: retrieval/data/v1/item_text_emb.npy [n_items,384].

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=0 python encode_items_bge.py
"""
from __future__ import annotations

import time

import duckdb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config as C

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHARS = 1200          # title + description truncation (bge ctx is short anyway)


def main():
    C.ensure_dirs()
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    pq = lambda n: f"read_parquet('{C.PARQUET}/{n}.parquet')"
    idm = lambda n: f"read_parquet('{C.ID_MAPS}/{n}.parquet')"
    n_items = con.execute(f"SELECT max(iid)+1 FROM {idm('book_iid_map')}").fetchone()[0]

    df = con.execute(f"""
        SELECT iid, title, description FROM (
            SELECT m.iid AS iid, b.title, b.description,
                   row_number() OVER (PARTITION BY m.iid ORDER BY b.ratings_count DESC NULLS LAST) AS rk
            FROM {pq('books_core')} b JOIN {idm('book_iid_map')} m ON b.book_id = m.book_id
        ) WHERE rk = 1 ORDER BY iid
    """).fetchdf()
    assert len(df) == n_items and (df["iid"].values == np.arange(n_items)).all()

    def mk_text(t, d):
        t = (t or "").strip(); d = (d or "").strip()
        s = (t + ". " + d) if d else t
        return s[:MAX_CHARS] if s else "unknown book"
    texts = [mk_text(t, d) for t, d in zip(df["title"], df["description"])]
    n_desc = int((df["description"].fillna("").str.len() > 0).sum())
    log(f"{n_items:,} works; {n_desc:,} have description ({n_desc/n_items:.1%})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {MODEL} on {device}")
    enc = SentenceTransformer(MODEL, device=device)
    emb = enc.encode(texts, batch_size=256, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    log(f"encoded -> {emb.shape}")
    np.save(C.OUT_DIR / "item_text_emb.npy", emb)
    log(f"saved {C.OUT_DIR/'item_text_emb.npy'}")


if __name__ == "__main__":
    main()
