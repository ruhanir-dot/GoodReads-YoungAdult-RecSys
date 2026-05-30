"""bge-encode the LLM item profiles (reader-view blurbs) -> item_profile_emb.npy [n_items,384],
aligned by iid, L2-normalized, same space as item_text_emb (bge title+desc). For items
without a profile, fall back to the title+desc embedding.

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> python encode_item_profiles.py
"""
from __future__ import annotations
import json, time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config as C
from env import P

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
MODEL = "BAAI/bge-small-en-v1.5"
ITEM_PROFILES = P.PROFILING_DATA / "v2" / "item_profiles.jsonl"


def main():
    meta = json.load(open(C.OUT_DIR / "meta.json"))
    n_items = meta["n_items"]
    prof = {}
    for l in open(ITEM_PROFILES):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "profile" in r and 0 <= r["iid"] < n_items:
            prof[r["iid"]] = r["profile"]
    log(f"{len(prof):,}/{n_items:,} items have a profile")
    iids = sorted(prof)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(MODEL, device=device)
    emb = enc.encode([prof[i] for i in iids], batch_size=512, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    # fall back to title+desc emb for items without a profile
    out = np.load(C.OUT_DIR / "item_text_emb.npy").astype(np.float32).copy()
    for k, i in enumerate(iids):
        out[i] = emb[k]
    np.save(C.OUT_DIR / "item_profile_emb.npy", out)
    log(f"saved item_profile_emb.npy [{n_items},{out.shape[1]}] (fallback to title+desc for {n_items-len(prof):,})")


if __name__ == "__main__":
    main()
