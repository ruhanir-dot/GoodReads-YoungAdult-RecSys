"""Build the serving artifact: a FAISS flat inner-product index over the final
item embeddings, plus a human-readable demo of top-K recommendations for a few
sample test users (history titles -> recommended titles, with hit marker).

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=0 python build_index.py
Writes retrieval/data/v1/faiss_item.index and prints a demo.
"""
from __future__ import annotations

import faiss
import numpy as np
import torch

import config as C
import dataset as D
from model import TwoTower


def load_titles(n_items):
    import pandas as pd
    from env import P
    titles = [""] * n_items
    df = pd.read_parquet(P.PROFILING_DATA / "v2" / "item_inputs.parquet")  # iid -> title
    for iid, t in zip(df["iid"], df["title"]):
        if 0 <= int(iid) < n_items:
            titles[int(iid)] = str(t or "")
    return titles


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = D.load_meta()
    n_items = meta["n_items"]

    # ---- build + save FAISS index from final item vectors ----
    item_vecs = np.load(C.OUT_DIR / "item_vecs.npy").astype(np.float32)
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs)
    faiss.write_index(index, str(C.OUT_DIR / "faiss_item.index"))
    print(f"[ok] FAISS IndexFlatIP: {index.ntotal:,} items x {item_vecs.shape[1]}d "
          f"-> {C.OUT_DIR/'faiss_item.index'}")

    # ---- demo: top-10 for a few sample test users ----
    feats = D.load_features(meta)
    model = TwoTower(n_items, meta["n_users"], meta["n_tags"], meta["item_cat_cardinality"],
                     feats["item_num"].shape[1], feats["user_num"].shape[1], C.MODEL).to(device)
    model.attach_features(feats, device)
    model.load_state_dict(torch.load(C.CKPT_DIR / "best.pt", map_location=device)["state_dict"])
    model.eval()
    titles = load_titles(n_items)
    hist = np.load(C.OUT_DIR / "user_hist.npy")
    test = D.load_eval("test")

    rng = np.random.RandomState(7)
    sample = test[rng.choice(len(test), 4, replace=False)]
    print("\n================  RECOMMENDATION DEMO (4 sample test users)  ================")
    with torch.no_grad():
        for uid, tgt in sample:
            uvec = model.encode_user(torch.tensor([uid], device=device)).cpu().numpy().astype(np.float32)
            scores, ids = index.search(uvec, 10 + hist.shape[1])      # over-fetch to drop history
            hset = set(int(x) for x in hist[uid] if x < n_items)
            recs = [i for i in ids[0] if int(i) not in hset][:10]
            h = [titles[i] for i in list(hset)[:5]]
            print(f"\nuser {uid}  (held-out TEST item = [{tgt}] {titles[tgt][:55]!r})")
            print(f"  history (sample): {', '.join(t[:35] for t in h)}")
            for rank, i in enumerate(recs, 1):
                hit = "  <== HELD-OUT HIT" if i == tgt else ""
                print(f"   {rank:2d}. [{i}] {titles[i][:60]}{hit}")
    print("============================================================================")


if __name__ == "__main__":
    main()
