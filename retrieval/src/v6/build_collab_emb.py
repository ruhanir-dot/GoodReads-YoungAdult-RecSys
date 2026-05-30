"""Collaborative item embedding for M2's semantic ID — a content-free, training-free latent
factorization of the interaction matrix (so M2's RQ-VAE codes capture CO-OCCURRENCE structure,
not bge content). Build sparse user x item (binary, train positives), apply TF-IDF-style row/col
weighting, truncated SVD -> item factors [n_items, COLLAB_DIM], L2-normalized. Pure CPU/sklearn.

Outputs -> data/v6/item_collab_emb.npy  [n_items, COLLAB_DIM] float32 (L2-normed)

  python build_collab_emb.py
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import config as C, dataset as D

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    C.ensure_dirs()
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    tp = np.load(C.BASE_DIR / "train_pairs.npy")           # (uid, iid) train positives
    u, it = tp[:, 0].astype(np.int64), tp[:, 1].astype(np.int64)
    M = sparse.csr_matrix((np.ones(len(u), np.float32), (u, it)), shape=(n_users, n_items))
    log(f"interaction matrix {M.shape} nnz={M.nnz:,}")

    # TF-IDF-ish col weighting (down-weight popular items) then SVD on item axis
    item_df = np.asarray((M > 0).sum(0)).ravel()           # users per item
    idf = np.log((n_users + 1) / (item_df + 1)) + 1.0
    Mw = (M @ sparse.diags(idf)).tocsr()                   # scale each item column by its idf
    svd = TruncatedSVD(n_components=C.COLLAB_DIM, random_state=42, n_iter=7)
    svd.fit(Mw)                                            # components_: [d, n_items]
    emb = svd.components_.T.astype(np.float32)             # [n_items, d] item factors
    emb = normalize(emb).astype(np.float32)
    np.save(C.V6_DIR / "item_collab_emb.npy", emb)
    log(f"saved item_collab_emb.npy {emb.shape}  explained_var_ratio_sum={svd.explained_variance_ratio_.sum():.3f}")
    # cold items (no train interactions) get a ~0 factor row -> RQ-VAE will still quantize them (content-free cold = weak)
    cold = item_df == 0
    log(f"  items with 0 train interactions: {int(cold.sum()):,} (their collab factor ~ untrained)")


if __name__ == "__main__":
    main()
