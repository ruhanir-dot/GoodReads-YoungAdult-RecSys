"""Precompute v7 channel artifacts -> data/v7/. Pure CPU (+ faiss). One pass.

  mf_U.npy / mf_V.npy           SVD-MF user/item factors [.,MF_DIM]
  itemknn_nbr.npy / _sim.npy    item-item co-occurrence cosine top-N neighbours
  content_nbr.npy / _sim.npy    bge-content FAISS top-N neighbours
  rules_i2g.npz                 iid -> group ids (series+authors) CSR
  rules_g2i.npz                 group id -> iids CSR  (for the series/author channel)
  (popularity reuses data/v6/base/popularity.npy)

  python build_channels.py
"""
from __future__ import annotations
import time
import duckdb, numpy as np, faiss
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import config as C, dataset as D

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    C.ensure_dirs()
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    tp = np.load(C.V6_BASE / "train_pairs.npy"); u, it = tp[:, 0].astype(np.int64), tp[:, 1].astype(np.int64)
    M = sparse.csr_matrix((np.ones(len(u), np.float32), (u, it)), shape=(n_users, n_items))
    deg = np.asarray((M > 0).sum(0)).ravel().astype(np.float64)     # users per item
    log(f"M {M.shape} nnz={M.nnz:,}")

    # ---- MF (SVD) ----
    idf = np.log((n_users + 1) / (deg + 1)) + 1.0
    Mw = (M @ sparse.diags(idf.astype(np.float32))).tocsr()
    svd = TruncatedSVD(n_components=C.MF_DIM, random_state=C.SEED, n_iter=7); svd.fit(Mw)
    V = svd.components_.T.astype(np.float32)                        # [n_items, d]
    U = svd.transform(M).astype(np.float32)                        # [n_users, d]
    np.save(C.V7_DIR / "mf_V.npy", V); np.save(C.V7_DIR / "mf_U.npy", U)
    log(f"MF SVD: U{U.shape} V{V.shape} expl_var={svd.explained_variance_ratio_.sum():.3f}")

    # ---- itemKNN (co-occurrence cosine, chunked) ----
    N = C.KNN_N; CH = 2000; sq = np.sqrt(deg + 1e-9)
    nbr = np.zeros((n_items, N), np.int32); sim = np.zeros((n_items, N), np.float32)
    Mc = M.tocsc()
    for s in range(0, n_items, CH):
        e = min(s + CH, n_items)
        co = (Mc[:, s:e].T @ M).toarray().astype(np.float32)        # [chunk, n_items] co-counts
        co = co / (sq[s:e, None] * sq[None, :])                     # cosine
        for r in range(e - s):
            co[r, s + r] = 0.0                                      # zero self
        idx = np.argpartition(-co, N, axis=1)[:, :N]
        rows = np.arange(e - s)[:, None]
        sc = co[rows, idx]; order = np.argsort(-sc, axis=1)
        nbr[s:e] = idx[rows, order]; sim[s:e] = sc[rows, order]
        if s % 10000 == 0: log(f"  itemknn {e}/{n_items}")
    np.save(C.V7_DIR / "itemknn_nbr.npy", nbr); np.save(C.V7_DIR / "itemknn_sim.npy", sim)
    log("itemknn done")

    # ---- content KNN (FAISS on bge content) ----
    X = np.load(C.V5_DIR / "item_content_emb.npy").astype(np.float32); faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
    Ds, Is = index.search(X, N + 1)                                # incl self at col 0
    cnbr = np.zeros((n_items, N), np.int32); csim = np.zeros((n_items, N), np.float32)
    for i in range(n_items):
        keep = Is[i] != i
        cnbr[i] = Is[i][keep][:N]; csim[i] = Ds[i][keep][:N]
    np.save(C.V7_DIR / "content_nbr.npy", cnbr); np.save(C.V7_DIR / "content_sim.npy", csim)
    log("content knn done")

    # ---- series/author rules (group inverted index) ----
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    idm = f"read_parquet('{C.ID_MAPS}/book_iid_map.parquet')"
    ba = f"read_parquet('{C.PARQUET}/book_authors.parquet')"
    bs = f"read_parquet('{C.PARQUET}/book_series.parquet')"
    # iid -> author groups (namespace 0) and series groups (namespace 1); dense-remap group ids
    rows = con.execute(f"""
        SELECT m.iid AS iid, 'a'||a.author_id AS grp FROM {ba} a JOIN {idm} m ON a.book_id=m.book_id
        UNION
        SELECT m.iid AS iid, 's'||x.series_id AS grp FROM {bs} x JOIN {idm} m ON x.book_id=m.book_id
    """).fetchdf()
    gids, gcodes = np.unique(rows["grp"].to_numpy(), return_inverse=True)
    iid_arr = rows["iid"].to_numpy(np.int64); grp_arr = gcodes.astype(np.int64); n_grp = len(gids)
    # iid -> groups CSR
    order = np.argsort(iid_arr, kind="stable"); ia, ga = iid_arr[order], grp_arr[order]
    i2g_off = np.zeros(n_items + 1, np.int64); np.add.at(i2g_off, ia + 1, 1); i2g_off = np.cumsum(i2g_off)
    np.savez(C.V7_DIR / "rules_i2g.npz", off=i2g_off, flat=ga.astype(np.int64))
    # group -> iids CSR
    order2 = np.argsort(grp_arr, kind="stable"); gg, ii = grp_arr[order2], iid_arr[order2]
    g2i_off = np.zeros(n_grp + 1, np.int64); np.add.at(g2i_off, gg + 1, 1); g2i_off = np.cumsum(g2i_off)
    np.savez(C.V7_DIR / "rules_g2i.npz", off=g2i_off, flat=ii.astype(np.int64))
    log(f"rules: {n_grp:,} groups (series+authors); {len(iid_arr):,} (iid,group) pairs")
    log("build_channels DONE")


if __name__ == "__main__":
    main()
