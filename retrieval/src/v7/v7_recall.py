"""v7 multi-channel recall + RRF merge + candidate-pool evaluation (v6 leave-k-out, set-based).

6 channels (two_tower, mf, itemknn, content, series_author, popularity) each emit top-CHANNEL_TOPK
per eval user (masking train+val); RRF merges them. Reports per-channel + merged set-Recall@K,
per-channel unique contribution / overlap, and per-(user/item)-tier recall for routing.

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> python v7_recall.py
"""
from __future__ import annotations
import json, time
import duckdb, numpy as np, torch
from scipy import sparse
import config as C, dataset as D
from model import TwoTowerV6

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
NEG = float("-inf")
CHANNELS = ["two_tower", "mf", "itemknn", "content", "series_author", "popularity"]
SPARSE_CH = {"itemknn", "content", "series_author", "popularity"}   # score<=0 means "no signal"


def csr_split(splits, n_users):
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    q = ",".join(f"'{s}'" for s in splits)
    df = con.execute(f"SELECT uid,iid FROM read_parquet('{C.V6_BASE}/split.parquet') WHERE split IN ({q}) ORDER BY uid").fetchdf()
    uid = df["uid"].to_numpy(np.int64); iid = df["iid"].to_numpy(np.int64)
    off = np.zeros(n_users + 1, np.int64); np.add.at(off, uid + 1, 1); off = np.cumsum(off)
    return off, iid


def sim_matrix(nbr, sim, n_items):
    rows = np.repeat(np.arange(n_items), nbr.shape[1])
    return sparse.csr_matrix((sim.ravel(), (rows, nbr.ravel().astype(np.int64))), shape=(n_items, n_items))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    K = C.CHANNEL_TOPK; Ks = C.EVAL_KS; maxK = max(Ks)

    mask_off, mask_flat = csr_split(["train", "val"], n_users)            # mask train+val at scoring
    test_off, test_flat = csr_split(["test"], n_users)
    eval_all = np.where(np.diff(test_off) > 0)[0]
    n_pos = (mask_off[1:] - mask_off[:-1]) + (test_off[1:] - test_off[:-1])
    if len(eval_all) > C.EVAL_SAMPLE:
        eval_uids = np.sort(np.random.RandomState(C.SEED).choice(eval_all, C.EVAL_SAMPLE, replace=False))
    else:
        eval_uids = eval_all
    log(f"eval users: {len(eval_uids):,} (of {len(eval_all):,} with test items)")

    user_hist = np.load(C.V6_BASE / "user_hist.npy").astype(np.int64)
    pop = np.load(C.V6_BASE / "popularity.npy").astype(np.float32)
    q50, q90 = np.quantile(pop, 0.5), np.quantile(pop, 0.9)
    itier = np.where(pop > q90, 0, np.where(pop <= q50, 2, 1))            # 0 head 1 mid 2 tail

    # ---- channel artifacts ----
    mf_U = np.load(C.V7_DIR / "mf_U.npy"); mf_V = torch.from_numpy(np.load(C.V7_DIR / "mf_V.npy")).to(dev)
    S = sim_matrix(np.load(C.V7_DIR / "itemknn_nbr.npy"), np.load(C.V7_DIR / "itemknn_sim.npy"), n_items)
    Sc = sim_matrix(np.load(C.V7_DIR / "content_nbr.npy"), np.load(C.V7_DIR / "content_sim.npy"), n_items)
    i2g = np.load(C.V7_DIR / "rules_i2g.npz"); rows = np.repeat(np.arange(n_items), np.diff(i2g["off"]))
    n_grp = int(i2g["flat"].max()) + 1
    IG = sparse.csr_matrix((np.ones(len(i2g["flat"]), np.float32), (rows, i2g["flat"])), shape=(n_items, n_grp))
    IGt = IG.T.tocsr()
    pop_t = torch.from_numpy(pop).to(dev)

    # ---- two_tower ----
    feats = {"item_num": torch.from_numpy(np.load(C.V6_BASE / "item_num.npy")),
             "item_cat": torch.from_numpy(np.load(C.V6_BASE / "item_cat.npy")),
             "item_tags": torch.full((n_items, 1), meta["tag_pad"], dtype=torch.int64),
             "user_num": torch.from_numpy(np.load(C.V6_BASE / "user_num.npy")),
             "user_hist": torch.from_numpy(user_hist),
             "item_text": torch.from_numpy(np.load(C.V5_DIR / "item_desc_emb.npy").astype(np.float32))}
    tt = TwoTowerV6(n_items, n_users, C.N_TAGS, meta["item_cat_cardinality"],
                    feats["item_num"].shape[1], feats["user_num"].shape[1], C.MODEL, id_mode="atomic",
                    use_text=True, d_content=128, n_item_text=feats["item_text"].shape[1]).to(dev)
    tt.load_state_dict(torch.load(C.TWO_TOWER_CKPT, map_location=dev)["state_dict"]); tt.eval()
    tt.attach_features(feats, dev)
    item_vecs = tt.encode_all_items(dev)
    log("two_tower ckpt loaded + items encoded")

    # ---- accumulators ----
    per_ch = {c: {k: 0.0 for k in Ks} for c in CHANNELS}; merged = {k: 0.0 for k in Ks}
    ug = {t: {"n": 0, **{k: 0.0 for k in Ks}} for t in ("u_cold", "u_warm", "u_hot")}
    mg_utier = {t: {"n": 0, **{k: 0.0 for k in Ks}} for t in ("u_cold", "u_warm", "u_hot")}
    ig_ch = {c: {t: {"n": 0, "hit": 0} for t in ("head", "mid", "tail")} for c in CHANNELS + ["merged"]}
    contrib = {c: {"surfaced": 0, "unique": 0} for c in CHANNELS}        # @500 in merged pool
    n_eval = 0

    @torch.no_grad()
    def topk_channel(name, ub, hist_csr):
        B = len(ub)
        if name == "two_tower":
            sc = tt.encode_user(torch.from_numpy(ub).long().to(dev)) @ item_vecs.t()
        elif name == "mf":
            sc = torch.from_numpy(mf_U[ub]).to(dev) @ mf_V.t()
        elif name == "popularity":
            sc = pop_t.unsqueeze(0).expand(B, -1).clone()
        else:
            mat = {"itemknn": S, "content": Sc}.get(name)
            dense = ((hist_csr @ mat) if mat is not None else ((hist_csr @ IG) @ IGt)).toarray()
            sc = torch.from_numpy(dense).to(dev)
        cols = np.concatenate([mask_flat[mask_off[u]:mask_off[u+1]] for u in ub])
        rows = np.repeat(np.arange(B), mask_off[ub+1]-mask_off[ub])
        if len(cols): sc[torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)] = NEG
        val, idx = torch.topk(sc, K, 1)
        idx = idx.cpu().numpy(); val = val.cpu().numpy()
        if name in SPARSE_CH: idx = np.where(val > 0, idx, -1)           # drop no-signal candidates
        return idx

    rl = np.full(n_items + 1, K, np.int32)                              # rank lookup (index n_items = invalid sink)
    for s in range(0, len(eval_uids), C.EVAL_USER_BATCH):
        ub = eval_uids[s:s+C.EVAL_USER_BATCH]; B = len(ub)
        h = user_hist[ub]; hr = np.repeat(np.arange(B), (h < n_items).sum(1)); hc = h[h < n_items]
        hist_csr = sparse.csr_matrix((np.ones(len(hc), np.float32), (hr, hc)), shape=(B, n_items))
        ch_topk = {c: topk_channel(c, ub, hist_csr) for c in CHANNELS}
        # RRF merge (vectorized scatter)
        rrf = torch.zeros(B, n_items, device=dev)
        w = torch.tensor(1.0 / (C.RRF_C + np.arange(K)), device=dev, dtype=torch.float32)
        for c in CHANNELS:
            ix = ch_topk[c]; valid = ix >= 0
            bi = np.repeat(np.arange(B), valid.sum(1)); ci = ix[valid]
            ws = w.repeat(B, 1).cpu().numpy()[valid]
            rrf[torch.from_numpy(bi).to(dev), torch.from_numpy(ci).to(dev)] += torch.from_numpy(ws).to(dev)
        mtopk = torch.topk(rrf, maxK, 1).indices.cpu().numpy()
        # per-user bookkeeping
        for i, u in enumerate(ub):
            t = test_flat[test_off[u]:test_off[u+1]]; m = len(t); n_eval += 1
            ut = "u_cold" if n_pos[u] <= 20 else ("u_warm" if n_pos[u] <= 60 else "u_hot")
            # per-channel recall
            ch_ranks = {}
            for c in CHANNELS:
                ix = ch_topk[c][i]; v = ix >= 0; rl[ix[v]] = np.arange(K)[v]
                r = rl[t]; rl[ix[v]] = K; ch_ranks[c] = r
                for k in Ks: per_ch[c][k] += (r < k).sum() / m
                tk = ig_ch[c]
                for j, it in enumerate(t):
                    g = ("head", "mid", "tail")[itier[it]]; tk[g]["n"] += 1; tk[g]["hit"] += int(r[j] < 100)
            # merged recall
            rl[mtopk[i]] = np.arange(maxK); rm = rl[t]; rl[mtopk[i]] = K
            for k in Ks:
                rec = (rm < k).sum() / m; merged[k] += rec; mg_utier[ut][k] += rec
            ug[ut]["n"] += 1; mg_utier[ut]["n"] += 1
            for j, it in enumerate(t):
                g = ("head", "mid", "tail")[itier[it]]; ig_ch["merged"][g]["n"] += 1; ig_ch["merged"][g]["hit"] += int(rm[j] < 100)
            # contribution @500 (which channels surfaced each merged-pool hit)
            for j, it in enumerate(t):
                if rm[j] < 500:
                    hits = [c for c in CHANNELS if ch_ranks[c][j] < 500]
                    for c in hits: contrib[c]["surfaced"] += 1
                    if len(hits) == 1: contrib[hits[0]]["unique"] += 1
        if s % (C.EVAL_USER_BATCH * 5) == 0: log(f"  eval {s+B}/{len(eval_uids)}")

    out = {"n_eval_users": n_eval, "n_channels": len(CHANNELS), "eval_sample": len(eval_uids),
           "per_channel_recall": {c: {f"R@{k}": per_ch[c][k]/n_eval for k in Ks} for c in CHANNELS},
           "merged_recall": {f"R@{k}": merged[k]/n_eval for k in Ks},
           "contribution@500": {c: {"surfaced_per_user": contrib[c]["surfaced"]/n_eval,
                                    "unique_per_user": contrib[c]["unique"]/n_eval} for c in CHANNELS},
           "merged_by_user_tier": {t: {f"R@{k}": mg_utier[t][k]/max(mg_utier[t]["n"],1) for k in Ks} | {"n": mg_utier[t]["n"]} for t in mg_utier},
           "channel_recall100_by_item_tier": {c: {t: ig_ch[c][t]["hit"]/max(ig_ch[c][t]["n"],1) for t in ("head","mid","tail")} for c in CHANNELS + ["merged"]}}
    json.dump(out, open(C.V7_DIR / "v7_results.json", "w"), indent=2)
    log("v7_recall DONE -> v7_results.json")
    print("\n=== per-channel vs merged set-Recall ===")
    for c in CHANNELS: print(f"  {c:14s} " + " ".join(f"R@{k}={out['per_channel_recall'][c][f'R@{k}']:.4f}" for k in Ks))
    print(f"  {'MERGED(RRF)':14s} " + " ".join(f"R@{k}={out['merged_recall'][f'R@{k}']:.4f}" for k in Ks))
    print("=== unique contribution @500 (hits/user only this channel found) ===")
    for c in CHANNELS: print(f"  {c:14s} surfaced={out['contribution@500'][c]['surfaced_per_user']:.3f} unique={out['contribution@500'][c]['unique_per_user']:.3f}")


if __name__ == "__main__":
    main()
