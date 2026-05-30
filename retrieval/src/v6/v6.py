"""v6 driver — semantic-ID model variants x graded-negative schemes, SET-based leave-k-out eval.

  RECSYS_MODEL = M0 (atomic) | M1 (hybrid+content codes) | M2 (hybrid+collab codes) | M3 (hybrid+big codes)
  RECSYS_NEG   = N0 (in-batch only) | N1 (+hard) | N2 (+hard+soft, default)
  RECSYS_SEED, RECSYS_TAG, RECSYS_EPOCHS/MAXPAIRS (smoke)

Eval: each user has a SET of held-out test items (~10%). Recall@K (user-macro) = mean_u |topK ∩ test_u|/|test_u|,
masking train+val. Item-tier / per-language slices use per-(user,item) micro hit-rate.

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> RECSYS_MODEL=M2 RECSYS_NEG=N2 python v6.py
"""
from __future__ import annotations
import copy, json, os, time
import duckdb
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as C
import dataset as D
from model import TwoTowerV6, info_nce_logq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
NEG = float("-inf")

MODEL = os.environ.get("RECSYS_MODEL", "M0")
NEGS = os.environ.get("RECSYS_NEG", "N2")
SEED = int(os.environ.get("RECSYS_SEED", C.SEED))
TAG = os.environ.get("RECSYS_TAG", f"{MODEL}_{NEGS}_s{SEED}")
EPOCHS = int(os.environ.get("RECSYS_EPOCHS", C.TRAIN.epochs))
MAXPAIRS = int(os.environ.get("RECSYS_MAXPAIRS", C.TRAIN.max_pairs_per_epoch))
# model -> (id_mode, sem_source)
MODELS = {"M0": ("atomic", None), "M1": ("hybrid", "content"), "M2": ("hybrid", "collab"), "M3": ("hybrid", "big")}
ID_MODE, SEM_SRC = MODELS[MODEL]
RQCFG = {"content": C.RQ_CONTENT, "collab": C.RQ_COLLAB, "big": C.RQ_BIG}


def build_mask(splits):
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    q = ",".join(f"'{s}'" for s in splits)
    df = con.execute(f"SELECT uid, iid FROM read_parquet('{C.BASE_DIR}/split.parquet') WHERE split IN ({q}) ORDER BY uid").fetchdf()
    uid = df["uid"].to_numpy(np.int64); iid = df["iid"].to_numpy(np.int64)
    return uid, iid


def csr(uid, iid, n_users):
    off = np.zeros(n_users + 1, np.int64); np.add.at(off, uid + 1, 1); off = np.cumsum(off)
    return off, iid           # iid already grouped by uid (ORDER BY uid)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    desc = np.load(C.CONTENT_DIR / "item_desc_emb.npy").astype(np.float32)
    feats = {
        "item_num": torch.from_numpy(np.load(C.BASE_DIR / "item_num.npy")),
        "item_cat": torch.from_numpy(np.load(C.BASE_DIR / "item_cat.npy")),
        "item_tags": torch.full((n_items, 1), meta["tag_pad"], dtype=torch.int64),
        "user_num": torch.from_numpy(np.load(C.BASE_DIR / "user_num.npy")),
        "user_hist": torch.from_numpy(np.load(C.BASE_DIR / "user_hist.npy").astype(np.int64)),
        "item_text": torch.from_numpy(desc),
    }
    L = K = None
    if SEM_SRC:
        codes = np.load(C.V6_DIR / f"sem_codes_{SEM_SRC}.npy").astype(np.int64)
        feats["sem_codes"] = torch.from_numpy(codes); L, K = RQCFG[SEM_SRC]["L"], RQCFG[SEM_SRC]["K"]
    mc = copy.deepcopy(C.MODEL)
    model = TwoTowerV6(n_items, n_users, meta["n_tags"], meta["item_cat_cardinality"],
                       feats["item_num"].shape[1], feats["user_num"].shape[1], mc, id_mode=ID_MODE,
                       use_text=True, d_content=128, n_item_text=desc.shape[1],
                       n_codes_L=L or 3, n_codes_K=K or 256).to(dev)
    model.attach_features(feats, dev)
    log(f"[{TAG}] model={MODEL}({ID_MODE},codes={SEM_SRC}) neg={NEGS} params={sum(p.numel() for p in model.parameters()):,}")

    pop = np.load(C.BASE_DIR / "popularity.npy").astype(np.float64)
    log_q = torch.from_numpy(np.log((pop + 1) / (pop.sum() + len(pop)))).float().to(dev)
    # graded negative pools
    use_hard = NEGS in ("N1", "N2"); use_soft = NEGS == "N2"
    hard_pad = torch.from_numpy(np.load(C.V6_DIR / "hard_pad.npy")).to(dev) if use_hard else None
    soft_pad = torch.from_numpy(np.load(C.V6_DIR / "soft_pad.npy")).to(dev) if use_soft else None
    b_hard, b_soft = float(np.log(C.LAMBDA_HARD)), float(np.log(C.LAMBDA_SOFT))

    def gather_negs(uid):
        out = []
        for pad, bias in ((hard_pad, b_hard), (soft_pad, b_soft)):
            if pad is None: continue
            nid = pad[uid]; nmask = nid < n_items
            nvec = model.encode_item(nid.clamp(max=n_items - 1).reshape(-1)).reshape(nid.size(0), nid.size(1), -1)
            out.append((nvec, nmask, bias, nid))
        return out or None

    p = torch.from_numpy(np.load(C.BASE_DIR / "train_pairs.npy").astype(np.int64))
    ds = TensorDataset(p[:, 0], p[:, 1])
    g = torch.Generator().manual_seed(C.SEED)
    sm = torch.utils.data.RandomSampler(ds, num_samples=min(MAXPAIRS, len(ds)), replacement=False, generator=g)
    ld = DataLoader(ds, batch_size=C.TRAIN.batch_size, sampler=sm, num_workers=C.TRAIN.num_workers, drop_last=True, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=C.TRAIN.lr, weight_decay=C.TRAIN.weight_decay)

    # ---- masks + held-out sets (CSR over n_users) ----
    mu, mi = build_mask(["train", "val"]); mask_off, mask_flat = csr(mu, mi, n_users)   # mask for TEST eval
    tru, tri = build_mask(["train"]); trmask_off, trmask_flat = csr(tru, tri, n_users)  # mask for VAL eval
    vu, vi = build_mask(["val"]); val_off, val_flat = csr(vu, vi, n_users)
    tu_, ti_ = build_mask(["test"]); test_off, test_flat = csr(tu_, ti_, n_users)
    eval_uids = np.where(np.diff(test_off) > 0)[0]
    val_uids = np.where(np.diff(val_off) > 0)[0]
    n_pos = (mask_off[1:] - mask_off[:-1]) + (test_off[1:] - test_off[:-1])   # total positives per user
    q50, q90 = np.quantile(pop, 0.5), np.quantile(pop, 0.9)
    itier = np.where(pop > q90, 0, np.where(pop <= q50, 2, 1))                # 0=head 1=mid 2=tail
    book_lang = np.load(C.CONTENT_DIR / "book_lang.npy")
    vocab = json.load(open(C.CONTENT_DIR / "lang_vocab.json"))
    inv = {v: k for k, v in vocab["vocab"].items()}
    blang = np.array([inv.get(int(x), "other") for x in book_lang])
    blang = np.array([c if c in C.PER_LANG else "other" for c in blang])

    @torch.no_grad()
    def set_recall(iv, uids, off, flat, moff, mflat, k=100, sample=None):
        model.eval()
        if sample and len(uids) > sample:
            uids = uids[np.random.RandomState(0).choice(len(uids), sample, replace=False)]
        rl = np.full(n_items, k, np.int32); tot = cnt = 0.0
        for s in range(0, len(uids), C.EVAL_USER_BATCH):
            ub = uids[s:s + C.EVAL_USER_BATCH]
            sc = model.encode_user(torch.from_numpy(ub).long().to(dev)) @ iv.t()
            cols = np.concatenate([mflat[moff[u]:moff[u + 1]] for u in ub]); rows = np.repeat(np.arange(len(ub)), moff[ub + 1] - moff[ub])
            if len(cols): sc[torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)] = NEG
            tk = torch.topk(sc, k, 1).indices.cpu().numpy()
            for i, u in enumerate(ub):
                t = flat[off[u]:off[u + 1]]
                hit = (np.isin(t, tk[i])).sum(); tot += hit / len(t); cnt += 1
        return tot / max(cnt, 1)

    best, pat = -1.0, 0; ckpt = C.SWEEP_DIR / f"ckpt_{TAG}.pt"
    for ep in range(1, EPOCHS + 1):
        model.train(); tl = nb = 0
        for uid, tgt in ld:
            uid, tgt = uid.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            loss = info_nce_logq(model.encode_user(uid, drop_target=tgt), model.encode_item(tgt),
                                 tgt, log_q[tgt], C.TRAIN.temperature, negs=gather_negs(uid))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.TRAIN.grad_clip); opt.step()
            tl += loss.item(); nb += 1
        iv = model.encode_all_items(dev)
        r = set_recall(iv, val_uids, val_off, val_flat, trmask_off, trmask_flat, k=100, sample=20000)
        msg = f"[{TAG}] ep{ep} loss={tl/nb:.4f} val_setR@100={r:.4f}"
        if r > best: best, pat = r, 0; torch.save({"state_dict": model.state_dict(), "tag": TAG}, ckpt); msg += " *"
        else: pat += 1
        log(msg)
        if pat >= C.TRAIN.early_stop_patience: log("early stop"); break
    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"]); model.eval()

    # ---- final SET-based eval on test (mask train+val) ----
    iv = model.encode_all_items(dev); Ks = C.EVAL_KS; maxK = max(Ks)
    ugroups = ["all", "u_cold", "u_warm", "u_hot"]
    igroups = ["i_head", "i_mid", "i_tail"] + [f"L_{l}" for l in C.PER_LANG] + ["L_other"]
    uacc = {gp: {"rec": {k: 0.0 for k in Ks}, "ndcg": 0.0, "n": 0} for gp in ugroups}
    iacc = {gp: {**{k: 0 for k in Ks}, "n": 0} for gp in igroups}
    def utier(u): npos = n_pos[u]; return "u_cold" if npos <= 20 else ("u_warm" if npos <= 60 else "u_hot")
    rl = np.full(n_items, maxK, np.int32)
    idcg_cache = {}
    def idcg(m, k):
        key = (min(m, k),);
        if key not in idcg_cache: idcg_cache[key] = sum(1.0 / np.log2(j + 2) for j in range(min(m, k)))
        return idcg_cache[key]
    for s in range(0, len(eval_uids), C.EVAL_USER_BATCH):
        ub = eval_uids[s:s + C.EVAL_USER_BATCH]
        sc = model.encode_user(torch.from_numpy(ub).long().to(dev)) @ iv.t()
        cols = np.concatenate([mask_flat[mask_off[u]:mask_off[u + 1]] for u in ub]); rows = np.repeat(np.arange(len(ub)), mask_off[ub + 1] - mask_off[ub])
        if len(cols): sc[torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)] = NEG
        tk = torch.topk(sc, maxK, 1).indices.cpu().numpy()
        for i, u in enumerate(ub):
            t = test_flat[test_off[u]:test_off[u + 1]]; m = len(t)
            rl[tk[i]] = np.arange(maxK); ranks = rl[t]; rl[tk[i]] = maxK     # rank (0..maxK) of each test item
            gp = utier(u); a = uacc["all"]; b = uacc[gp]
            for k in Ks:
                rec = (ranks < k).sum() / m
                a["rec"][k] += rec; b["rec"][k] += rec
            dcg = sum(1.0 / np.log2(r + 2) for r in ranks if r < 10)
            nd = dcg / idcg(m, 10)
            a["ndcg"] += nd; b["ndcg"] += nd; a["n"] += 1; b["n"] += 1
            for j, it in enumerate(t):                                       # per-item slices (micro)
                ig = f"i_{['head','mid','tail'][itier[it]]}"; lg = f"L_{blang[it]}"
                for k in Ks:
                    hit = int(ranks[j] < k)
                    iacc[ig][k] += hit; iacc[lg][k] += hit
                iacc[ig]["n"] += 1; iacc[lg]["n"] += 1
    out = {}
    for gp, a in uacc.items():
        out[gp] = {"n": a["n"], "NDCG@10": a["ndcg"] / max(a["n"], 1), **{f"Recall@{k}": a["rec"][k] / max(a["n"], 1) for k in Ks}}
    for gp, a in iacc.items():
        out[gp] = {"n": a["n"], **{f"Recall@{k}": a[k] / max(a["n"], 1) for k in Ks}}
    res = {"tag": TAG, "model": MODEL, "id_mode": ID_MODE, "sem_src": SEM_SRC, "neg": NEGS, "seed": SEED,
           "best_val_setR100": best, "dnn": out}
    json.dump(res, open(C.SWEEP_DIR / f"eval_{TAG}.json", "w"), indent=2)
    print(f"\n===== v6 [{TAG}] model={MODEL} neg={NEGS} (best_val_setR@100={best:.4f}) =====")
    for gp in ugroups:
        x = out[gp]; print(f"  {gp:7s} n={x['n']:>7,} R@10={x['Recall@10']:.4f} R@100={x['Recall@100']:.4f} NDCG@10={x['NDCG@10']:.4f}")
    for gp in ["i_head", "i_mid", "i_tail"]:
        print(f"  {gp:7s} n={out[gp]['n']:>7,} R@100={out[gp]['Recall@100']:.4f}")
    print("  per-lang R@100: " + " ".join(f"{l}={out['L_'+l]['Recall@100']:.3f}" for l in C.PER_LANG if out['L_'+l]['n'] > 0))


if __name__ == "__main__":
    main()
