"""v5 driver — A/B/C/D ablation on the two-tower (warm full corpus + user/item tiers + per-language).

Variant (env RECSYS_VARIANT) selects the lever combo (each isolates one mechanism vs A0 base):
  A0  atomic   id, in-batch negs,           no user-lang, no lang-penalty   (= v4-equivalent base)
  A1  semantic id (RQ-VAE codes replace atomic)
  A2  hybrid   id (atomic ⊕ semantic)
  B1  atomic + dislike hard-negatives (rating<=2 explicit negs in the loss)
  C1  atomic + user-language feature branch
  C2  atomic + user-language feature + retrieval-time language-match penalty
  D   hybrid + dislike-negs + user-lang + lang-penalty   (= A2 + B1 + C2, the delivered v5)

Other env: RECSYS_CONTENT=desc|tags|content|both (default desc), RECSYS_SEED, RECSYS_TAG,
           RECSYS_MLP/DOUT/DC (default 256,128 / 64 / 128).

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> RECSYS_VARIANT=D RECSYS_SEED=42 python v5.py
"""
from __future__ import annotations
import copy, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as C
import dataset as D
from evaluate import build_train_mask, user_npos, tier_of
from model import TwoTowerV5, info_nce_logq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
NEG = float("-inf")

# variant -> (id_mode, use_dislike, use_ulang, use_lang_penalty)
VARIANTS = {
    "A0": ("atomic",   False, False, False),
    "A1": ("semantic", False, False, False),
    "A2": ("hybrid",   False, False, False),
    "B1": ("atomic",   True,  False, False),
    "C1": ("atomic",   False, True,  False),
    "C2": ("atomic",   False, True,  True),
    "D":  ("hybrid",   True,  True,  True),
}
VARIANT = os.environ.get("RECSYS_VARIANT", "A0")
ID_MODE, USE_DISLIKE, USE_ULANG, USE_LANGPEN = VARIANTS[VARIANT]
CONTENT = os.environ.get("RECSYS_CONTENT", "desc")
MLP = tuple(int(x) for x in os.environ.get("RECSYS_MLP", "256,128").split(","))
DOUT = int(os.environ.get("RECSYS_DOUT", "64")); DC = int(os.environ.get("RECSYS_DC", "128"))
SEED = int(os.environ.get("RECSYS_SEED", C.SEED))
TAG = os.environ.get("RECSYS_TAG", f"{VARIANT}_s{SEED}")
EPOCHS = int(os.environ.get("RECSYS_EPOCHS", C.TRAIN.epochs))           # smoke override
MAXPAIRS = int(os.environ.get("RECSYS_MAXPAIRS", C.TRAIN.max_pairs_per_epoch))


def load_content():
    desc = np.load(C.V5_DIR / "item_desc_emb.npy").astype(np.float32)
    if CONTENT == "desc": return desc
    tags = np.load(C.V5_DIR / "item_tags_emb.npy").astype(np.float32)
    if CONTENT == "tags": return tags
    if CONTENT == "content": return np.load(C.V5_DIR / "item_content_emb.npy").astype(np.float32)
    if CONTENT == "both": return np.concatenate([desc, tags], axis=1)
    raise ValueError(CONTENT)


def item_tier_fn(pop):
    q50, q90 = np.quantile(pop, 0.5), np.quantile(pop, 0.9)
    def f(iid):
        p = pop[iid]
        return "head" if p > q90 else ("tail" if p <= q50 else "mid")
    return f


def lang_bucket(book_lang, vocab):
    inv = {v: k for k, v in vocab["vocab"].items()}
    code = np.array([inv.get(int(x), "other") for x in book_lang])
    return np.array([c if c in C.PER_LANG else "other" for c in code])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    ct = load_content(); n_text = ct.shape[1]
    feats = {
        "item_num": torch.from_numpy(np.load(C.BASE_DIR / "item_num.npy")),
        "item_cat": torch.from_numpy(np.load(C.BASE_DIR / "item_cat.npy")),
        "item_tags": torch.full((n_items, 1), meta["tag_pad"], dtype=torch.int64),
        "user_num": torch.from_numpy(np.load(C.BASE_DIR / "user_num.npy")),
        "user_hist": torch.from_numpy(np.load(C.BASE_DIR / "user_hist.npy").astype(np.int64)),
        "item_text": torch.from_numpy(ct),
    }
    sem_codes = np.load(C.V5_DIR / "sem_codes.npy").astype(np.int64)
    if ID_MODE in ("semantic", "hybrid"):
        feats["sem_codes"] = torch.from_numpy(sem_codes)
    Wlang = np.load(C.V5_DIR / "user_lang_w.npy").astype(np.float32)
    if USE_ULANG:
        feats["user_lang"] = torch.from_numpy(Wlang)
    n_ulang = Wlang.shape[1]

    mc = copy.deepcopy(C.MODEL); mc.mlp_hidden = MLP; mc.d_out = DOUT
    model = TwoTowerV5(n_items, n_users, meta["n_tags"], meta["item_cat_cardinality"],
                       feats["item_num"].shape[1], feats["user_num"].shape[1], mc,
                       id_mode=ID_MODE, use_text=True, d_content=DC, n_item_text=n_text,
                       n_codes_L=C.RQ_L, n_codes_K=C.RQ_K, use_ulang=USE_ULANG, n_ulang=n_ulang).to(dev)
    model.attach_features(feats, dev)
    log(f"[{TAG}] variant={VARIANT} id={ID_MODE} dislike={USE_DISLIKE} ulang={USE_ULANG} "
        f"langpen={USE_LANGPEN} content={CONTENT}({n_text}d) params={sum(p.numel() for p in model.parameters()):,}")

    pop = np.load(C.BASE_DIR / "popularity.npy").astype(np.float64)
    log_q = torch.from_numpy(np.log((pop + 1) / (pop.sum() + len(pop)))).float().to(dev)
    dislike_pad = torch.from_numpy(np.load(C.V5_DIR / "dislike_pad.npy")).to(dev) if USE_DISLIKE else None
    neg_bias = float(np.log(C.DISLIKE_LAMBDA))
    book_lang = torch.from_numpy(np.load(C.V5_DIR / "book_lang.npy")).to(dev)
    Wlang_t = torch.from_numpy(Wlang).to(dev)

    # train loader (subsample pairs/epoch, fixed seed for fair sweep — mirror v4)
    p = torch.from_numpy(np.load(C.BASE_DIR / "train_pairs.npy").astype(np.int64))
    ds = TensorDataset(p[:, 0], p[:, 1])
    g = torch.Generator().manual_seed(C.SEED)
    sm = torch.utils.data.RandomSampler(ds, num_samples=min(MAXPAIRS, len(ds)),
                                        replacement=False, generator=g)
    ld = DataLoader(ds, batch_size=C.TRAIN.batch_size, sampler=sm, num_workers=C.TRAIN.num_workers,
                    drop_last=True, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=C.TRAIN.lr, weight_decay=C.TRAIN.weight_decay)
    val = np.load(C.BASE_DIR / "val.npy"); vu, vt = val[:, 0].copy(), val[:, 1].copy()
    best, pat = -1.0, 0; ckpt = C.SWEEP_DIR / f"ckpt_{TAG}.pt"

    def lang_penalty(sc, uid):                              # sc:[B,n_items], uid:[B]
        wb = Wlang_t[uid][:, book_lang]                     # [B, n_items] user weight on each book's lang
        return sc - C.LANG_PENALTY * (1.0 - wb)

    @torch.no_grad()
    def quick_val(iv, k=100, sample=30000):
        model.eval(); u, t = vu, vt
        if len(u) > sample:
            s = np.random.RandomState(0).choice(len(u), sample, replace=False); u, t = u[s], t[s]
        hit = 0
        for s in range(0, len(u), C.EVAL_USER_BATCH):
            uid = torch.from_numpy(u[s:s+C.EVAL_USER_BATCH]).long().to(dev)
            tgt = torch.from_numpy(t[s:s+C.EVAL_USER_BATCH]).long().to(dev)
            sc = model.encode_user(uid) @ iv.t()
            if USE_LANGPEN: sc = lang_penalty(sc, uid)
            s2 = torch.cat([sc, torch.full((sc.size(0), 1), NEG, device=dev)], 1)
            s2.scatter_(1, model.user_hist[uid], NEG); sc = s2[:, :n_items]
            hit += ((sc > sc.gather(1, tgt.unsqueeze(1))).sum(1) + 1 <= k).sum().item()
        return hit / len(u)

    for ep in range(1, EPOCHS + 1):
        model.train(); tot = nb = 0
        for uid, tgt in ld:
            uid, tgt = uid.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            uvec = model.encode_user(uid, drop_target=tgt)
            ivec = model.encode_item(tgt)
            nvec = nmask = nid = None
            if USE_DISLIKE:
                nid = dislike_pad[uid]                       # [B, M] (pad=n_items)
                nmask = nid < n_items
                nvec = model.encode_item(nid.clamp(max=n_items-1).reshape(-1)).reshape(nid.size(0), nid.size(1), -1)
            loss = info_nce_logq(uvec, ivec, tgt, log_q[tgt], C.TRAIN.temperature,
                                 neg_vec=nvec, neg_mask=nmask, neg_bias=neg_bias, neg_ids=nid)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.TRAIN.grad_clip); opt.step()
            tot += loss.item(); nb += 1
        iv = model.encode_all_items(dev); r = quick_val(iv)
        msg = f"[{TAG}] ep{ep} loss={tot/nb:.4f} val_R@100={r:.4f}"
        if r > best: best, pat = r, 0; torch.save({"state_dict": model.state_dict(), "tag": TAG}, ckpt); msg += " *"
        else: pat += 1
        log(msg)
        if pat >= C.TRAIN.early_stop_patience: log("early stop"); break
    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"]); model.eval()

    # ---- eval: warm full + user tiers + item tiers + per-language ----
    off, flat = build_train_mask(n_users); npos = user_npos(n_users)
    itier = item_tier_fn(pop)
    vocab = json.load(open(C.V5_DIR / "lang_vocab.json"))
    tbl_lang = lang_bucket(np.load(C.V5_DIR / "book_lang.npy"), vocab)
    test = np.load(C.BASE_DIR / "test.npy"); tu, tt = test[:, 0].copy(), test[:, 1].copy()
    iv = model.encode_all_items(dev); Ks = C.EVAL_KS; maxK = max(Ks)
    langs = [f"L_{l}" for l in C.PER_LANG] + ["L_other"]
    groups = ["all", "u_cold", "u_warm", "u_hot", "i_head", "i_mid", "i_tail"] + langs
    acc = {g: {"n": 0, "mrr": 0.0, **{f"r{k}": 0 for k in Ks}, **{f"n{k}": 0.0 for k in Ks}} for g in groups}
    covered = np.zeros(n_items, bool)
    for s in range(0, len(tu), C.EVAL_USER_BATCH):
        ub, tb = tu[s:s+C.EVAL_USER_BATCH], tt[s:s+C.EVAL_USER_BATCH]; b = len(ub)
        uid = torch.from_numpy(ub).long().to(dev)
        sc = model.encode_user(uid) @ iv.t()
        if USE_LANGPEN: sc = lang_penalty(sc, uid)
        cols = np.concatenate([flat[off[u]:off[u+1]] for u in ub]) if b else np.array([], np.int64)
        rows = np.repeat(np.arange(b), off[ub+1]-off[ub])
        if len(cols): sc[torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)] = NEG
        tgt = torch.from_numpy(tb).long().to(dev)
        rank = ((sc > sc.gather(1, tgt.unsqueeze(1))).sum(1) + 1).cpu().numpy()
        covered[torch.topk(sc, maxK, 1).indices.cpu().numpy().ravel()] = True
        for i in range(b):
            r = rank[i]
            for g in ("all", "u_" + tier_of(npos[ub[i]]), "i_" + itier(tb[i]), "L_" + tbl_lang[tb[i]]):
                a = acc[g]; a["n"] += 1; a["mrr"] += 1.0/r
                for k in Ks:
                    if r <= k: a[f"r{k}"] += 1; a[f"n{k}"] += 1.0/np.log2(r+1)
    out = {g: {"n": a["n"], "MRR": a["mrr"]/max(a["n"],1),
               **{f"Recall@{k}": a[f"r{k}"]/max(a["n"],1) for k in Ks},
               **{f"NDCG@{k}": a[f"n{k}"]/max(a["n"],1) for k in Ks}} for g, a in acc.items()}
    out["coverage"] = {f"Coverage@{maxK}": float(covered.mean())}
    res = {"tag": TAG, "variant": VARIANT, "id_mode": ID_MODE, "use_dislike": USE_DISLIKE,
           "use_ulang": USE_ULANG, "use_langpen": USE_LANGPEN, "content": CONTENT, "seed": SEED,
           "mlp": list(MLP), "d_out": DOUT, "best_val_R100": best, "dnn": out}
    json.dump(res, open(C.SWEEP_DIR / f"eval_{TAG}.json", "w"), indent=2)
    print(f"\n===== v5 [{TAG}] variant={VARIANT} (best_val_R@100={best:.4f}) =====")
    for g in ["all", "u_cold", "u_warm", "u_hot", "i_head", "i_mid", "i_tail"]:
        x = out[g]; print(f"  {g:7s} n={x['n']:>7,} R@10={x['Recall@10']:.4f} R@100={x['Recall@100']:.4f} NDCG@10={x['NDCG@10']:.4f}")
    print("  per-lang R@100: " + " ".join(f"{l[2:]}={out[l]['Recall@100']:.3f}(n{out[l]['n']})" for l in langs if out[l]['n']>0))
    print(f"  Coverage@200={out['coverage']['Coverage@200']:.4f}")


if __name__ == "__main__":
    main()
