"""v4: item-content + architecture exploration on the two-tower (framework unchanged).

User tower = collaborative only (history pooling + behavior stats); NO user content
(v3 showed it doesn't help). Item tower = hybrid (id + content + numeric + categorical).
Configurable via env (for the background sweep):
  RECSYS_CONTENT = desc | profile | both     (item content source)
  RECSYS_MLP     = "256,128"                  (MLP hidden dims)
  RECSYS_DOUT    = 64                          (output embedding dim)
  RECSYS_DC      = 128                         (content projection dim)
  RECSYS_DID     = 64                          (item-ID embedding dim)
  RECSYS_TAG     = <label>                     (filename tag; default auto)

Eval: warm full-corpus + USER tiers (cold/warm/hot) + ITEM popularity tiers (head/mid/tail).

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> RECSYS_CONTENT=profile python v4.py
"""
from __future__ import annotations
import copy, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as C
import dataset as D
from evaluate import build_train_mask, user_npos, tier_of
from model import TwoTower, info_nce_logq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
NEG = float("-inf")
V4 = C.OUT_DIR.parent / "v4" / "sweep"; V4.mkdir(parents=True, exist_ok=True)

CONTENT = os.environ.get("RECSYS_CONTENT", "desc")
MLP = tuple(int(x) for x in os.environ.get("RECSYS_MLP", "256,128").split(","))
DOUT = int(os.environ.get("RECSYS_DOUT", "64"))
DC = int(os.environ.get("RECSYS_DC", "128"))
DID = int(os.environ.get("RECSYS_DID", "64"))
TAG = os.environ.get("RECSYS_TAG", f"{CONTENT}_mlp{'-'.join(map(str,MLP))}_d{DOUT}_dc{DC}_id{DID}")


def load_content():
    desc = np.load(C.OUT_DIR / "item_text_emb.npy").astype(np.float32)
    if CONTENT == "desc": return desc
    prof = np.load(C.OUT_DIR / "item_profile_emb.npy").astype(np.float32)
    if CONTENT == "profile": return prof
    if CONTENT == "both": return np.concatenate([desc, prof], axis=1)
    raise ValueError(CONTENT)


def load_feats(meta):
    n_items, n_users, tag_pad = meta["n_items"], meta["n_users"], meta["tag_pad"]
    ct = load_content()
    return {
        "item_num": torch.from_numpy(np.load(C.OUT_DIR / "item_num.npy")),
        "item_cat": torch.from_numpy(np.load(C.OUT_DIR / "item_cat.npy")),
        "item_tags": torch.full((n_items, 1), tag_pad, dtype=torch.int64),
        "user_num": torch.from_numpy(np.load(C.OUT_DIR / "user_num.npy")),
        "user_liked": torch.full((n_users, 1), tag_pad, dtype=torch.int64),
        "user_disliked": torch.full((n_users, 1), tag_pad, dtype=torch.int64),
        "user_hist": torch.from_numpy(np.load(C.OUT_DIR / "user_hist.npy").astype(np.int64)),
        "item_text": torch.from_numpy(ct),
    }, ct.shape[1]


def loader(bs, mx, nw, seed):
    p = torch.from_numpy(np.load(C.OUT_DIR / "train_pairs.npy").astype(np.int64))
    ds = TensorDataset(p[:, 0], p[:, 1])
    if mx and mx < len(ds):
        g = torch.Generator().manual_seed(seed)
        sm = torch.utils.data.RandomSampler(ds, num_samples=mx, replacement=False, generator=g)
        return DataLoader(ds, batch_size=bs, sampler=sm, num_workers=nw, drop_last=True, pin_memory=True)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True)


def item_tier_fn(pop):
    q50, q90 = np.quantile(pop, 0.5), np.quantile(pop, 0.9)
    def f(iid):
        p = pop[iid]
        return "head" if p > q90 else ("tail" if p <= q50 else "mid")
    return f


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(os.environ.get("RECSYS_SEED", C.SEED))
    torch.manual_seed(seed); np.random.seed(seed)        # reproducible weight init (fair sweep)
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    feats, n_text = load_feats(meta)
    mc = copy.deepcopy(C.MODEL); mc.mlp_hidden = MLP; mc.d_out = DOUT; mc.d_id = DID
    model = TwoTower(n_items, n_users, meta["n_tags"], meta["item_cat_cardinality"],
                     feats["item_num"].shape[1], feats["user_num"].shape[1], mc,
                     item_mode="hybrid", use_text=True, d_content=DC, n_item_text=n_text,
                     use_user_content=False).to(dev)
    model.attach_features(feats, dev)
    log(f"[{TAG}] content={CONTENT}({n_text}d) mlp={MLP} d_out={DOUT} d_content={DC} params={sum(p.numel() for p in model.parameters()):,}")
    pop = np.load(C.OUT_DIR / "popularity.npy").astype(np.float64)
    log_q = torch.from_numpy(np.log((pop + 1) / (pop.sum() + len(pop)))).float().to(dev)
    ld = loader(C.TRAIN.batch_size, C.TRAIN.max_pairs_per_epoch, C.TRAIN.num_workers, C.SEED)
    opt = torch.optim.Adam(model.parameters(), lr=C.TRAIN.lr, weight_decay=C.TRAIN.weight_decay)
    val = np.load(C.OUT_DIR / "val.npy"); vu, vt = val[:, 0].copy(), val[:, 1].copy()
    best, pat = -1.0, 0; ckpt = V4 / f"ckpt_{TAG}.pt"

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
            s2 = torch.cat([sc, torch.full((sc.size(0), 1), NEG, device=dev)], 1)
            s2.scatter_(1, model.user_hist[uid], NEG); sc = s2[:, :n_items]
            hit += ((sc > sc.gather(1, tgt.unsqueeze(1))).sum(1) + 1 <= k).sum().item()
        return hit / len(u)

    for ep in range(1, C.TRAIN.epochs + 1):
        model.train(); tot = nb = 0
        for uid, tgt in ld:
            uid, tgt = uid.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            loss = info_nce_logq(model.encode_user(uid, drop_target=tgt), model.encode_item(tgt),
                                 tgt, log_q[tgt], C.TRAIN.temperature)
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

    # ---- eval: warm full + user tiers + item popularity tiers ----
    off, flat = build_train_mask(n_users); npos = user_npos(n_users)
    itier = item_tier_fn(pop)
    test = np.load(C.OUT_DIR / "test.npy"); tu, tt = test[:, 0].copy(), test[:, 1].copy()
    iv = model.encode_all_items(dev)
    Ks = C.EVAL_KS; maxK = max(Ks)
    groups = ["all", "u_cold", "u_warm", "u_hot", "i_head", "i_mid", "i_tail"]
    acc = {g: {"n": 0, "mrr": 0.0, **{f"r{k}": 0 for k in Ks}, **{f"n{k}": 0.0 for k in Ks}} for g in groups}
    covered = np.zeros(n_items, bool)
    for s in range(0, len(tu), C.EVAL_USER_BATCH):
        ub, tb = tu[s:s+C.EVAL_USER_BATCH], tt[s:s+C.EVAL_USER_BATCH]; b = len(ub)
        sc = model.encode_user(torch.from_numpy(ub).long().to(dev)) @ iv.t()
        cols = np.concatenate([flat[off[u]:off[u+1]] for u in ub]) if b else np.array([], np.int64)
        rows = np.repeat(np.arange(b), off[ub+1]-off[ub])
        if len(cols): sc[torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)] = NEG
        tgt = torch.from_numpy(tb).long().to(dev)
        rank = ((sc > sc.gather(1, tgt.unsqueeze(1))).sum(1) + 1).cpu().numpy()
        covered[torch.topk(sc, maxK, 1).indices.cpu().numpy().ravel()] = True
        for i in range(b):
            r = rank[i]
            for g in ("all", "u_" + tier_of(npos[ub[i]]), "i_" + itier(tb[i])):
                a = acc[g]; a["n"] += 1; a["mrr"] += 1.0/r
                for k in Ks:
                    if r <= k: a[f"r{k}"] += 1; a[f"n{k}"] += 1.0/np.log2(r+1)
    out = {g: {"n": a["n"], "MRR": a["mrr"]/max(a["n"],1),
               **{f"Recall@{k}": a[f"r{k}"]/max(a["n"],1) for k in Ks},
               **{f"NDCG@{k}": a[f"n{k}"]/max(a["n"],1) for k in Ks}} for g, a in acc.items()}
    out["coverage"] = {f"Coverage@{maxK}": float(covered.mean())}
    res = {"tag": TAG, "content": CONTENT, "mlp": list(MLP), "d_out": DOUT, "d_content": DC,
           "best_val_R100": best, "dnn": out}
    json.dump(res, open(V4 / f"eval_{TAG}.json", "w"), indent=2)
    print(f"\n===== v4 [{TAG}]  (best_val_R@100={best:.4f}) =====")
    for g in groups:
        x = out[g]; print(f"  {g:7s} n={x['n']:>7,} R@10={x['Recall@10']:.4f} R@100={x['Recall@100']:.4f} "
                          f"NDCG@10={x['NDCG@10']:.4f} MRR={x['MRR']:.4f}")
    print(f"  Coverage@200={out['coverage']['Coverage@200']:.4f}")


if __name__ == "__main__":
    main()
