"""v4 cold-ITEM holdout eval: does item content help on items with NO training signal,
and does the reader-view PROFILE beat title+desc there?

Honest cold-start handling: a truly new item has no reviews, so for COLD items the
"profile" content falls back to title+desc (item_profile_emb already does this, but we
also force it here for the de-leaked holdout). Thus profile can only differ from desc on
WARM items (where reviews exist). Item usage features for cold items are neutralized
(de-leaked) by prepare_coldstart.

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> RECSYS_CONTENT=profile python v4_cold.py
"""
from __future__ import annotations
import copy, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as C
from evaluate import build_train_mask
from model import TwoTower, info_nce_logq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
NEG = float("-inf")
CONTENT = os.environ.get("RECSYS_CONTENT", "desc")
MLP = tuple(int(x) for x in os.environ.get("RECSYS_MLP", "256,128").split(","))
DOUT = int(os.environ.get("RECSYS_DOUT", "64")); DC = int(os.environ.get("RECSYS_DC", "128"))
TAG = os.environ.get("RECSYS_TAG", f"cold_{CONTENT}")


def load_content(meta, cold):
    desc = np.load(C.OUT_DIR / "item_text_emb.npy").astype(np.float32)
    if CONTENT == "desc":
        return desc
    prof = np.load(C.OUT_DIR / "item_profile_emb.npy").astype(np.float32).copy()
    prof[cold] = desc[cold]                       # cold items have no reviews -> profile == title+desc
    if CONTENT == "profile":
        return prof
    if CONTENT == "both":
        return np.concatenate([desc, prof], axis=1)
    raise ValueError(CONTENT)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(os.environ.get("RECSYS_SEED", C.SEED))
    torch.manual_seed(seed); np.random.seed(seed)
    meta = json.load(open(C.COLD_DIR / "meta_cold.json"))
    n_items, n_users, tag_pad = meta["n_items"], meta["n_users"], meta["tag_pad"]
    cold = np.load(C.COLD_DIR / "cold_items.npy")
    ct = load_content(meta, cold)
    feats = {
        "item_num": torch.from_numpy(np.load(C.COLD_DIR / "item_num_cold.npy")),
        "item_cat": torch.from_numpy(np.load(C.COLD_DIR / "item_cat_cold.npy")),
        "item_tags": torch.full((n_items, 1), tag_pad, dtype=torch.int64),
        "user_num": torch.from_numpy(np.load(C.OUT_DIR / "user_num.npy")),
        "user_liked": torch.full((n_users, 1), tag_pad, dtype=torch.int64),
        "user_disliked": torch.full((n_users, 1), tag_pad, dtype=torch.int64),
        "user_hist": torch.from_numpy(np.load(C.COLD_DIR / "user_hist_cold.npy").astype(np.int64)),
        "item_text": torch.from_numpy(ct),
        "id_lookup": torch.from_numpy(np.load(C.COLD_DIR / "id_lookup.npy")),
    }
    mc = copy.deepcopy(C.MODEL); mc.mlp_hidden = MLP; mc.d_out = DOUT
    model = TwoTower(n_items, n_users, meta["n_tags"], meta["item_cat_cardinality"],
                     feats["item_num"].shape[1], feats["user_num"].shape[1], mc,
                     item_mode="hybrid", use_text=True, d_content=DC, n_item_text=ct.shape[1],
                     use_user_content=False).to(dev)
    model.attach_features(feats, dev)
    log(f"[{TAG}] content={CONTENT}({ct.shape[1]}d) params={sum(p.numel() for p in model.parameters()):,}")
    pop = np.load(C.COLD_DIR / "popularity_cold.npy").astype(np.float64)
    log_q = torch.from_numpy(np.log((pop + 1) / (pop.sum() + len(pop)))).float().to(dev)

    pairs = torch.from_numpy(np.load(C.COLD_DIR / "train_pairs_cold.npy").astype(np.int64))
    ds = TensorDataset(pairs[:, 0], pairs[:, 1])
    g = torch.Generator().manual_seed(C.SEED)
    sm = torch.utils.data.RandomSampler(ds, num_samples=min(C.TRAIN.max_pairs_per_epoch, len(ds)),
                                        replacement=False, generator=g)
    ld = DataLoader(ds, batch_size=C.TRAIN.batch_size, sampler=sm, num_workers=C.TRAIN.num_workers,
                    drop_last=True, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=C.TRAIN.lr, weight_decay=C.TRAIN.weight_decay)
    val = np.load(C.OUT_DIR / "val.npy"); vu, vt = val[:, 0].copy(), val[:, 1].copy()
    best, patc = -1.0, 0; ckpt = C.COLD_DIR / f"ckpt_{TAG}.pt"

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
        if r > best: best, patc = r, 0; torch.save({"state_dict": model.state_dict()}, ckpt); msg += " *"
        else: patc += 1
        log(msg)
        if patc >= C.TRAIN.early_stop_patience: log("early stop"); break
    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"]); model.eval()

    off, flat = build_train_mask(n_users)
    test = np.load(C.OUT_DIR / "test.npy"); tu, tt = test[:, 0].copy(), test[:, 1].copy()
    iv = model.encode_all_items(dev); Ks = C.EVAL_KS; maxK = max(Ks)
    groups = ["all", "item_warm", "item_cold"]
    acc = {gp: {"n": 0, "mrr": 0.0, **{f"r{k}": 0 for k in Ks}, **{f"n{k}": 0.0 for k in Ks}} for gp in groups}
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
            for gp in ("all", "item_cold" if cold[tb[i]] else "item_warm"):
                a = acc[gp]; a["n"] += 1; a["mrr"] += 1.0/r
                for k in Ks:
                    if r <= k: a[f"r{k}"] += 1; a[f"n{k}"] += 1.0/np.log2(r+1)
    out = {gp: {"n": a["n"], "MRR": a["mrr"]/max(a["n"],1),
                **{f"Recall@{k}": a[f"r{k}"]/max(a["n"],1) for k in Ks},
                **{f"NDCG@{k}": a[f"n{k}"]/max(a["n"],1) for k in Ks}} for gp, a in acc.items()}
    out["cold_item_coverage@200"] = float(covered[cold].mean())
    json.dump({"tag": TAG, "content": CONTENT, "dnn": out}, open(C.COLD_DIR / f"eval_{TAG}.json", "w"), indent=2)
    print(f"\n===== v4-cold [{TAG}] =====")
    for gp in groups:
        x = out[gp]; print(f"  {gp:10s} n={x['n']:>7,} R@10={x['Recall@10']:.4f} R@100={x['Recall@100']:.4f} NDCG@10={x['NDCG@10']:.4f}")
    print(f"  ColdItemCoverage@200={out['cold_item_coverage@200']:.4f}")


if __name__ == "__main__":
    main()
