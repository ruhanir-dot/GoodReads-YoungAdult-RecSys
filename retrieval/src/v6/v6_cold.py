"""v6 cold-ITEM holdout: do CONTENT vs COLLAB semantic codes give cold items (no training signal)
a usable vector? Trains on the de-leaked cold holdout, reports per-(user,test-item) hit bucketed by
cold/warm item + cold-item coverage. Set-based test (leave-k-out).

  RECSYS_MODEL = M0 (atomic) | M1 (content codes) | M2 (collab codes)
  CUDA_VISIBLE_DEVICES=<idle> RECSYS_MODEL=M2 python v6_cold.py
"""
from __future__ import annotations
import copy, json, os, time
import duckdb, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import config as C
from model import TwoTowerV6, info_nce_logq

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
NEG = float("-inf")
MODEL = os.environ.get("RECSYS_MODEL", "M0"); SEED = int(os.environ.get("RECSYS_SEED", C.SEED))
TAG = os.environ.get("RECSYS_TAG", f"cold_{MODEL}_s{SEED}")
MODELS = {"M0": ("atomic", None), "M1": ("hybrid", "content"), "M2": ("hybrid", "collab"), "M3": ("hybrid", "big")}
ID_MODE, SEM_SRC = MODELS[MODEL]; RQCFG = {"content": C.RQ_CONTENT, "collab": C.RQ_COLLAB, "big": C.RQ_BIG}


def csr_from_split(splits, n_users):
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    q = ",".join(f"'{s}'" for s in splits)
    df = con.execute(f"SELECT uid,iid FROM read_parquet('{C.BASE_DIR}/split.parquet') WHERE split IN ({q}) ORDER BY uid").fetchdf()
    uid = df["uid"].to_numpy(np.int64); iid = df["iid"].to_numpy(np.int64)
    off = np.zeros(n_users + 1, np.int64); np.add.at(off, uid + 1, 1); off = np.cumsum(off)
    return off, iid


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    meta = json.load(open(C.COLD_DIR / "meta_cold.json")); n_items, n_users = meta["n_items"], meta["n_users"]
    cold = np.load(C.COLD_DIR / "cold_items.npy")
    desc = np.load(C.CONTENT_DIR / "item_desc_emb.npy").astype(np.float32)
    feats = {
        "item_num": torch.from_numpy(np.load(C.COLD_DIR / "item_num_cold.npy")),
        "item_cat": torch.from_numpy(np.load(C.COLD_DIR / "item_cat_cold.npy")),
        "item_tags": torch.full((n_items, 1), meta["tag_pad"], dtype=torch.int64),
        "user_num": torch.from_numpy(np.load(C.BASE_DIR / "user_num.npy")),
        "user_hist": torch.from_numpy(np.load(C.COLD_DIR / "user_hist_cold.npy").astype(np.int64)),
        "item_text": torch.from_numpy(desc),
        "id_lookup": torch.from_numpy(np.load(C.COLD_DIR / "id_lookup.npy")),
    }
    L = K = None
    if SEM_SRC:
        feats["sem_codes"] = torch.from_numpy(np.load(C.V6_DIR / f"sem_codes_{SEM_SRC}.npy").astype(np.int64))
        L, K = RQCFG[SEM_SRC]["L"], RQCFG[SEM_SRC]["K"]
    mc = copy.deepcopy(C.MODEL)
    model = TwoTowerV6(n_items, n_users, meta["n_tags"], meta["item_cat_cardinality"],
                       feats["item_num"].shape[1], feats["user_num"].shape[1], mc, id_mode=ID_MODE,
                       use_text=True, d_content=128, n_item_text=desc.shape[1], n_codes_L=L or 3, n_codes_K=K or 256).to(dev)
    model.attach_features(feats, dev)
    log(f"[{TAG}] model={MODEL}({ID_MODE},codes={SEM_SRC}) params={sum(p.numel() for p in model.parameters()):,}")

    pop = np.load(C.COLD_DIR / "popularity_cold.npy").astype(np.float64)
    log_q = torch.from_numpy(np.log((pop + 1) / (pop.sum() + len(pop)))).float().to(dev)
    pairs = torch.from_numpy(np.load(C.COLD_DIR / "train_pairs_cold.npy").astype(np.int64))
    ds = TensorDataset(pairs[:, 0], pairs[:, 1])
    g = torch.Generator().manual_seed(C.SEED)
    sm = torch.utils.data.RandomSampler(ds, num_samples=min(C.TRAIN.max_pairs_per_epoch, len(ds)), replacement=False, generator=g)
    ld = DataLoader(ds, batch_size=C.TRAIN.batch_size, sampler=sm, num_workers=C.TRAIN.num_workers, drop_last=True, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=C.TRAIN.lr, weight_decay=C.TRAIN.weight_decay)

    mask_off, mask_flat = csr_from_split(["train", "val"], n_users)
    test_off, test_flat = csr_from_split(["test"], n_users)
    eval_uids = np.where(np.diff(test_off) > 0)[0]
    best, patc = -1.0, 0; ckpt = C.COLD_DIR / f"ckpt_{TAG}.pt"

    @torch.no_grad()
    def quick(iv, sample=20000, k=100):
        model.eval(); uids = eval_uids
        if len(uids) > sample: uids = uids[np.random.RandomState(0).choice(len(uids), sample, replace=False)]
        tot = cnt = 0.0
        for s in range(0, len(uids), C.EVAL_USER_BATCH):
            ub = uids[s:s + C.EVAL_USER_BATCH]
            sc = model.encode_user(torch.from_numpy(ub).long().to(dev)) @ iv.t()
            cols = np.concatenate([mask_flat[mask_off[u]:mask_off[u+1]] for u in ub]); rows = np.repeat(np.arange(len(ub)), mask_off[ub+1]-mask_off[ub])
            if len(cols): sc[torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)] = NEG
            tk = torch.topk(sc, k, 1).indices.cpu().numpy()
            for i, u in enumerate(ub):
                t = test_flat[test_off[u]:test_off[u+1]]; tot += np.isin(t, tk[i]).sum()/len(t); cnt += 1
        return tot/max(cnt,1)

    for ep in range(1, C.TRAIN.epochs + 1):
        model.train(); tl = nb = 0
        for uid, tgt in ld:
            uid, tgt = uid.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            loss = info_nce_logq(model.encode_user(uid, drop_target=tgt), model.encode_item(tgt), tgt, log_q[tgt], C.TRAIN.temperature)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.TRAIN.grad_clip); opt.step()
            tl += loss.item(); nb += 1
        iv = model.encode_all_items(dev); r = quick(iv)
        msg = f"[{TAG}] ep{ep} loss={tl/nb:.4f} val_setR@100={r:.4f}"
        if r > best: best, patc = r, 0; torch.save({"state_dict": model.state_dict()}, ckpt); msg += " *"
        else: patc += 1
        log(msg)
        if patc >= C.TRAIN.early_stop_patience: log("early stop"); break
    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"]); model.eval()

    iv = model.encode_all_items(dev); Ks = C.EVAL_KS; maxK = max(Ks)
    acc = {gp: {**{k: 0 for k in Ks}, "n": 0} for gp in ["all", "item_warm", "item_cold"]}
    rl = np.full(n_items, maxK, np.int32); covered = np.zeros(n_items, bool)
    for s in range(0, len(eval_uids), C.EVAL_USER_BATCH):
        ub = eval_uids[s:s + C.EVAL_USER_BATCH]
        sc = model.encode_user(torch.from_numpy(ub).long().to(dev)) @ iv.t()
        cols = np.concatenate([mask_flat[mask_off[u]:mask_off[u+1]] for u in ub]); rows = np.repeat(np.arange(len(ub)), mask_off[ub+1]-mask_off[ub])
        if len(cols): sc[torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)] = NEG
        tk = torch.topk(sc, maxK, 1).indices.cpu().numpy()
        covered[tk.ravel()] = True
        for i, u in enumerate(ub):
            t = test_flat[test_off[u]:test_off[u+1]]; rl[tk[i]] = np.arange(maxK); ranks = rl[t]; rl[tk[i]] = maxK
            for j, it in enumerate(t):
                gp = "item_cold" if cold[it] else "item_warm"
                for k in Ks:
                    h = int(ranks[j] < k); acc["all"][k] += h; acc[gp][k] += h
                acc["all"]["n"] += 1; acc[gp]["n"] += 1
    out = {gp: {"n": a["n"], **{f"Recall@{k}": a[k]/max(a["n"],1) for k in Ks}} for gp, a in acc.items()}
    out["cold_item_coverage@200"] = float(covered[cold].mean())
    json.dump({"tag": TAG, "model": MODEL, "sem_src": SEM_SRC, "dnn": out}, open(C.COLD_DIR / f"eval_{TAG}.json", "w"), indent=2)
    print(f"\n===== v6-cold [{TAG}] model={MODEL} =====")
    for gp in ["all", "item_warm", "item_cold"]:
        x = out[gp]; print(f"  {gp:10s} n={x['n']:>7,} R@100={x['Recall@100']:.4f}")
    print(f"  ColdItemCoverage@200={out['cold_item_coverage@200']:.4f}")


if __name__ == "__main__":
    main()
