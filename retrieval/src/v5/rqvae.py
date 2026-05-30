"""Lever A: Residual-Quantized VAE over item content -> discrete semantic codes (TIGER-style).

Train an RQ-VAE on item_content_emb (bge(title+desc+tags)). Each item gets L discrete codes
(one per residual codebook). In the recommender (model.py) the item-ID representation becomes the
SUM of L learned per-level code embeddings -> content-similar books share codes (semantic
structure), and a brand-new (cold) book is quantized into already-trained codes (its vector is
NOT zero, unlike an untrained atomic-ID row).

This file: (1) RQVAE nn.Module (encoder -> L-level residual VQ -> decoder),
(2) a train CLI that fits it on data/v5/item_content_emb.npy and writes:
  sem_codes.npy   [n_items, L] int   the discrete code per item per level (THE artifact model.py uses)
  rqvae.pt        full module state + config (for inspection / re-quantizing new items)
  rqvae_report.json  recon error, per-codebook usage/perplexity (codebook-collapse check)

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=<idle> python rqvae.py
"""
from __future__ import annotations
import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


class VQLayer(nn.Module):
    """One residual codebook with EMA updates, straight-through estimator, data-dependent init,
    and dead-code revival (resets rarely-used codes to random current inputs -> prevents the
    single-code collapse that residual VQ is prone to at the first level)."""
    def __init__(self, K, dim, ema=0.99, eps=1e-5, revive_thr=1.0):
        super().__init__()
        self.K, self.dim, self.ema, self.eps, self.revive_thr = K, dim, ema, eps, revive_thr
        self.register_buffer("codebook", torch.randn(K, dim) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("ema_w", self.codebook.clone())
        self.register_buffer("initted", torch.zeros(1, dtype=torch.bool))

    def _data_init(self, x):                               # seed codebook from real inputs
        n = x.size(0)
        idx = torch.randint(0, n, (self.K,), device=x.device) if n >= 1 else None
        if idx is not None:
            self.codebook.copy_(x[idx].detach())
            self.ema_w.copy_(self.codebook)
            self.cluster_size.fill_(1.0)
            self.initted.fill_(True)

    def forward(self, x):                                  # x: [B, dim] (a residual)
        if self.training and not bool(self.initted): self._data_init(x)
        d = (x.pow(2).sum(1, keepdim=True) - 2 * x @ self.codebook.t()
             + self.codebook.pow(2).sum(1))                # [B, K] squared dists
        idx = d.argmin(1)                                  # [B]
        q = self.codebook[idx]                             # [B, dim]
        if self.training and self.ema is not None:
            with torch.no_grad():
                oh = F.one_hot(idx, self.K).type_as(x)     # [B, K]
                n = oh.sum(0)                              # counts
                self.cluster_size.mul_(self.ema).add_(n, alpha=1 - self.ema)
                dw = oh.t() @ x                            # [K, dim]
                self.ema_w.mul_(self.ema).add_(dw, alpha=1 - self.ema)
                tot = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (tot + self.K * self.eps) * tot
                self.codebook.copy_(self.ema_w / cs.unsqueeze(1))
                dead = self.cluster_size < self.revive_thr  # revive dead codes from current batch
                nd = int(dead.sum())
                if nd > 0 and x.size(0) > 0:
                    r = torch.randint(0, x.size(0), (nd,), device=x.device)
                    self.codebook[dead] = x[r].detach()
                    self.ema_w[dead] = x[r].detach()
                    self.cluster_size[dead] = 1.0
        q_st = x + (q - x).detach()                        # straight-through
        commit = F.mse_loss(q.detach(), x)
        return q_st, q, idx, commit


class RQVAE(nn.Module):
    def __init__(self, in_dim, latent, L, K, ema=0.99):
        super().__init__()
        self.L = L
        h = max(latent * 2, 256)
        self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h), nn.ReLU(), nn.Linear(h, in_dim))
        self.vqs = nn.ModuleList([VQLayer(K, latent, ema) for _ in range(L)])

    def quantize(self, z):
        res, q_sum, codes, commit = z, 0.0, [], 0.0
        for vq in self.vqs:
            q_st, q, idx, c = vq(res)
            q_sum = q_sum + q_st
            res = res - q.detach()                          # quantize the residual next level
            codes.append(idx); commit = commit + c
        return q_sum, torch.stack(codes, 1), commit         # [B,latent], [B,L], scalar

    def forward(self, x):
        z = self.enc(x)
        q, codes, commit = self.quantize(z)
        xr = self.dec(q)
        recon = F.mse_loss(xr, x)
        return recon, commit, codes


def main():
    C.ensure_dirs()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    X = np.load(C.V5_DIR / "item_content_emb.npy").astype(np.float32)
    n, d = X.shape
    log(f"item content {X.shape}; RQ-VAE L={C.RQ_L} K={C.RQ_K} latent={C.RQ_DIM}")
    Xt = torch.from_numpy(X).to(dev)
    model = RQVAE(d, C.RQ_DIM, C.RQ_L, C.RQ_K, C.RQ_EMA).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=C.RQ_LR)

    idx_all = np.arange(n)
    for ep in range(1, C.RQ_EPOCHS + 1):
        model.train(); np.random.shuffle(idx_all); tot = nb = 0.0
        for s in range(0, n, C.RQ_BATCH):
            b = torch.from_numpy(idx_all[s:s + C.RQ_BATCH]).to(dev)
            recon, commit, _ = model(Xt[b])
            loss = recon + C.RQ_BETA * commit
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(recon); nb += 1
        if ep % 10 == 0 or ep == 1:
            log(f"ep{ep} recon_mse={tot/nb:.5f}")

    # final codes for ALL items (incl. would-be-cold) -> the artifact
    model.eval()
    with torch.no_grad():
        codes = []
        for s in range(0, n, 8192):
            _, _, c = model(Xt[s:s + 8192]); codes.append(c.cpu().numpy())  # forward -> (recon, commit, codes)
        codes = np.concatenate(codes).astype(np.int64)      # [n, L]
    np.save(C.V5_DIR / "sem_codes.npy", codes)
    torch.save({"state_dict": model.state_dict(),
                "cfg": {"in_dim": d, "latent": C.RQ_DIM, "L": C.RQ_L, "K": C.RQ_K}},
               C.V5_DIR / "rqvae.pt")

    # codebook usage / perplexity (collapse check) + recon on full set
    with torch.no_grad():
        recon_full = float(F.mse_loss(model.dec(model.enc(Xt)), Xt))  # approx (pre-quant) lower bound
    usage = []
    for l in range(C.RQ_L):
        u, cnt = np.unique(codes[:, l], return_counts=True)
        p = cnt / cnt.sum(); ppl = float(np.exp(-(p * np.log(p)).sum()))
        usage.append({"level": l, "used": int(len(u)), "of": C.RQ_K, "perplexity": round(ppl, 1)})
    uniq = len({tuple(r) for r in codes.tolist()})
    rep = {"n_items": n, "L": C.RQ_L, "K": C.RQ_K, "latent": C.RQ_DIM, "epochs": C.RQ_EPOCHS,
           "final_recon_mse": round(tot / nb, 5), "codebook_usage": usage,
           "unique_code_tuples": uniq, "collision_rate": round(1 - uniq / n, 4)}
    json.dump(rep, open(C.V5_DIR / "rqvae_report.json", "w"), indent=2)
    log(f"saved sem_codes.npy [{n},{C.RQ_L}] | usage={[u['used'] for u in usage]}/{C.RQ_K} "
        f"ppl={[u['perplexity'] for u in usage]} | unique tuples {uniq:,}/{n:,} (collision {rep['collision_rate']:.1%})")
    log("rqvae DONE")


if __name__ == "__main__":
    main()
