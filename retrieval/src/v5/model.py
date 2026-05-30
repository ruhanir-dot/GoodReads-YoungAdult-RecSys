"""v5 two-tower with the three orthogonal levers wired in (framework unchanged from v1/v4).

  Lever A (semantic ID): item id-branch = `id_mode`
      atomic   -> nn.Embedding(n_items) row            (= v4; cold item row untrained -> ~0)
      semantic -> sem_proj( sum_l code_emb_l[code_l] ) (RQ-VAE codes; cold item gets REAL vector)
      hybrid   -> concat(atomic, semantic)             (the "atomic ⊕ semantic")
  Lever B (dislike hard-negatives): handled in the loss `info_nce_logq` via optional extra
      negative columns (the user's rating<=2 items) -> see driver.
  Lever C (language): C1 = user-language feature branch on the user tower (`use_ulang`);
      C2 = retrieval-time language-match penalty applied at SCORING by the driver (not here).

Item tower : [ id-branch | item_num | lang/fmt/pop emb | tag_pool(inert) | content_proj ] -> MLP -> L2
User tower : [ hist_pool | user_num | (ulang_proj) ]                                       -> MLP -> L2
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim, dropout):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def masked_mean(emb: nn.Embedding, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    vecs = emb(ids)
    mask = (ids != pad_id).float().unsqueeze(-1)
    summed = (vecs * mask).sum(1)
    cnt = mask.sum(1).clamp(min=1.0)
    return summed / cnt


class TwoTowerV5(nn.Module):
    def __init__(self, n_items, n_users, n_tags, cat_card, n_item_num, n_user_num, cfg,
                 id_mode="atomic", use_text=True, d_content=128, n_item_text=384,
                 n_codes_L=3, n_codes_K=256, use_ulang=False, n_ulang=21):
        super().__init__()
        assert id_mode in ("atomic", "semantic", "hybrid")
        self.n_items, self.n_tags = n_items, n_tags
        self.item_pad, self.tag_pad = n_items, n_tags
        self.id_mode = id_mode
        self.use_text = use_text
        self.use_ulang = use_ulang
        c = cfg

        # atomic id + shared tag emb (tag branch inert in v5, kept for parity with v4 item_in)
        self.item_id_emb = nn.Embedding(n_items + 1, c.d_id, padding_idx=n_items)
        self.tag_emb = nn.Embedding(n_tags + 1, c.d_tag, padding_idx=n_tags)
        lang_card, fmt_card, pop_card = cat_card
        self.lang_emb = nn.Embedding(lang_card, c.d_lang)
        self.fmt_emb = nn.Embedding(fmt_card, c.d_format)
        self.pop_emb = nn.Embedding(pop_card, c.d_poptier)
        self.text_proj = nn.Linear(n_item_text, d_content) if use_text else None

        # lever A: per-level learned code embeddings + projection to d_id
        if id_mode in ("semantic", "hybrid"):
            self.code_emb = nn.ModuleList([nn.Embedding(n_codes_K, c.d_code) for _ in range(n_codes_L)])
            self.sem_proj = nn.Linear(c.d_code, c.d_id)
        else:
            self.code_emb = None
        # lever C1: user-language feature
        self.ulang_proj = nn.Linear(n_ulang, c.d_ulang) if use_ulang else None

        id_dim = c.d_id * (2 if id_mode == "hybrid" else 1)
        item_in = id_dim + n_item_num + c.d_lang + c.d_format + c.d_poptier + c.d_tag
        if use_text:
            item_in += d_content
        user_in = c.d_id + n_user_num + (c.d_ulang if use_ulang else 0)
        self.item_mlp = _mlp(item_in, c.mlp_hidden, c.d_out, c.dropout)
        self.user_mlp = _mlp(user_in, c.mlp_hidden, c.d_out, c.dropout)

        for emb in (self.item_id_emb, self.tag_emb, self.lang_emb, self.fmt_emb, self.pop_emb):
            nn.init.normal_(emb.weight, std=0.05)
        if self.code_emb is not None:
            for e in self.code_emb:
                nn.init.normal_(e.weight, std=0.05)
        with torch.no_grad():
            self.item_id_emb.weight[n_items].zero_()
            self.tag_emb.weight[n_tags].zero_()

        # side tensors (attach_features)
        self.item_num = self.item_cat = self.item_tags = None
        self.user_num = self.user_hist = None
        self.item_text = self.sem_codes = self.id_lookup = self.user_lang = None

    def attach_features(self, feats: dict, device):
        self.item_num = feats["item_num"].to(device)
        self.item_cat = feats["item_cat"].to(device)
        self.item_tags = feats["item_tags"].to(device)
        self.user_num = feats["user_num"].to(device)
        self.user_hist = feats["user_hist"].to(device)
        if feats.get("item_text") is not None:
            self.item_text = feats["item_text"].to(device)
        if feats.get("sem_codes") is not None:
            self.sem_codes = feats["sem_codes"].to(device)        # [n_items, L] int
        if feats.get("id_lookup") is not None:
            self.id_lookup = feats["id_lookup"].to(device)        # cold-start: warm->iid, cold->pad
        if feats.get("user_lang") is not None:
            self.user_lang = feats["user_lang"].to(device)        # [n_users, n_ulang] float

    # ------------------------------------------------------------------ towers
    def _sem(self, iid):
        codes = self.sem_codes[iid]                               # [B, L]
        s = 0.0
        for l, e in enumerate(self.code_emb):
            s = s + e(codes[:, l])
        return self.sem_proj(s)                                   # [B, d_id]

    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        cat = self.item_cat[iid]
        parts = [self.item_num[iid],
                 self.lang_emb(cat[:, 0]), self.fmt_emb(cat[:, 1]), self.pop_emb(cat[:, 2]),
                 masked_mean(self.tag_emb, self.item_tags[iid], self.tag_pad)]
        # id-branch
        if self.id_mode == "atomic":
            lid = self.id_lookup[iid] if self.id_lookup is not None else iid
            parts.insert(0, self.item_id_emb(lid))
        elif self.id_mode == "semantic":
            parts.insert(0, self._sem(iid))                       # cold items get a real vector
        else:  # hybrid
            lid = self.id_lookup[iid] if self.id_lookup is not None else iid
            parts.insert(0, torch.cat([self.item_id_emb(lid), self._sem(iid)], dim=1))
        if self.use_text:
            parts.append(self.text_proj(self.item_text[iid]))
        return F.normalize(self.item_mlp(torch.cat(parts, dim=1)), dim=1)

    def encode_user(self, uid: torch.Tensor, drop_target: torch.Tensor | None = None) -> torch.Tensor:
        hist = self.user_hist[uid]
        if drop_target is not None:
            hist = hist.masked_fill(hist == drop_target.unsqueeze(1), self.item_pad)
        parts = [masked_mean(self.item_id_emb, hist, self.item_pad), self.user_num[uid]]
        if self.use_ulang:
            parts.append(self.ulang_proj(self.user_lang[uid]))
        return F.normalize(self.user_mlp(torch.cat(parts, dim=1)), dim=1)

    @torch.no_grad()
    def encode_all_items(self, device, batch=8192) -> torch.Tensor:
        self.eval()
        d = self.item_mlp[-1].out_features
        out = torch.empty((self.n_items, d), device=device)
        for s in range(0, self.n_items, batch):
            e = min(s + batch, self.n_items)
            out[s:e] = self.encode_item(torch.arange(s, e, device=device))
        return out


def info_nce_logq(user_vec, item_vec, target_ids, log_q, temperature,
                  neg_vec=None, neg_mask=None, neg_bias=0.0, neg_ids=None):
    """In-batch sampled softmax + logQ + accidental-hit masking, with OPTIONAL extra explicit
    negatives (lever B). neg_vec: [B, M, d] encoded dislike items; neg_mask: [B, M] bool valid;
    neg_bias: log(lambda) added to dislike logits to up/down-weight them."""
    logits = (user_vec @ item_vec.t()) / temperature            # [B,B]
    logits = logits - log_q.unsqueeze(0)
    same = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)
    eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    if neg_vec is not None:
        # per-row dot of user with its own M dislike items -> [B, M]
        extra = torch.einsum("bd,bmd->bm", user_vec, neg_vec) / temperature + neg_bias
        extra = extra.masked_fill(~neg_mask, float("-inf"))
        if neg_ids is not None:                                  # safety: never penalize the positive
            extra = extra.masked_fill(neg_ids == target_ids.unsqueeze(1), float("-inf"))
        logits = torch.cat([logits, extra], dim=1)              # [B, B+M]; labels still on diag
    labels = torch.arange(user_vec.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
