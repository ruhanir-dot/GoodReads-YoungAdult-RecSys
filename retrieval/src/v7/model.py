"""v6 two-tower: atomic / hybrid (atomic ⊕ semantic-code) item id + GRADED explicit negatives.

Item id-branch (`id_mode`): atomic -> nn.Embedding row; hybrid -> concat(atomic, sem_proj(Σ_l code_emb_l)).
The semantic-code SOURCE (content / collab / big) is chosen by which sem_codes array is attached and
the (n_codes_L, n_codes_K) passed in. No language branch (v6 uses language only as an eval slice).

Graded negatives: info_nce_logq takes a LIST of explicit negative pools (hard rating<=2, soft rating==3),
each appended as extra softmax-denominator columns with its own logit weight log(λ) and a self-collision
mask so a true positive is never penalized.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim, dropout):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]; d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def masked_mean(emb, ids, pad_id):
    vecs = emb(ids); mask = (ids != pad_id).float().unsqueeze(-1)
    return (vecs * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class TwoTowerV6(nn.Module):
    def __init__(self, n_items, n_users, n_tags, cat_card, n_item_num, n_user_num, cfg,
                 id_mode="atomic", use_text=True, d_content=128, n_item_text=384,
                 n_codes_L=3, n_codes_K=256):
        super().__init__()
        assert id_mode in ("atomic", "hybrid")
        self.n_items, self.n_tags = n_items, n_tags
        self.item_pad, self.tag_pad = n_items, n_tags
        self.id_mode, self.use_text = id_mode, use_text
        c = cfg
        self.item_id_emb = nn.Embedding(n_items + 1, c.d_id, padding_idx=n_items)
        self.tag_emb = nn.Embedding(n_tags + 1, c.d_tag, padding_idx=n_tags)
        lang_card, fmt_card, pop_card = cat_card
        self.lang_emb = nn.Embedding(lang_card, c.d_lang)
        self.fmt_emb = nn.Embedding(fmt_card, c.d_format)
        self.pop_emb = nn.Embedding(pop_card, c.d_poptier)
        self.text_proj = nn.Linear(n_item_text, d_content) if use_text else None
        if id_mode == "hybrid":
            self.code_emb = nn.ModuleList([nn.Embedding(n_codes_K, c.d_code) for _ in range(n_codes_L)])
            self.sem_proj = nn.Linear(c.d_code, c.d_id)
        else:
            self.code_emb = None

        id_dim = c.d_id * (2 if id_mode == "hybrid" else 1)
        item_in = id_dim + n_item_num + c.d_lang + c.d_format + c.d_poptier + c.d_tag + (d_content if use_text else 0)
        user_in = c.d_id + n_user_num
        self.item_mlp = _mlp(item_in, c.mlp_hidden, c.d_out, c.dropout)
        self.user_mlp = _mlp(user_in, c.mlp_hidden, c.d_out, c.dropout)
        for e in (self.item_id_emb, self.tag_emb, self.lang_emb, self.fmt_emb, self.pop_emb):
            nn.init.normal_(e.weight, std=0.05)
        if self.code_emb is not None:
            for e in self.code_emb: nn.init.normal_(e.weight, std=0.05)
        with torch.no_grad():
            self.item_id_emb.weight[n_items].zero_(); self.tag_emb.weight[n_tags].zero_()
        self.item_num = self.item_cat = self.item_tags = None
        self.user_num = self.user_hist = self.item_text = self.sem_codes = self.id_lookup = None

    def attach_features(self, feats, device):
        self.item_num = feats["item_num"].to(device)
        self.item_cat = feats["item_cat"].to(device)
        self.item_tags = feats["item_tags"].to(device)
        self.user_num = feats["user_num"].to(device)
        self.user_hist = feats["user_hist"].to(device)
        if feats.get("item_text") is not None: self.item_text = feats["item_text"].to(device)
        if feats.get("sem_codes") is not None: self.sem_codes = feats["sem_codes"].to(device)
        if feats.get("id_lookup") is not None: self.id_lookup = feats["id_lookup"].to(device)

    def _sem(self, iid):
        codes = self.sem_codes[iid]
        s = 0.0
        for l, e in enumerate(self.code_emb): s = s + e(codes[:, l])
        return self.sem_proj(s)

    def encode_item(self, iid):
        cat = self.item_cat[iid]
        parts = [self.item_num[iid], self.lang_emb(cat[:, 0]), self.fmt_emb(cat[:, 1]), self.pop_emb(cat[:, 2]),
                 masked_mean(self.tag_emb, self.item_tags[iid], self.tag_pad)]
        if self.id_mode == "atomic":
            lid = self.id_lookup[iid] if self.id_lookup is not None else iid
            parts.insert(0, self.item_id_emb(lid))
        else:  # hybrid
            lid = self.id_lookup[iid] if self.id_lookup is not None else iid
            parts.insert(0, torch.cat([self.item_id_emb(lid), self._sem(iid)], dim=1))
        if self.use_text: parts.append(self.text_proj(self.item_text[iid]))
        return F.normalize(self.item_mlp(torch.cat(parts, dim=1)), dim=1)

    def encode_user(self, uid, drop_target=None):
        hist = self.user_hist[uid]
        if drop_target is not None:
            hist = hist.masked_fill(hist == drop_target.unsqueeze(1), self.item_pad)
        parts = [masked_mean(self.item_id_emb, hist, self.item_pad), self.user_num[uid]]
        return F.normalize(self.user_mlp(torch.cat(parts, dim=1)), dim=1)

    @torch.no_grad()
    def encode_all_items(self, device, batch=8192):
        self.eval(); d = self.item_mlp[-1].out_features
        out = torch.empty((self.n_items, d), device=device)
        for s in range(0, self.n_items, batch):
            e = min(s + batch, self.n_items)
            out[s:e] = self.encode_item(torch.arange(s, e, device=device))
        return out


def info_nce_logq(user_vec, item_vec, target_ids, log_q, temperature, negs=None):
    """In-batch sampled softmax + logQ + accidental-hit mask, with optional GRADED explicit negatives.
    negs: list of (neg_vec[B,M,d], neg_mask[B,M] bool, neg_bias float=log(lambda), neg_ids[B,M]).
    Each pool is appended as extra denominator columns; self-collision (==positive) masked to -inf."""
    logits = (user_vec @ item_vec.t()) / temperature
    logits = logits - log_q.unsqueeze(0)
    same = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)
    eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    cols = [logits]
    if negs:
        for neg_vec, neg_mask, neg_bias, neg_ids in negs:
            extra = torch.einsum("bd,bmd->bm", user_vec, neg_vec) / temperature + neg_bias
            extra = extra.masked_fill(~neg_mask, float("-inf"))
            extra = extra.masked_fill(neg_ids == target_ids.unsqueeze(1), float("-inf"))
            cols.append(extra)
    logits = torch.cat(cols, dim=1)
    labels = torch.arange(user_vec.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
