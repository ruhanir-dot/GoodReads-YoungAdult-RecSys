"""Two-tower (dual-encoder) retrieval model (shared across versions; v4 is the current driver).

Constructor flags select the variant: `item_mode` (id | content | hybrid), `use_text`
(frozen-bge content branch + learned projection), `use_user_content`/`use_avoid` (v3 user
taste branch; off in v4). Shared embeddings:
  * item_id_emb : Embedding(n_items+1, d_id)  pad row = n_items  (item-tower ID input AND
                  the basis the user tower pools its history over)
  * tag_emb     : Embedding(n_tags+1,  d_tag) pad row = n_tags   (item/user tag bags;
                  DEPRECATED tag profiling -> fed all-pad in v4, so this branch is inert)

Item tower : [ (id_emb) | numeric | lang/fmt/poptier emb | tag_pool | (content_proj) ] -> MLP -> L2
User tower : [ hist_pool | numeric | liked/disliked pool | (user_content_proj) ]       -> MLP -> L2

All side-feature tensors are held as (non-parameter) device attributes so a batch is
just an index gather on the GPU; nothing is recomputed per sample.
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
    """Mean of embeddings over non-pad ids. ids: [B,L] -> [B,d]. Empty -> 0 vector."""
    vecs = emb(ids)                                   # [B,L,d] (pad row is 0 via padding_idx)
    mask = (ids != pad_id).float().unsqueeze(-1)      # [B,L,1]
    summed = (vecs * mask).sum(1)
    cnt = mask.sum(1).clamp(min=1.0)
    return summed / cnt


class TwoTower(nn.Module):
    def __init__(self, n_items, n_users, n_tags, cat_card, n_item_num, n_user_num, cfg,
                 item_mode="id", use_text=False, d_content=128, n_item_text=384,
                 use_user_content=False, use_avoid=False, d_ucontent=128, n_user_text=384):
        super().__init__()
        self.n_items, self.n_tags = n_items, n_tags
        self.item_pad, self.tag_pad = n_items, n_tags
        self.item_mode = item_mode                  # id | content | hybrid
        self.use_text = use_text or item_mode in ("content", "hybrid")
        self.use_user_content = use_user_content    # bge user-taste branch on the user tower
        self.use_avoid = use_avoid
        c = cfg

        self.item_id_emb = nn.Embedding(n_items + 1, c.d_id, padding_idx=n_items)
        self.tag_emb = nn.Embedding(n_tags + 1, c.d_tag, padding_idx=n_tags)
        lang_card, fmt_card, pop_card = cat_card
        self.lang_emb = nn.Embedding(lang_card, c.d_lang)
        self.fmt_emb = nn.Embedding(fmt_card, c.d_format)
        self.pop_emb = nn.Embedding(pop_card, c.d_poptier)
        self.text_proj = nn.Linear(n_item_text, d_content) if self.use_text else None
        self.uc_proj = nn.Linear(n_user_text, d_ucontent) if use_user_content else None
        self.avoid_proj = nn.Linear(n_user_text, d_ucontent) if (use_user_content and use_avoid) else None

        # item-tower input = always [num | lang | fmt | pop | tag] (+ id) (+ content)
        item_in = n_item_num + c.d_lang + c.d_format + c.d_poptier + c.d_tag
        if item_mode in ("id", "hybrid"):
            item_in += c.d_id
        if item_mode in ("content", "hybrid"):
            item_in += d_content
        user_in = c.d_id + n_user_num + c.d_tag + c.d_tag   # hist | num | liked | disliked
        if use_user_content:
            user_in += d_ucontent
        if use_user_content and use_avoid:
            user_in += d_ucontent
        self.item_mlp = _mlp(item_in, c.mlp_hidden, c.d_out, c.dropout)
        self.user_mlp = _mlp(user_in, c.mlp_hidden, c.d_out, c.dropout)

        for emb in (self.item_id_emb, self.tag_emb, self.lang_emb, self.fmt_emb, self.pop_emb):
            nn.init.normal_(emb.weight, std=0.05)
        with torch.no_grad():               # keep pad rows exactly zero
            self.item_id_emb.weight[n_items].zero_()
            self.tag_emb.weight[n_tags].zero_()

        # side-feature tensors (set via attach_features); not parameters
        self.item_num = self.item_cat = self.item_tags = None
        self.user_num = self.user_liked = self.user_disliked = self.user_hist = None
        self.item_text = None        # [n_items, 384] bge content (content/hybrid modes)
        self.id_lookup = None        # [n_items] warm->iid, cold->pad (cold-start holdout)
        self.user_content = None     # [n_users, 384] bge user-taste (likes)
        self.user_avoid = None       # [n_users, 384] bge user-dislikes

    # ------------------------------------------------------------------ features
    def attach_features(self, feats: dict, device):
        self.item_num = feats["item_num"].to(device)
        self.item_cat = feats["item_cat"].to(device)
        self.item_tags = feats["item_tags"].to(device)        # [n_items, Lmax] padded
        self.user_num = feats["user_num"].to(device)
        self.user_liked = feats["user_liked"].to(device)
        self.user_disliked = feats["user_disliked"].to(device)
        self.user_hist = feats["user_hist"].to(device)        # [n_users, H] padded (pad=n_items)
        if feats.get("item_text") is not None:
            self.item_text = feats["item_text"].to(device)
        if feats.get("id_lookup") is not None:
            self.id_lookup = feats["id_lookup"].to(device)
        if feats.get("user_content") is not None:
            self.user_content = feats["user_content"].to(device)
        if feats.get("user_avoid") is not None:
            self.user_avoid = feats["user_avoid"].to(device)

    # ------------------------------------------------------------------ towers
    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        cat = self.item_cat[iid]
        parts = [self.item_num[iid],
                 self.lang_emb(cat[:, 0]), self.fmt_emb(cat[:, 1]), self.pop_emb(cat[:, 2]),
                 masked_mean(self.tag_emb, self.item_tags[iid], self.tag_pad)]
        if self.item_mode in ("id", "hybrid"):
            lid = self.id_lookup[iid] if self.id_lookup is not None else iid  # cold -> pad row
            parts.insert(0, self.item_id_emb(lid))
        if self.item_mode in ("content", "hybrid"):
            parts.append(self.text_proj(self.item_text[iid]))
        return F.normalize(self.item_mlp(torch.cat(parts, dim=1)), dim=1)

    def encode_user(self, uid: torch.Tensor, drop_target: torch.Tensor | None = None,
                    uc_override: torch.Tensor | None = None) -> torch.Tensor:
        hist = self.user_hist[uid]                            # [B,H]
        if drop_target is not None:                           # mask the training target out of history
            hist = hist.masked_fill(hist == drop_target.unsqueeze(1), self.item_pad)
        parts = [
            masked_mean(self.item_id_emb, hist, self.item_pad),
            self.user_num[uid],
            masked_mean(self.tag_emb, self.user_liked[uid], self.tag_pad),
            masked_mean(self.tag_emb, self.user_disliked[uid], self.tag_pad),
        ]
        if self.use_user_content:
            uc = uc_override if uc_override is not None else self.user_content[uid]  # LOO content at train
            parts.append(self.uc_proj(uc))
            if self.use_avoid:
                parts.append(self.avoid_proj(self.user_avoid[uid]))
        return F.normalize(self.user_mlp(torch.cat(parts, dim=1)), dim=1)

    @torch.no_grad()
    def encode_all_items(self, device, batch=8192) -> torch.Tensor:
        self.eval()
        d = self.item_mlp[-1].out_features
        out = torch.empty((self.n_items, d), device=device)
        for s in range(0, self.n_items, batch):
            e = min(s + batch, self.n_items)
            idx = torch.arange(s, e, device=device)
            out[s:e] = self.encode_item(idx)
        return out


def info_nce_logq(user_vec, item_vec, target_ids, log_q, temperature):
    """In-batch sampled softmax with logQ correction + accidental-hit masking.

    user_vec/item_vec: [B,d] L2-normalized. target_ids: [B] item ids of the positives.
    log_q: [B] = log P(item sampled as in-batch negative) ~ log(item frequency).
    Returns scalar loss.
    """
    logits = (user_vec @ item_vec.t()) / temperature          # [B,B], diag = positives
    logits = logits - log_q.unsqueeze(0)                      # subtract log Q over columns
    # accidental hits: same item appearing as another row's positive is not a true negative
    same = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)
    eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
