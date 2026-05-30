"""Lever C data: normalize v3 book language codes, build a shared language vocab, and the
per-user language distribution. Pure CPU. Outputs -> data/v5/.

  lang_vocab.json    {code: id} for top-K user languages (+ 'other' bucket = K)
  book_lang.npy      [n_items] int  vocab id of each book's (normalized) language
  user_lang_w.npy    [n_users, K+1] float32  user's normalized language weights over the vocab
                     (row all-zero = user has no v3 language info -> uniform penalty = no-op at scoring)

  python build_language.py
"""
from __future__ import annotations
import json
import numpy as np
import config as C
import dataset as D

# ISO 639-2/B (3-letter) + region -> 639-1 (2-letter). Covers the 973 non-2-letter book codes.
MAP3 = {
    "eng": "en", "spa": "es", "deu": "de", "ger": "de", "ind": "id", "jpn": "ja", "fil": "tl",
    "tgl": "tl", "ara": "ar", "swe": "sv", "ita": "it", "pol": "pl", "tam": "ta", "est": "et",
    "cat": "ca", "srp": "sr", "sun": "su", "fin": "fi", "ben": "bn", "rus": "ru", "por": "pt",
    "nor": "no", "tha": "th", "dan": "da", "fre": "fr", "fra": "fr", "nld": "nl", "dut": "nl",
    "zho": "zh", "chi": "zh", "kor": "ko", "vie": "vi", "tur": "tr", "ell": "el", "gre": "el",
    "heb": "he", "hun": "hu", "ces": "cs", "cze": "cs", "ron": "ro", "rum": "ro", "ukr": "uk",
    "bul": "bg", "hrv": "hr", "slk": "sk", "slo": "sk", "slv": "sl", "lit": "lt", "lav": "lv",
    "hin": "hi", "fas": "fa", "per": "fa", "msa": "ms", "may": "ms", "afr": "af",
}

def norm_lang(code):
    if not code: return None
    c = str(code).strip().lower().split("-")[0].split("_")[0]
    c = MAP3.get(c, c)
    return c if len(c) == 2 and c.isalpha() else "other"


def main():
    C.ensure_dirs()
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]

    # --- read book languages (aligned by iid) ---
    book_code = [None] * n_items
    for l in open(C.BOOK_PROFILES):
        try: r = json.loads(l)
        except Exception: continue
        i = r.get("iid")
        if isinstance(i, int) and 0 <= i < n_items:
            book_code[i] = norm_lang(r.get("language"))

    # --- read user language dicts (aligned by uid) + accumulate global weight for vocab ranking ---
    user_dicts = [None] * n_users
    weight = {}
    for l in open(C.USER_PROFILES):
        try: r = json.loads(l)
        except Exception: continue
        u = r.get("uid"); lang = r.get("language")
        if not (isinstance(u, int) and 0 <= u < n_users and isinstance(lang, dict)): continue
        d = {}
        for code, w in lang.items():
            c = norm_lang(code)
            if c is None: continue
            d[c] = d.get(c, 0.0) + float(w)
            weight[c] = weight.get(c, 0.0) + float(w)
        if d: user_dicts[u] = d

    # --- vocab = top-K languages by user weight (exclude 'other'); id K = 'other' bucket ---
    ranked = [c for c, _ in sorted(weight.items(), key=lambda kv: -kv[1]) if c != "other"][:C.LANG_TOPK]
    vocab = {c: i for i, c in enumerate(ranked)}
    OTHER = len(ranked)                       # 'other' bucket id
    vocab_out = dict(vocab); vocab_out["other"] = OTHER
    json.dump({"vocab": vocab_out, "K": OTHER, "ranked": ranked,
               "global_weight": {c: round(weight[c], 1) for c in ranked}},
              open(C.V5_DIR / "lang_vocab.json", "w"), indent=2)

    def vid(code): return vocab.get(code, OTHER) if code else OTHER

    book_lang = np.array([vid(book_code[i]) for i in range(n_items)], dtype=np.int64)
    np.save(C.V5_DIR / "book_lang.npy", book_lang)

    W = np.zeros((n_users, OTHER + 1), np.float32)
    nz = 0
    for u in range(n_users):
        d = user_dicts[u]
        if not d: continue
        for c, w in d.items():
            W[u, vid(c)] += w
        s = W[u].sum()
        if s > 0: W[u] /= s; nz += 1
    np.save(C.V5_DIR / "user_lang_w.npy", W)

    # stats
    import collections
    bl = collections.Counter(book_lang.tolist())
    print(f"lang vocab (top-{C.LANG_TOPK}): {ranked}")
    print(f"book_lang: {n_items:,} items; top buckets: "
          + ", ".join(f"{[k for k,v in vocab_out.items() if v==i][0]}={bl[i]}" for i in sorted(bl, key=lambda i:-bl[i])[:8]))
    print(f"user_lang_w: {nz:,}/{n_users:,} users have language info (rest all-zero = no constraint)")
    print(f"wrote lang_vocab.json / book_lang.npy / user_lang_w.npy to {C.V5_DIR}")


if __name__ == "__main__":
    main()
