"""v1 data preparation for the two-tower retrieval baseline.

Reads directly from ingest parquet + profiling jsonl/vocab (self-contained, no
fusion bundle). Produces aligned numpy/npz arrays under retrieval/data/v1/.

Outputs (all int keys, row i == iid/uid i):
  meta.json            shapes, dims, counts, scaler params, vocabs
  split.parquet        uid, iid, ts, rating, split  (positives only)
  popularity.npy       [n_items] train-pool positive counts (logQ + pop baseline)
  item_num.npy         [n_items, *] standardized numeric + flags
  item_cat.npy         [n_items, 3] language_idx, format_idx, pop_tier
  item_tags.npz        flat + offsets  (tags+style+cw -> 4000 vocab, EmbeddingBag)
  user_num.npy         [n_users, *] standardized behaviour stats + flags
  user_liked.npz       flat + offsets
  user_disliked.npz    flat + offsets
  user_hist.npy        [n_users, HIST_LEN] last-N train items (pad = n_items)
  user_hist_len.npy    [n_users]
  train_pairs.npy      [P, 2] (uid, target_iid)   users with >=2 train items
  val.npy / test.npy   [*, 2] (uid, target_iid)   eval users (n_pos>=3)

Run:  CUDA_VISIBLE_DEVICES="" python prepare_data.py    (pure CPU/duckdb)
"""
from __future__ import annotations

import json
import time

import duckdb
import numpy as np
import config as C  # noqa: E402

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- tag bags (DEPRECATED)
# The old tag/vocab LLM profiling was removed (replaced by the text-based profile_bge pipeline;
# v3/v4 showed tag features don't help). We emit EMPTY (all-pad) tag bags so the model's
# tag-embedding branch stays shape-compatible but contributes a zero vector.
def empty_tags(n):
    """n empty bags -> (flat=[TAG_PAD]*n, offsets=arange(n)); each row is one pad id."""
    return np.full(n, C.TAG_PAD, dtype=np.int64), np.arange(n, dtype=np.int64)


# --------------------------------------------------------------------------- main
def main():
    C.ensure_dirs()
    con = duckdb.connect()
    con.execute("PRAGMA threads=16")

    pq = lambda name: f"read_parquet('{C.PARQUET}/{name}.parquet')"
    idm = lambda name: f"read_parquet('{C.ID_MAPS}/{name}.parquet')"

    n_items = con.execute(f"SELECT max(iid)+1 FROM {idm('book_iid_map')}").fetchone()[0]
    n_users = con.execute(f"SELECT max(uid)+1 FROM {idm('uid_map')}").fetchone()[0]
    C.ITEM_PAD = n_items
    log(f"n_items={n_items:,}  n_users={n_users:,}")

    # ---- 1. positives at work level, deduped (uid,iid) ----
    log("building positives (work-level dedup)...")
    con.execute(f"""
        CREATE TABLE pos AS
        SELECT m.iid AS iid, u.uid AS uid,
               max(i.rating) AS rating,
               epoch(max(i.date_added))::BIGINT AS ts   -- most-recent interaction => correct leave-last-out recency
        FROM {pq('interactions_core')} i
        JOIN {idm('book_iid_map')} m ON i.book_id = m.book_id
        JOIN {idm('uid_map')} u      ON i.user_id = u.user_id
        WHERE i.is_read OR i.rating >= 4
        GROUP BY m.iid, u.uid
    """)
    n_pos = con.execute("SELECT count(*) FROM pos").fetchone()[0]
    log(f"  positive (uid,iid) pairs = {n_pos:,}")

    # ---- 2. Leave-Last-Out split ----
    log("computing leave-last-out split...")
    con.execute("""
        CREATE TABLE ranked AS
        SELECT *,
               row_number() OVER (PARTITION BY uid ORDER BY ts DESC, iid DESC) AS rn,
               count(*)     OVER (PARTITION BY uid) AS np
        FROM pos
    """)
    con.execute(f"""
        CREATE TABLE split AS
        SELECT uid, iid, ts, rating, np,
               CASE WHEN np >= {C.MIN_POS_FOR_SPLIT} AND rn = 1 THEN 'test'
                    WHEN np >= {C.MIN_POS_FOR_SPLIT} AND rn = 2 THEN 'val'
                    ELSE 'train' END AS split
        FROM ranked
    """)
    con.execute(f"COPY split TO '{C.OUT_DIR}/split.parquet' (FORMAT parquet)")
    counts = dict(con.execute("SELECT split, count(*) FROM split GROUP BY split").fetchall())
    log(f"  split counts: {counts}")

    # ---- 3. popularity from train pool ----
    pop = np.zeros(n_items, dtype=np.int64)
    rows = con.execute("SELECT iid, count(*) FROM split WHERE split='train' GROUP BY iid").fetchall()
    for iid, c in rows:
        pop[iid] = c
    np.save(C.OUT_DIR / "popularity.npy", pop)
    # popularity quantile tiers (over items with >0 train interactions; 0-pop -> tier 0)
    nz = pop[pop > 0]
    edges = np.quantile(nz, np.linspace(0, 1, C.POP_TIERS)) if len(nz) else np.array([0])
    pop_tier = np.digitize(pop, edges[1:-1], right=True).astype(np.int64)
    pop_tier[pop == 0] = 0
    log(f"  popularity: nonzero items={len(nz):,}  max={pop.max():,}")

    # ---- 4. item features (representative edition = max ratings_count) ----
    log("building item features...")
    con.execute(f"""
        CREATE TABLE bf AS
        SELECT * FROM (
            SELECT m.iid AS iid, b.average_rating, b.ratings_count, b.text_reviews_count,
                   b.num_pages, b.publication_year, b.description_len, b.is_ebook,
                   b.language_code, b.format,
                   row_number() OVER (PARTITION BY m.iid ORDER BY b.ratings_count DESC NULLS LAST) AS rk
            FROM {pq('books_core')} b
            JOIN {idm('book_iid_map')} m ON b.book_id = m.book_id
        ) WHERE rk = 1
    """)
    bf = con.execute("SELECT * FROM bf ORDER BY iid").fetchdf()
    assert len(bf) == n_items, (len(bf), n_items)
    assert (bf["iid"].values == np.arange(n_items)).all()

    def fill_flag(col, fill):
        x = bf[col].to_numpy(dtype="float64")
        miss = np.isnan(x).astype(np.float32)
        x = np.where(np.isnan(x), fill, x)
        return x, miss

    avg_rating, _ = fill_flag("average_rating", np.nanmedian(bf["average_rating"]))
    log_rc = np.log1p(np.nan_to_num(bf["ratings_count"].to_numpy("float64"), nan=0.0))
    log_tr = np.log1p(np.nan_to_num(bf["text_reviews_count"].to_numpy("float64"), nan=0.0))
    num_pages, pages_miss = fill_flag("num_pages", np.nanmedian(bf["num_pages"]))
    pub_year, year_miss = fill_flag("publication_year", np.nanmedian(bf["publication_year"]))
    desc_len = bf["description_len"].to_numpy("float64")
    has_desc = (np.nan_to_num(desc_len, nan=0.0) > 0).astype(np.float32)
    log_desc = np.log1p(np.nan_to_num(desc_len, nan=0.0))
    is_ebook = bf["is_ebook"].fillna(False).astype(np.float32).to_numpy()

    cont = np.stack([avg_rating, log_rc, log_tr, num_pages, pub_year, log_desc], axis=1).astype(np.float32)
    cont_mu, cont_sd = cont.mean(0), cont.std(0) + 1e-6
    cont = (cont - cont_mu) / cont_sd
    flags = np.stack([pages_miss, year_miss, has_desc, is_ebook], axis=1).astype(np.float32)
    item_num = np.concatenate([cont, flags], axis=1).astype(np.float32)
    item_num_cols = ["z_avg_rating", "z_log_ratings", "z_log_textrev", "z_num_pages",
                     "z_pub_year", "z_log_desc", "f_pages_missing", "f_year_missing",
                     "f_has_desc", "f_is_ebook"]
    np.save(C.OUT_DIR / "item_num.npy", item_num)

    # item categoricals: language, format, pop_tier
    def topk_index(series, k):
        vc = series.fillna("<na>").value_counts()
        vocab = {v: i + 1 for i, v in enumerate(vc.index[:k])}  # 0 reserved <unk>
        idx = series.fillna("<na>").map(lambda v: vocab.get(v, 0)).to_numpy(np.int64)
        return idx, vocab
    lang_idx, lang_vocab = topk_index(bf["language_code"], C.LANG_TOPK)
    fmt_idx, fmt_vocab = topk_index(bf["format"], C.FORMAT_TOPK)
    item_cat = np.stack([lang_idx, fmt_idx, pop_tier], axis=1).astype(np.int64)
    np.save(C.OUT_DIR / "item_cat.npy", item_cat)
    cat_card = [len(lang_vocab) + 1, len(fmt_vocab) + 1, C.POP_TIERS]

    # item tags: DEPRECATED -> empty bags (old tag profiling removed; v4 does not use tags)
    it_flat, it_off = empty_tags(n_items)
    np.savez(C.OUT_DIR / "item_tags.npz", flat=it_flat, offsets=it_off)

    # ---- 5. user behaviour features (train pool only) ----
    log("building user features (train pool)...")
    uf = con.execute("""
        SELECT uid,
               count(*)                                          AS n_train,
               avg(CASE WHEN rating>0 THEN rating END)           AS avg_rating,
               coalesce(stddev_samp(CASE WHEN rating>0 THEN rating END),0) AS rating_std,
               (max(ts)-min(ts))/86400.0                         AS active_days
        FROM split WHERE split='train' GROUP BY uid
    """).fetchdf()
    n_train_arr = np.zeros(n_users); avg_r = np.full(n_users, np.nan)
    rstd = np.zeros(n_users); adays = np.zeros(n_users); has_train = np.zeros(n_users, np.float32)
    u = uf["uid"].to_numpy()
    n_train_arr[u] = uf["n_train"].to_numpy()
    avg_r[u] = uf["avg_rating"].to_numpy()
    rstd[u] = np.nan_to_num(uf["rating_std"].to_numpy())
    adays[u] = np.nan_to_num(uf["active_days"].to_numpy())
    has_train[u] = 1.0
    global_avg = np.nanmedian(avg_r)
    avg_r = np.where(np.isnan(avg_r), global_avg, avg_r)
    ucont = np.stack([np.log1p(n_train_arr), avg_r, rstd, np.log1p(adays)], axis=1).astype(np.float32)
    umu, usd = ucont.mean(0), ucont.std(0) + 1e-6
    ucont = (ucont - umu) / usd

    # ---- 6. user tag bags: DEPRECATED -> empty (old tag profiling removed).
    # has_profile kept as an all-zero column so user_num stays 6-wide (shape-compatible with ckpts).
    has_profile = np.zeros(n_users, np.float32)
    lk_flat, lk_off = empty_tags(n_users)
    dk_flat, dk_off = empty_tags(n_users)
    np.savez(C.OUT_DIR / "user_liked.npz", flat=lk_flat, offsets=lk_off)
    np.savez(C.OUT_DIR / "user_disliked.npz", flat=dk_flat, offsets=dk_off)

    user_num = np.concatenate([ucont, has_train[:, None], has_profile[:, None]], axis=1).astype(np.float32)
    user_num_cols = ["z_log_ntrain", "z_avg_rating", "z_rating_std", "z_log_active_days",
                     "f_has_train", "f_has_profile"]
    np.save(C.OUT_DIR / "user_num.npy", user_num)

    # ---- 7. user history (last-N train items by ts) ----
    log("building user history matrix...")
    tr = con.execute("SELECT uid, iid, ts FROM split WHERE split='train' ORDER BY uid, ts DESC, iid DESC").fetchdf()
    hist = np.full((n_users, C.HIST_LEN), n_items, dtype=np.int32)   # pad = n_items
    hist_len = np.zeros(n_users, dtype=np.int32)
    uids = tr["uid"].to_numpy(); iids = tr["iid"].to_numpy(np.int32)
    # rows already sorted by (uid, ts desc); fill first HIST_LEN per uid
    start = 0
    N = len(uids)
    i = 0
    while i < N:
        j = i
        while j < N and uids[j] == uids[i]:
            j += 1
        uid = uids[i]
        items = iids[i:j][:C.HIST_LEN]
        hist[uid, :len(items)] = items
        hist_len[uid] = len(items)
        i = j
    np.save(C.OUT_DIR / "user_hist.npy", hist)
    np.save(C.OUT_DIR / "user_hist_len.npy", hist_len)

    # ---- 8. train pairs (user with >=2 train items) + val/test ----
    log("building train pairs + eval sets...")
    trc = con.execute("""
        SELECT uid, iid FROM split WHERE split='train'
          AND uid IN (SELECT uid FROM split WHERE split='train' GROUP BY uid HAVING count(*)>=2)
    """).fetchdf()
    train_pairs = trc[["uid", "iid"]].to_numpy(np.int32)
    val = con.execute("SELECT uid, iid FROM split WHERE split='val'").fetchdf()[["uid", "iid"]].to_numpy(np.int32)
    test = con.execute("SELECT uid, iid FROM split WHERE split='test'").fetchdf()[["uid", "iid"]].to_numpy(np.int32)
    np.save(C.OUT_DIR / "train_pairs.npy", train_pairs)
    np.save(C.OUT_DIR / "val.npy", val)
    np.save(C.OUT_DIR / "test.npy", test)
    log(f"  train_pairs={len(train_pairs):,}  val={len(val):,}  test={len(test):,}")

    # ---- 9. meta ----
    meta = {
        "n_items": int(n_items), "n_users": int(n_users), "hist_len": C.HIST_LEN,
        "n_tags": C.N_TAGS, "tag_pad": C.TAG_PAD, "item_pad": int(n_items),
        "item_num_cols": item_num_cols, "user_num_cols": user_num_cols,
        "item_cat_cols": ["language", "format", "pop_tier"], "item_cat_cardinality": cat_card,
        "counts": {k: int(v) for k, v in counts.items()},
        "n_train_pairs": int(len(train_pairs)), "n_val": int(len(val)), "n_test": int(len(test)),
        "scalers": {"item_cont_mu": cont_mu.tolist(), "item_cont_sd": cont_sd.tolist(),
                    "user_cont_mu": umu.tolist(), "user_cont_sd": usd.tolist()},
        "lang_vocab_size": len(lang_vocab) + 1, "format_vocab_size": len(fmt_vocab) + 1,
        "positive_rule": C.POSITIVE_RULE, "min_pos_for_split": C.MIN_POS_FOR_SPLIT,
    }
    json.dump(meta, open(C.OUT_DIR / "meta.json", "w"), indent=2)
    log(f"DONE. wrote artifacts to {C.OUT_DIR}")
    log(f"  item_num{item_num.shape} item_cat{item_cat.shape} user_num{user_num.shape} hist{hist.shape}")


if __name__ == "__main__":
    main()
