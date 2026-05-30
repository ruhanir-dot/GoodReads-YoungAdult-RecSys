"""v5 base build — IDENTICAL to v4 prepare_data EXCEPT positives = `rating>=4` ONLY
(v1-v4 used `is_read OR rating>=4`). This cleanly separates positives (rating>=4) from the
lever-B dislike pool (rating<=2) so they are DISJOINT by construction — removing the
self-contradiction the audit found (where is_read rating1-2 items were both positive and
hard-negative). NOTE: this changes the eval/positive set, so R@100 here is NOT comparable to
the v1-v4 0.486 baseline (intentional, per user choice).

Writes the collaborative base to data/v5/base/ (C.BASE_DIR). Item-content arrays (bge/sem_codes/
language) are positive-rule-independent and stay in data/v5/. Pure CPU/duckdb.

  python prepare_data_v5.py
"""
from __future__ import annotations
import json, time
import duckdb
import numpy as np
import config as C

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def empty_tags(n):
    return np.full(n, C.TAG_PAD, dtype=np.int64), np.arange(n, dtype=np.int64)


def main():
    C.ensure_dirs()
    OUT = C.BASE_DIR
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    pq = lambda n: f"read_parquet('{C.PARQUET}/{n}.parquet')"
    idm = lambda n: f"read_parquet('{C.ID_MAPS}/{n}.parquet')"
    n_items = con.execute(f"SELECT max(iid)+1 FROM {idm('book_iid_map')}").fetchone()[0]
    n_users = con.execute(f"SELECT max(uid)+1 FROM {idm('uid_map')}").fetchone()[0]
    log(f"n_items={n_items:,} n_users={n_users:,}  POSITIVE_RULE=rating>=4")

    # ---- 1. positives (rating>=4 ONLY), work-level dedup ----
    con.execute(f"""
        CREATE TABLE pos AS
        SELECT m.iid AS iid, u.uid AS uid, max(i.rating) AS rating,
               epoch(max(i.date_added))::BIGINT AS ts
        FROM {pq('interactions_core')} i
        JOIN {idm('book_iid_map')} m ON i.book_id = m.book_id
        JOIN {idm('uid_map')} u      ON i.user_id = u.user_id
        WHERE i.rating >= 4
        GROUP BY m.iid, u.uid
    """)
    log(f"  positive (uid,iid) pairs = {con.execute('SELECT count(*) FROM pos').fetchone()[0]:,}")

    # ---- 2. leave-last-out split ----
    con.execute("""CREATE TABLE ranked AS SELECT *,
        row_number() OVER (PARTITION BY uid ORDER BY ts DESC, iid DESC) AS rn,
        count(*) OVER (PARTITION BY uid) AS np FROM pos""")
    con.execute(f"""CREATE TABLE split AS SELECT uid, iid, ts, rating, np,
        CASE WHEN np >= {C.MIN_POS_FOR_SPLIT} AND rn=1 THEN 'test'
             WHEN np >= {C.MIN_POS_FOR_SPLIT} AND rn=2 THEN 'val'
             ELSE 'train' END AS split FROM ranked""")
    con.execute(f"COPY split TO '{OUT}/split.parquet' (FORMAT parquet)")
    counts = dict(con.execute("SELECT split, count(*) FROM split GROUP BY split").fetchall())
    log(f"  split counts: {counts}")

    # ---- 3. popularity (train pool) + tiers ----
    pop = np.zeros(n_items, np.int64)
    for iid, c in con.execute("SELECT iid, count(*) FROM split WHERE split='train' GROUP BY iid").fetchall():
        pop[iid] = c
    np.save(OUT / "popularity.npy", pop)
    nz = pop[pop > 0]
    edges = np.quantile(nz, np.linspace(0, 1, C.POP_TIERS)) if len(nz) else np.array([0])
    pop_tier = np.digitize(pop, edges[1:-1], right=True).astype(np.int64); pop_tier[pop == 0] = 0
    log(f"  popularity nonzero items={len(nz):,} max={pop.max():,}")

    # ---- 4. item features (representative edition; pos-rule-independent except pop_tier) ----
    con.execute(f"""CREATE TABLE bf AS SELECT * FROM (
        SELECT m.iid AS iid, b.average_rating, b.ratings_count, b.text_reviews_count,
               b.num_pages, b.publication_year, b.description_len, b.is_ebook, b.language_code, b.format,
               row_number() OVER (PARTITION BY m.iid ORDER BY b.ratings_count DESC NULLS LAST) AS rk
        FROM {pq('books_core')} b JOIN {idm('book_iid_map')} m ON b.book_id=m.book_id) WHERE rk=1""")
    bf = con.execute("SELECT * FROM bf ORDER BY iid").fetchdf()
    assert len(bf) == n_items and (bf["iid"].values == np.arange(n_items)).all()

    def fill_flag(col, fill):
        x = bf[col].to_numpy("float64"); miss = np.isnan(x).astype(np.float32)
        return np.where(np.isnan(x), fill, x), miss
    avg_rating, _ = fill_flag("average_rating", np.nanmedian(bf["average_rating"]))
    log_rc = np.log1p(np.nan_to_num(bf["ratings_count"].to_numpy("float64"), nan=0.0))
    log_tr = np.log1p(np.nan_to_num(bf["text_reviews_count"].to_numpy("float64"), nan=0.0))
    num_pages, pages_miss = fill_flag("num_pages", np.nanmedian(bf["num_pages"]))
    pub_year, year_miss = fill_flag("publication_year", np.nanmedian(bf["publication_year"]))
    desc_len = bf["description_len"].to_numpy("float64")
    has_desc = (np.nan_to_num(desc_len, nan=0.0) > 0).astype(np.float32)
    log_desc = np.log1p(np.nan_to_num(desc_len, nan=0.0))
    is_ebook = bf["is_ebook"].fillna(False).astype(np.float32).to_numpy()
    cont = np.stack([avg_rating, log_rc, log_tr, num_pages, pub_year, log_desc], 1).astype(np.float32)
    cont_mu, cont_sd = cont.mean(0), cont.std(0) + 1e-6; cont = (cont - cont_mu) / cont_sd
    flags = np.stack([pages_miss, year_miss, has_desc, is_ebook], 1).astype(np.float32)
    item_num = np.concatenate([cont, flags], 1).astype(np.float32)
    np.save(OUT / "item_num.npy", item_num)
    item_num_cols = ["z_avg_rating","z_log_ratings","z_log_textrev","z_num_pages","z_pub_year",
                     "z_log_desc","f_pages_missing","f_year_missing","f_has_desc","f_is_ebook"]

    def topk_index(series, k):
        vc = series.fillna("<na>").value_counts(); vocab = {v: i+1 for i, v in enumerate(vc.index[:k])}
        return series.fillna("<na>").map(lambda v: vocab.get(v, 0)).to_numpy(np.int64), vocab
    lang_idx, lang_vocab = topk_index(bf["language_code"], C.LANG_TOPK)
    fmt_idx, fmt_vocab = topk_index(bf["format"], C.FORMAT_TOPK)
    item_cat = np.stack([lang_idx, fmt_idx, pop_tier], 1).astype(np.int64)
    np.save(OUT / "item_cat.npy", item_cat)
    cat_card = [len(lang_vocab)+1, len(fmt_vocab)+1, C.POP_TIERS]
    it_flat, it_off = empty_tags(n_items); np.savez(OUT / "item_tags.npz", flat=it_flat, offsets=it_off)

    # ---- 5. user behaviour features (train pool) ----
    uf = con.execute("""SELECT uid, count(*) AS n_train,
        avg(CASE WHEN rating>0 THEN rating END) AS avg_rating,
        coalesce(stddev_samp(CASE WHEN rating>0 THEN rating END),0) AS rating_std,
        (max(ts)-min(ts))/86400.0 AS active_days
        FROM split WHERE split='train' GROUP BY uid""").fetchdf()
    n_train_arr = np.zeros(n_users); avg_r = np.full(n_users, np.nan)
    rstd = np.zeros(n_users); adays = np.zeros(n_users); has_train = np.zeros(n_users, np.float32)
    u = uf["uid"].to_numpy()
    n_train_arr[u] = uf["n_train"].to_numpy(); avg_r[u] = uf["avg_rating"].to_numpy()
    rstd[u] = np.nan_to_num(uf["rating_std"].to_numpy()); adays[u] = np.nan_to_num(uf["active_days"].to_numpy())
    has_train[u] = 1.0
    avg_r = np.where(np.isnan(avg_r), np.nanmedian(avg_r), avg_r)
    ucont = np.stack([np.log1p(n_train_arr), avg_r, rstd, np.log1p(adays)], 1).astype(np.float32)
    umu, usd = ucont.mean(0), ucont.std(0) + 1e-6; ucont = (ucont - umu) / usd
    has_profile = np.zeros(n_users, np.float32)
    lk = empty_tags(n_users); dk = empty_tags(n_users)
    np.savez(OUT / "user_liked.npz", flat=lk[0], offsets=lk[1])
    np.savez(OUT / "user_disliked.npz", flat=dk[0], offsets=dk[1])
    user_num = np.concatenate([ucont, has_train[:, None], has_profile[:, None]], 1).astype(np.float32)
    np.save(OUT / "user_num.npy", user_num)
    user_num_cols = ["z_log_ntrain","z_avg_rating","z_rating_std","z_log_active_days","f_has_train","f_has_profile"]

    # ---- 6. user history (last-N train items) ----
    tr = con.execute("SELECT uid, iid FROM split WHERE split='train' ORDER BY uid, ts DESC, iid DESC").fetchdf()
    hist = np.full((n_users, C.HIST_LEN), n_items, np.int32); hist_len = np.zeros(n_users, np.int32)
    uids = tr["uid"].to_numpy(); iids = tr["iid"].to_numpy(np.int32)
    i, N = 0, len(uids)
    while i < N:
        j = i
        while j < N and uids[j] == uids[i]: j += 1
        items = iids[i:j][:C.HIST_LEN]; hist[uids[i], :len(items)] = items; hist_len[uids[i]] = len(items); i = j
    np.save(OUT / "user_hist.npy", hist); np.save(OUT / "user_hist_len.npy", hist_len)

    # ---- 7. train pairs (>=2 train items) + val/test ----
    trc = con.execute("""SELECT uid, iid FROM split WHERE split='train'
        AND uid IN (SELECT uid FROM split WHERE split='train' GROUP BY uid HAVING count(*)>=2)""").fetchdf()
    train_pairs = trc[["uid","iid"]].to_numpy(np.int32)
    val = con.execute("SELECT uid,iid FROM split WHERE split='val'").fetchdf()[["uid","iid"]].to_numpy(np.int32)
    test = con.execute("SELECT uid,iid FROM split WHERE split='test'").fetchdf()[["uid","iid"]].to_numpy(np.int32)
    np.save(OUT / "train_pairs.npy", train_pairs); np.save(OUT / "val.npy", val); np.save(OUT / "test.npy", test)
    log(f"  train_pairs={len(train_pairs):,} val={len(val):,} test={len(test):,}")

    meta = {"n_items": int(n_items), "n_users": int(n_users), "hist_len": C.HIST_LEN,
            "n_tags": C.N_TAGS, "tag_pad": C.TAG_PAD, "item_pad": int(n_items),
            "item_num_cols": item_num_cols, "user_num_cols": user_num_cols,
            "item_cat_cols": ["language","format","pop_tier"], "item_cat_cardinality": cat_card,
            "counts": {k: int(v) for k, v in counts.items()},
            "n_train_pairs": int(len(train_pairs)), "n_val": int(len(val)), "n_test": int(len(test)),
            "scalers": {"item_cont_mu": cont_mu.tolist(), "item_cont_sd": cont_sd.tolist(),
                        "user_cont_mu": umu.tolist(), "user_cont_sd": usd.tolist()},
            "lang_vocab_size": len(lang_vocab)+1, "format_vocab_size": len(fmt_vocab)+1,
            "positive_rule": C.POSITIVE_RULE, "min_pos_for_split": C.MIN_POS_FOR_SPLIT}
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    log(f"DONE -> {OUT}  item_num{item_num.shape} item_cat{item_cat.shape} user_num{user_num.shape}")


if __name__ == "__main__":
    main()
