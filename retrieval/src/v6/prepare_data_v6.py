"""v6 base build — LEAVE-K-OUT split (proportional, temporal). Positives = rating>=4.

Per user with n positives: if n >= MIN_POS_FOR_EVAL, hold out the most-recent
k_test = max(1, round(TEST_FRAC*n)) as TEST and the next k_val = max(1, round(VAL_FRAC*n)) as
VAL; the oldest rest is TRAIN. Users with n < MIN_POS_FOR_EVAL are all-train. This keeps the
held-out fraction ~10/10/80 regardless of activity (unlike leave-LAST-out's 1/n). Eval becomes
SET-based (each user has a SET of held-out items). Temporal -> no future leakage. Pure CPU.

Writes data/v6/base/. Item/user content arrays (bge/lang) are split-independent -> reused from data/v5.

  python prepare_data_v6.py
"""
from __future__ import annotations
import json, time
import duckdb, numpy as np
import config as C

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
def empty_tags(n): return np.full(n, C.TAG_PAD, dtype=np.int64), np.arange(n, dtype=np.int64)


def main():
    C.ensure_dirs(); OUT = C.BASE_DIR
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    pq = lambda n: f"read_parquet('{C.PARQUET}/{n}.parquet')"
    idm = lambda n: f"read_parquet('{C.ID_MAPS}/{n}.parquet')"
    n_items = con.execute(f"SELECT max(iid)+1 FROM {idm('book_iid_map')}").fetchone()[0]
    n_users = con.execute(f"SELECT max(uid)+1 FROM {idm('uid_map')}").fetchone()[0]
    log(f"n_items={n_items:,} n_users={n_users:,}  leave-k-out TEST/VAL={C.TEST_FRAC}/{C.VAL_FRAC} min_eval={C.MIN_POS_FOR_EVAL}")

    con.execute(f"""CREATE TABLE pos AS
        SELECT m.iid AS iid, u.uid AS uid, max(i.rating) AS rating, epoch(max(i.date_added))::BIGINT AS ts
        FROM {pq('interactions_core')} i
        JOIN {idm('book_iid_map')} m ON i.book_id=m.book_id JOIN {idm('uid_map')} u ON i.user_id=u.user_id
        WHERE i.rating >= 4 GROUP BY m.iid, u.uid""")
    log(f"  positive (uid,iid) = {con.execute('SELECT count(*) FROM pos').fetchone()[0]:,}")

    # leave-k-out: rn=1 is most recent; kt/kv = max(1, round(frac*np)) when np>=MIN else 0
    con.execute("""CREATE TABLE ranked AS SELECT *,
        row_number() OVER (PARTITION BY uid ORDER BY ts DESC, iid DESC) AS rn,
        count(*) OVER (PARTITION BY uid) AS np FROM pos""")
    con.execute(f"""CREATE TABLE split AS
        WITH k AS (SELECT *,
            CASE WHEN np >= {C.MIN_POS_FOR_EVAL} THEN greatest(1, round({C.TEST_FRAC}*np)) ELSE 0 END AS kt,
            CASE WHEN np >= {C.MIN_POS_FOR_EVAL} THEN greatest(1, round({C.VAL_FRAC}*np))  ELSE 0 END AS kv
            FROM ranked)
        SELECT uid, iid, ts, rating, np,
            CASE WHEN rn <= kt THEN 'test' WHEN rn <= kt+kv THEN 'val' ELSE 'train' END AS split
        FROM k""")
    con.execute(f"COPY split TO '{OUT}/split.parquet' (FORMAT parquet)")
    counts = dict(con.execute("SELECT split, count(*) FROM split GROUP BY split").fetchall())
    nu_test = con.execute("SELECT count(DISTINCT uid) FROM split WHERE split='test'").fetchone()[0]
    log(f"  split rows: {counts} | eval users={nu_test:,}")

    pop = np.zeros(n_items, np.int64)
    for iid, c in con.execute("SELECT iid,count(*) FROM split WHERE split='train' GROUP BY iid").fetchall(): pop[iid] = c
    np.save(OUT / "popularity.npy", pop)
    nz = pop[pop > 0]; edges = np.quantile(nz, np.linspace(0, 1, C.POP_TIERS)) if len(nz) else np.array([0])
    pop_tier = np.digitize(pop, edges[1:-1], right=True).astype(np.int64); pop_tier[pop == 0] = 0

    con.execute(f"""CREATE TABLE bf AS SELECT * FROM (
        SELECT m.iid AS iid, b.average_rating, b.ratings_count, b.text_reviews_count, b.num_pages,
               b.publication_year, b.description_len, b.is_ebook, b.language_code, b.format,
               row_number() OVER (PARTITION BY m.iid ORDER BY b.ratings_count DESC NULLS LAST) AS rk
        FROM {pq('books_core')} b JOIN {idm('book_iid_map')} m ON b.book_id=m.book_id) WHERE rk=1""")
    bf = con.execute("SELECT * FROM bf ORDER BY iid").fetchdf()
    assert len(bf) == n_items and (bf["iid"].values == np.arange(n_items)).all()
    def fill(col, f): x = bf[col].to_numpy("float64"); return np.where(np.isnan(x), f, x), np.isnan(x).astype(np.float32)
    avg_rating, _ = fill("average_rating", np.nanmedian(bf["average_rating"]))
    log_rc = np.log1p(np.nan_to_num(bf["ratings_count"].to_numpy("float64"), nan=0.0))
    log_tr = np.log1p(np.nan_to_num(bf["text_reviews_count"].to_numpy("float64"), nan=0.0))
    num_pages, pmiss = fill("num_pages", np.nanmedian(bf["num_pages"]))
    pub_year, ymiss = fill("publication_year", np.nanmedian(bf["publication_year"]))
    desc_len = bf["description_len"].to_numpy("float64")
    has_desc = (np.nan_to_num(desc_len, nan=0.0) > 0).astype(np.float32); log_desc = np.log1p(np.nan_to_num(desc_len, nan=0.0))
    is_ebook = bf["is_ebook"].fillna(False).astype(np.float32).to_numpy()
    cont = np.stack([avg_rating, log_rc, log_tr, num_pages, pub_year, log_desc], 1).astype(np.float32)
    cmu, csd = cont.mean(0), cont.std(0)+1e-6; cont = (cont-cmu)/csd
    item_num = np.concatenate([cont, np.stack([pmiss, ymiss, has_desc, is_ebook], 1).astype(np.float32)], 1).astype(np.float32)
    np.save(OUT / "item_num.npy", item_num)
    def topk(series, k):
        vc = series.fillna("<na>").value_counts(); v = {x: i+1 for i, x in enumerate(vc.index[:k])}
        return series.fillna("<na>").map(lambda z: v.get(z, 0)).to_numpy(np.int64), v
    lang_idx, lv = topk(bf["language_code"], C.LANG_TOPK); fmt_idx, fv = topk(bf["format"], C.FORMAT_TOPK)
    np.save(OUT / "item_cat.npy", np.stack([lang_idx, fmt_idx, pop_tier], 1).astype(np.int64))
    cat_card = [len(lv)+1, len(fv)+1, C.POP_TIERS]
    _it = empty_tags(n_items); np.savez(OUT / "item_tags.npz", flat=_it[0], offsets=_it[1])

    uf = con.execute("""SELECT uid, count(*) n_train, avg(CASE WHEN rating>0 THEN rating END) avg_rating,
        coalesce(stddev_samp(CASE WHEN rating>0 THEN rating END),0) rating_std, (max(ts)-min(ts))/86400.0 active_days
        FROM split WHERE split='train' GROUP BY uid""").fetchdf()
    n_tr = np.zeros(n_users); avg_r = np.full(n_users, np.nan); rstd = np.zeros(n_users); adays = np.zeros(n_users); has_tr = np.zeros(n_users, np.float32)
    u = uf["uid"].to_numpy(); n_tr[u] = uf["n_train"].to_numpy(); avg_r[u] = uf["avg_rating"].to_numpy()
    rstd[u] = np.nan_to_num(uf["rating_std"].to_numpy()); adays[u] = np.nan_to_num(uf["active_days"].to_numpy()); has_tr[u] = 1.0
    avg_r = np.where(np.isnan(avg_r), np.nanmedian(avg_r), avg_r)
    uc = np.stack([np.log1p(n_tr), avg_r, rstd, np.log1p(adays)], 1).astype(np.float32)
    umu, usd = uc.mean(0), uc.std(0)+1e-6; uc = (uc-umu)/usd
    user_num = np.concatenate([uc, has_tr[:, None], np.zeros((n_users, 1), np.float32)], 1).astype(np.float32)
    np.save(OUT / "user_num.npy", user_num)
    for nm in ("user_liked", "user_disliked"):
        a = empty_tags(n_users); np.savez(OUT / f"{nm}.npz", flat=a[0], offsets=a[1])

    tr = con.execute("SELECT uid,iid FROM split WHERE split='train' ORDER BY uid, ts DESC, iid DESC").fetchdf()
    hist = np.full((n_users, C.HIST_LEN), n_items, np.int32); hlen = np.zeros(n_users, np.int32)
    uu = tr["uid"].to_numpy(); ii = tr["iid"].to_numpy(np.int32); i, N = 0, len(uu)
    while i < N:
        j = i
        while j < N and uu[j] == uu[i]: j += 1
        it = ii[i:j][:C.HIST_LEN]; hist[uu[i], :len(it)] = it; hlen[uu[i]] = len(it); i = j
    np.save(OUT / "user_hist.npy", hist); np.save(OUT / "user_hist_len.npy", hlen)

    train_pairs = con.execute("SELECT uid,iid FROM split WHERE split='train'").fetchdf()[["uid","iid"]].to_numpy(np.int32)
    val = con.execute("SELECT uid,iid FROM split WHERE split='val' ORDER BY uid").fetchdf()[["uid","iid"]].to_numpy(np.int32)
    test = con.execute("SELECT uid,iid FROM split WHERE split='test' ORDER BY uid").fetchdf()[["uid","iid"]].to_numpy(np.int32)
    np.save(OUT / "train_pairs.npy", train_pairs); np.save(OUT / "val.npy", val); np.save(OUT / "test.npy", test)
    log(f"  train_pairs={len(train_pairs):,} val_rows={len(val):,} test_rows={len(test):,}")

    meta = {"n_items": int(n_items), "n_users": int(n_users), "hist_len": C.HIST_LEN, "n_tags": C.N_TAGS,
            "tag_pad": C.TAG_PAD, "item_pad": int(n_items), "item_cat_cardinality": cat_card,
            "item_cat_cols": ["language","format","pop_tier"], "counts": {k: int(v) for k, v in counts.items()},
            "n_eval_users": int(nu_test), "n_train_pairs": int(len(train_pairs)), "n_val": int(len(val)), "n_test": int(len(test)),
            "lang_vocab_size": len(lv)+1, "format_vocab_size": len(fv)+1, "positive_rule": C.POSITIVE_RULE,
            "split": "leave-k-out", "test_frac": C.TEST_FRAC, "val_frac": C.VAL_FRAC, "min_pos_for_eval": C.MIN_POS_FOR_EVAL}
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
