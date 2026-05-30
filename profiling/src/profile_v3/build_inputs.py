"""Build leakage-free profiling inputs.

USER (train reviews only): recent K_POS liked + K_NEG disliked reviews per user
(>=U_MIN_REV), with text from the raw NDJSON; per-review language detected in CODE
(langdetect) and aggregated into a normalized frequency dict.
BOOK: representative edition's title + description + language_code (NO reviews).

  python build_inputs.py user
  python build_inputs.py book
  python build_inputs.py both
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from multiprocessing import Pool

import duckdb
import numpy as np
import pandas as pd

import config as C

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


# ----------------------------------------------------------------- language detection (code-level)
def _detect(text):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        if not text or len(text.strip()) < 12:
            return "unknown"
        lg = detect(text)
        return "zh" if lg.startswith("zh") else lg          # zh-cn/zh-tw -> zh
    except Exception:
        return "unknown"


def _lang_dict(langs):
    c = Counter(l for l in langs if l != "unknown")
    if not c:
        return {"unknown": 1.0}
    tot = sum(c.values())
    d = {k: v / tot for k, v in c.items()}
    d = {k: v for k, v in d.items() if v >= 0.05}           # drop noise
    s = sum(d.values()) or 1.0
    return {k: round(v / s, 3) for k, v in sorted(d.items(), key=lambda x: -x[1])}


def build_user():
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    log("joining TRAIN reviews (split='train' only -> no val/test leakage)")
    con.execute(f"""create temp table jr as
        select u.uid, r.review_id, r.rating, r.date_added, b.title
        from read_parquet('{C.PARQUET}/reviews_core.parquet') r
        join read_parquet('{C.ID_MAPS}/book_iid_map.parquet') m on r.book_id=m.book_id
        join read_parquet('{C.ID_MAPS}/uid_map.parquet') u      on r.user_id=u.user_id
        join read_parquet('{C.SPLIT}') s on s.uid=u.uid and s.iid=m.iid and s.split='train'
        join read_parquet('{C.PARQUET}/books_core.parquet') b on b.book_id=r.book_id
        where r.has_review_text and r.rating is not null""")
    con.execute(f"create temp table elig as select uid, count(*) n from jr group by uid having count(*) >= {C.U_MIN_REV}")
    n_users = con.execute("select count(*) from elig").fetchone()[0]
    log(f"eligible users (>= {C.U_MIN_REV} train reviews): {n_users:,}")
    con.execute(f"""create temp table sel as
        with j as (select jr.* from jr join elig using(uid)),
        pos as (select *, row_number() over (partition by uid order by date_added desc, review_id) rn from j where rating>=4),
        neg as (select *, row_number() over (partition by uid order by date_added desc, review_id) rn from j where rating<=2)
        select uid, review_id, rating, date_added, title, 0 is_neg, rn from pos where rn<={C.K_POS}
        union all
        select uid, review_id, rating, date_added, title, 1 is_neg, rn from neg where rn<={C.K_NEG}""")
    con.execute(f"""create temp table rawtext as
        select review_id, review_text from read_json_auto('{C.RAW_REVIEWS}', format='newline_delimited',
            maximum_object_size=67108864) where review_text is not null and length(review_text) >= {C.MIN_REVIEW_LEN}""")
    df = con.execute("""select s.uid, s.title, s.rating, s.is_neg, s.date_added, rt.review_text
        from sel s join rawtext rt on rt.review_id=s.review_id order by s.uid, s.is_neg, s.date_added desc""").fetchdf()
    log(f"selected {len(df):,} reviews across {df['uid'].nunique():,} users; detecting language (multiproc)...")
    with Pool(32) as p:
        df["lang"] = p.map(_detect, df["review_text"].tolist(), chunksize=2000)
    log("grouping per user...")
    recs = []
    for uid, g in df.groupby("uid", sort=False):
        reviews = [{"title": str(t)[:C.TITLE_CAP], "rating": int(r), "neg": bool(n),
                    "text": str(x)[:C.REVIEW_CHAR_CAP]}
                   for t, r, n, x in zip(g.title, g.rating, g.is_neg, g.review_text)]
        recs.append({"uid": int(uid), "n_train_rev": len(reviews),
                     "language_json": json.dumps(_lang_dict(g["lang"].tolist())),
                     "reviews_json": json.dumps(reviews, ensure_ascii=False)})
    pd.DataFrame(recs).to_parquet(C.USER_INPUTS, index=False)
    log(f"wrote {C.USER_INPUTS}  ({len(recs):,} users)")


def build_book():
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    log("book inputs: representative edition title + description + language_code (no reviews)")
    df = con.execute(f"""select iid, title, description, language_code from (
            select m.iid, b.title, b.description, b.language_code,
                   row_number() over (partition by m.iid order by b.ratings_count desc nulls last) rk
            from read_parquet('{C.PARQUET}/books_core.parquet') b
            join read_parquet('{C.ID_MAPS}/book_iid_map.parquet') m on b.book_id=m.book_id
        ) where rk=1 order by iid""").fetchdf()
    out = pd.DataFrame({
        "iid": df["iid"].astype(int),
        "title": df["title"].fillna("").str.slice(0, C.TITLE_CAP),
        "description": df["description"].fillna("").str.slice(0, C.DESC_CAP),
        "language_code": df["language_code"].fillna(""),
    })
    out.to_parquet(C.BOOK_INPUTS, index=False)
    n_miss = int((out["language_code"] == "").sum())
    log(f"wrote {C.BOOK_INPUTS}  ({len(out):,} books; language_code missing={n_miss:,} -> LLM will infer)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("user", "both"): build_user()
    if which in ("book", "both"): build_book()
