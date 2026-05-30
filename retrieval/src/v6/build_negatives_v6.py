"""v6 graded negatives: HARD (rating<=2) + SOFT (rating==3) explicit-negative pools, leakage-safe.

Both pulled from interactions_core, work-level dedup, kept only if strictly older than the user's
earliest held-out (val) item (ts < min val_ts) so they sit in the training window. Disjoint from
positives (rating>=4) by construction. Pure CPU.

Outputs -> data/v6/:
  hard_pad.npy  [n_users, NEG_M_HARD] int  rating<=2 items, pad=n_items
  soft_pad.npy  [n_users, NEG_M_SOFT] int  rating==3 items, pad=n_items

  python build_negatives_v6.py
"""
from __future__ import annotations
import duckdb, numpy as np
import config as C, dataset as D


def build(con, where, M, n_items, n_users, pq, idm):
    df = con.execute(f"""
        WITH g AS (
            SELECT m.iid AS iid, u.uid AS uid, epoch(max(i.date_added))::BIGINT AS ts
            FROM {pq('interactions_core')} i
            JOIN {idm('book_iid_map')} m ON i.book_id=m.book_id JOIN {idm('uid_map')} u ON i.user_id=u.user_id
            WHERE {where} GROUP BY m.iid, u.uid)
        SELECT g.uid, g.iid FROM g LEFT JOIN valcut v ON g.uid=v.uid
        WHERE v.cut IS NULL OR g.ts < v.cut ORDER BY g.uid, g.iid""").fetchdf()
    uid = df["uid"].to_numpy(np.int64); iid = df["iid"].to_numpy(np.int64)
    off = np.zeros(n_users+1, np.int64); np.add.at(off, uid+1, 1); off = np.cumsum(off)
    pad = np.full((n_users, M), n_items, np.int64)
    for u in range(n_users):
        s, e = off[u], off[u+1]
        if e > s: pad[u, :min(M, e-s)] = iid[s:e][:M]
    lens = off[1:]-off[:-1]
    return pad, len(iid), int((lens > 0).sum()), (lens[lens > 0].mean() if (lens > 0).any() else 0)


def main():
    C.ensure_dirs()
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    pq = lambda n: f"read_parquet('{C.PARQUET}/{n}.parquet')"
    idm = lambda n: f"read_parquet('{C.ID_MAPS}/{n}.parquet')"
    con.execute(f"""CREATE TEMP TABLE valcut AS
        SELECT uid, min(ts) AS cut FROM read_parquet('{C.BASE_DIR}/split.parquet') WHERE split='val' GROUP BY uid""")

    hard, nh, uh, mh = build(con, "i.rating BETWEEN 1 AND 2", C.NEG_M_HARD, n_items, n_users, pq, idm)
    np.save(C.V6_DIR / "hard_pad.npy", hard)
    soft, ns, us, msf = build(con, "i.rating = 3", C.NEG_M_SOFT, n_items, n_users, pq, idm)
    np.save(C.V6_DIR / "soft_pad.npy", soft)
    print(f"HARD (rating<=2): {nh:,} pairs / {uh:,} users (mean {mh:.1f}); pad [n_users,{C.NEG_M_HARD}]")
    print(f"SOFT (rating==3): {ns:,} pairs / {us:,} users (mean {msf:.1f}); pad [n_users,{C.NEG_M_SOFT}]")
    print(f"wrote hard_pad.npy / soft_pad.npy to {C.V6_DIR}")


if __name__ == "__main__":
    main()
