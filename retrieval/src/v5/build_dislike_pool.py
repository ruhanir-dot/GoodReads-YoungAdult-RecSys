"""Lever B data: explicit dislike hard-negative pool, REBUILT for the rating>=4 base.

With positives = rating>=4, split.parquet no longer contains rating<=2 rows, so dislikes are
taken directly from interactions_core (rating<=2, work-level dedup). Leakage-safe: a dislike is
kept only if it is strictly OLDER than the user's held-out val target (ts < val_ts), i.e. inside
the training window; users with no val/test (all-train) keep all their dislikes. Positives
(rating>=4) and dislikes (rating<=2) are now DISJOINT by construction -> no pos/neg
contradiction, no self-collision, and dislikes never enter user_hist (which pools rating>=4).

Outputs -> data/v5/:
  dislike_pad.npy   [n_users, M] int   up to M dislike item ids per user, pad = n_items
  dislike_pool.npz  CSR (offsets, flat)

  python build_dislike_pool.py
"""
from __future__ import annotations
import duckdb
import numpy as np
import config as C
import dataset as D


def main():
    C.ensure_dirs()
    meta = D.load_meta(); n_items, n_users = meta["n_items"], meta["n_users"]
    con = duckdb.connect(); con.execute("PRAGMA threads=16")
    pq = lambda n: f"read_parquet('{C.PARQUET}/{n}.parquet')"
    idm = lambda n: f"read_parquet('{C.ID_MAPS}/{n}.parquet')"

    # per-user val timestamp = leakage cutoff (dislike must be strictly older); +inf if no val
    con.execute(f"""CREATE TEMP TABLE valts AS
        SELECT uid, ts AS val_ts FROM read_parquet('{C.BASE_DIR}/split.parquet') WHERE split='val'""")

    df = con.execute(f"""
        WITH dis AS (
            SELECT m.iid AS iid, u.uid AS uid, epoch(max(i.date_added))::BIGINT AS ts
            FROM {pq('interactions_core')} i
            JOIN {idm('book_iid_map')} m ON i.book_id = m.book_id
            JOIN {idm('uid_map')} u      ON i.user_id = u.user_id
            WHERE i.rating BETWEEN 1 AND 2
            GROUP BY m.iid, u.uid
        )
        SELECT d.uid, d.iid FROM dis d LEFT JOIN valts v ON d.uid = v.uid
        WHERE v.val_ts IS NULL OR d.ts < v.val_ts
        ORDER BY d.uid, d.iid
    """).fetchdf()
    uid = df["uid"].to_numpy(np.int64); iid = df["iid"].to_numpy(np.int64)
    assert (iid < n_items).all() and (uid < n_users).all()

    off = np.zeros(n_users + 1, np.int64); np.add.at(off, uid + 1, 1); off = np.cumsum(off)
    np.savez(C.V5_DIR / "dislike_pool.npz", offsets=off, flat=iid)

    M = C.DISLIKE_M
    pad = np.full((n_users, M), n_items, np.int64)
    for u in range(n_users):
        s, e = off[u], off[u + 1]
        if e > s: pad[u, :min(M, e - s)] = iid[s:e][:M]
    np.save(C.V5_DIR / "dislike_pad.npy", pad)

    lens = off[1:] - off[:-1]
    print(f"dislike pool (rating<=2, ts<val_ts): {len(iid):,} pairs over {int((lens>0).sum()):,}/{n_users:,} users "
          f"(mean {lens[lens>0].mean():.1f}, max {lens.max()}); pad [n_users,{M}]")
    print(f"wrote dislike_pad.npy / dislike_pool.npz to {C.V5_DIR}")


if __name__ == "__main__":
    main()
