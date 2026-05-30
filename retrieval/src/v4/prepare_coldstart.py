"""Cold-start experiment prep: hold ~COLD_FRAC of items fully OUT of training
(never a target, never in any user's pooled history; at inference they get the
pad/<unk> ID = a realistic brand-new item). Reuses all v1 arrays; only writes the
holdout-specific arrays to retrieval/data/v4/cold/ (= config.COLD_DIR).

  python prepare_coldstart.py            (pure CPU)

Outputs (data/v4/cold/):
  cold_items.npy        [n_items] bool   (True = held out of training)
  id_lookup.npy         [n_items] int    warm->iid, cold->n_items (pad row)
  train_pairs_cold.npy  [P,2]            v1 train pairs minus cold targets
  user_hist_cold.npy    [n_users,H]      v1 history with cold items -> pad
  popularity_cold.npy   [n_items]        target frequency in cold-holdout train (for logQ)
  meta_cold.json        v1 meta + cold-holdout info
"""
from __future__ import annotations

import json

import numpy as np

import config as C


def main():
    C.COLD_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(C.OUT_DIR / "meta.json"))
    n_items, n_users = meta["n_items"], meta["n_users"]

    rng = np.random.RandomState(C.COLD_SEED)
    cold = np.zeros(n_items, dtype=bool)
    cold[rng.choice(n_items, int(round(C.COLD_FRAC * n_items)), replace=False)] = True
    np.save(C.COLD_DIR / "cold_items.npy", cold)

    id_lookup = np.arange(n_items, dtype=np.int64)
    id_lookup[cold] = n_items                      # cold -> pad row (zero, untrained)
    np.save(C.COLD_DIR / "id_lookup.npy", id_lookup)

    # train pairs: drop cold targets
    tp = np.load(C.OUT_DIR / "train_pairs.npy")
    keep = ~cold[tp[:, 1]]
    tp_cold = tp[keep]
    np.save(C.COLD_DIR / "train_pairs_cold.npy", tp_cold)

    # history: replace cold items with pad
    hist = np.load(C.OUT_DIR / "user_hist.npy").astype(np.int64)
    is_cold_hist = (hist < n_items) & cold[np.clip(hist, 0, n_items - 1)]
    hist_cold = hist.copy()
    hist_cold[is_cold_hist] = n_items
    np.save(C.COLD_DIR / "user_hist_cold.npy", hist_cold.astype(np.int32))

    # popularity over cold-holdout training targets (for logQ)
    pop = np.zeros(n_items, dtype=np.int64)
    u, c = np.unique(tp_cold[:, 1], return_counts=True)
    pop[u] = c
    np.save(C.COLD_DIR / "popularity_cold.npy", pop)

    # de-leak: a truly-cold item is known only by CONTENT + intrinsic metadata, NOT by its
    # usage-derived stats. Neutralize cold rows' usage features so absolute cold numbers are honest.
    # item_num cols 0,1,2 = z_avg_rating, z_log_ratings, z_log_textrev -> 0 (=z-mean); item_cat col 2 = pop_tier -> 0.
    item_num = np.load(C.OUT_DIR / "item_num.npy").copy()
    item_num[cold, 0:3] = 0.0
    np.save(C.COLD_DIR / "item_num_cold.npy", item_num)
    item_cat = np.load(C.OUT_DIR / "item_cat.npy").copy()
    item_cat[cold, 2] = 0
    np.save(C.COLD_DIR / "item_cat_cold.npy", item_cat)

    # how much of the test set lands on cold items (the cold-item eval slice size)
    test = np.load(C.OUT_DIR / "test.npy")
    n_cold_test = int(cold[test[:, 1]].sum())

    meta_cold = dict(meta)
    meta_cold.update({
        "cold_frac": C.COLD_FRAC, "cold_seed": C.COLD_SEED, "n_cold_items": int(cold.sum()),
        "d_content": C.D_CONTENT, "n_train_pairs_cold": int(len(tp_cold)),
        "n_cold_test_targets": n_cold_test, "n_warm_test_targets": int(len(test) - n_cold_test),
        "hist_cold_dropped": int(is_cold_hist.sum()),
    })
    json.dump(meta_cold, open(C.COLD_DIR / "meta_cold.json", "w"), indent=2)
    print(f"cold items: {cold.sum():,}/{n_items:,} ({cold.mean():.1%})")
    print(f"train pairs: {len(tp):,} -> {len(tp_cold):,} (dropped {len(tp)-len(tp_cold):,} cold targets)")
    print(f"history cells dropped (cold): {is_cold_hist.sum():,}")
    print(f"TEST split -> cold targets={n_cold_test:,}  warm targets={len(test)-n_cold_test:,}")
    print(f"wrote {C.COLD_DIR}")


if __name__ == "__main__":
    main()
