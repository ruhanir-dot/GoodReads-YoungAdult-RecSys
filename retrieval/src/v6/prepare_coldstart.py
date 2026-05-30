"""Cold-ITEM holdout prep for lever A (semantic ID). Holds ~COLD_FRAC of items fully out of
training (never a target, never in pooled history). Reuses the data/v1 base arrays; writes the
holdout-specific arrays to data/v5/cold/. The RQ-VAE sem_codes (data/v5/sem_codes.npy) cover ALL
items incl. cold ones, so semantic/hybrid id-modes can still represent cold items (the test).
Pure CPU.

  python prepare_coldstart.py
"""
from __future__ import annotations
import json
import numpy as np
import config as C


def main():
    C.COLD_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(C.BASE_DIR / "meta.json"))
    n_items, n_users = meta["n_items"], meta["n_users"]

    rng = np.random.RandomState(C.COLD_SEED)
    cold = np.zeros(n_items, bool)
    cold[rng.choice(n_items, int(round(C.COLD_FRAC * n_items)), replace=False)] = True
    np.save(C.COLD_DIR / "cold_items.npy", cold)

    id_lookup = np.arange(n_items, dtype=np.int64); id_lookup[cold] = n_items   # cold -> pad row
    np.save(C.COLD_DIR / "id_lookup.npy", id_lookup)

    tp = np.load(C.BASE_DIR / "train_pairs.npy")
    tp_cold = tp[~cold[tp[:, 1]]]
    np.save(C.COLD_DIR / "train_pairs_cold.npy", tp_cold)

    hist = np.load(C.BASE_DIR / "user_hist.npy").astype(np.int64)
    is_cold = (hist < n_items) & cold[np.clip(hist, 0, n_items - 1)]
    hist_cold = hist.copy(); hist_cold[is_cold] = n_items
    np.save(C.COLD_DIR / "user_hist_cold.npy", hist_cold.astype(np.int32))

    pop = np.zeros(n_items, np.int64)
    u, c = np.unique(tp_cold[:, 1], return_counts=True); pop[u] = c
    np.save(C.COLD_DIR / "popularity_cold.npy", pop)

    # de-leak cold rows' usage-derived features (honest absolute cold numbers)
    item_num = np.load(C.BASE_DIR / "item_num.npy").copy(); item_num[cold, 0:3] = 0.0
    np.save(C.COLD_DIR / "item_num_cold.npy", item_num)
    item_cat = np.load(C.BASE_DIR / "item_cat.npy").copy(); item_cat[cold, 2] = 0
    np.save(C.COLD_DIR / "item_cat_cold.npy", item_cat)

    test = np.load(C.BASE_DIR / "test.npy"); n_cold_test = int(cold[test[:, 1]].sum())
    meta_cold = dict(meta); meta_cold.update({
        "cold_frac": C.COLD_FRAC, "cold_seed": C.COLD_SEED, "n_cold_items": int(cold.sum()),
        "n_train_pairs_cold": int(len(tp_cold)), "n_cold_test_targets": n_cold_test,
        "n_warm_test_targets": int(len(test) - n_cold_test)})
    json.dump(meta_cold, open(C.COLD_DIR / "meta_cold.json", "w"), indent=2)
    print(f"cold items: {cold.sum():,}/{n_items:,} ({cold.mean():.1%}); "
          f"train pairs {len(tp):,}->{len(tp_cold):,}; cold test targets {n_cold_test:,}")
    print(f"wrote {C.COLD_DIR}")


if __name__ == "__main__":
    main()
