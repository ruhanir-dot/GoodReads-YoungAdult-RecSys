"""Ranking + rating metrics. One implementation, used by every notebook,
so the comparison table in 08 is honest.

All DataFrames here use the in-memory `user_id`/`item_id` schema.

Conventions (pinned, do not change once 01 has run):
- positions are 0-indexed internally; DCG uses log2(rank+2) so the top
  position contributes 1/log2(2) = 1.0
- IDCG is computed against the actual ground-truth set size for that
  user (capped at K)
- only users with >= 1 ground-truth item in the chosen split count
  toward the mean (eval_users)
- predictions are sorted by `rank` ascending; ties broken by `score` desc
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .data import Bundle


# ---------- per-user metric primitives ----------


def _dcg_at_k(hits: np.ndarray, k: int) -> float:
    """DCG with binary relevance. `hits` is a 0/1 array of length >= k."""
    h = hits[:k]
    if h.sum() == 0:
        return 0.0
    positions = np.arange(len(h))
    return float(np.sum(h / np.log2(positions + 2)))


def _ndcg_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    dcg = _dcg_at_k(hits, k)
    ideal = min(n_relevant, k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def _ap_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    h = hits[:k]
    if h.sum() == 0:
        return 0.0
    positions = np.arange(1, len(h) + 1)
    precision_at_i = np.cumsum(h) / positions
    return float(np.sum(precision_at_i * h) / min(n_relevant, k))


# ---------- aggregation over users ----------


def _truth_by_user(bundle: Bundle, split: str) -> Dict[int, set]:
    s = bundle.split
    sub = s.loc[s["split"] == split, ["user_id", "item_id"]]
    return {
        user_id: set(grp["item_id"].to_numpy().tolist())
        for user_id, grp in sub.groupby("user_id", sort=False)
    }


def _sorted_preds_by_user(pred_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    df = pred_df.sort_values(["user_id", "rank"], kind="stable")
    return {
        user_id: grp["item_id"].to_numpy()
        for user_id, grp in df.groupby("user_id", sort=False)
    }


def evaluate(
    pred_df: pd.DataFrame,
    bundle: Bundle,
    split: str = "test",
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[str, float]:
    """Compute Recall/Precision/NDCG/MAP at each k, averaged over users
    that have >=1 ground-truth item in `split`.

    `pred_df` must have columns (user_id, item_id, rank, score). Users
    missing from pred_df contribute 0 to every metric (i.e. the model
    failed to recommend for them) — this keeps coverage gaps honest.
    """
    truth = _truth_by_user(bundle, split)
    preds = _sorted_preds_by_user(pred_df)

    ks = sorted(set(int(k) for k in ks))
    kmax = max(ks)
    sums = {f"{m}@{k}": 0.0 for m in ("recall", "precision", "ndcg", "map") for k in ks}
    n_users = len(truth)

    for user_id, gt in truth.items():
        ranked = preds.get(user_id, np.empty(0, dtype=np.int64))
        # binary hit vector up to kmax
        hits = np.zeros(kmax, dtype=np.float64)
        for i, item_id in enumerate(ranked[:kmax]):
            if item_id in gt:
                hits[i] = 1.0
        n_rel = len(gt)
        for k in ks:
            n_hits = float(hits[:k].sum())
            sums[f"recall@{k}"] += n_hits / n_rel
            sums[f"precision@{k}"] += n_hits / k
            sums[f"ndcg@{k}"] += _ndcg_at_k(hits, n_rel, k)
            sums[f"map@{k}"] += _ap_at_k(hits, n_rel, k)

    if n_users == 0:
        return {k: 0.0 for k in sums}
    return {k: v / n_users for k, v in sums.items()} | {"n_eval_users": float(n_users)}


def evaluate_by_tier(
    pred_df: pd.DataFrame,
    bundle: Bundle,
    split: str = "test",
    ks: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Break out `evaluate()` by the popularity tier of each user's
    ground-truth item (head/mid/tail). Lets us show whether a model is
    just predicting head titles for everyone.

    Returns a long-format DataFrame: rows=(tier, metric, k), values=mean.
    """
    pop_tier = dict(zip(
        bundle.popularity["item_id"].to_numpy(),
        bundle.popularity["pop_tier"].to_numpy(),
    ))
    truth = _truth_by_user(bundle, split)
    preds = _sorted_preds_by_user(pred_df)
    ks = sorted(set(int(k) for k in ks))
    kmax = max(ks)

    # Bucket users by the tier of (any) ground-truth item.
    # With leave-last-out there is typically one test item per user.
    tier_buckets: Dict[str, List[int]] = {"head": [], "mid": [], "tail": []}
    for user_id, gt in truth.items():
        # tier = tier of the most-popular held-out item
        tiers = [pop_tier.get(i) for i in gt]
        tiers = [t for t in tiers if t is not None]
        if not tiers:
            continue
        # head dominates mid dominates tail when multiple
        order = {"head": 0, "mid": 1, "tail": 2}
        bucket = sorted(tiers, key=lambda t: order.get(t, 9))[0]
        tier_buckets[bucket].append(user_id)

    rows = []
    for tier, user_ids in tier_buckets.items():
        if not user_ids:
            for k in ks:
                for m in ("recall", "precision", "ndcg", "map"):
                    rows.append((tier, m, k, 0.0, 0))
            continue
        sums = {f"{m}@{k}": 0.0 for m in ("recall", "precision", "ndcg", "map") for k in ks}
        for user_id in user_ids:
            gt = truth[user_id]
            ranked = preds.get(user_id, np.empty(0, dtype=np.int64))
            hits = np.zeros(kmax, dtype=np.float64)
            for i, item_id in enumerate(ranked[:kmax]):
                if item_id in gt:
                    hits[i] = 1.0
            n_rel = len(gt)
            for k in ks:
                n_hits = float(hits[:k].sum())
                sums[f"recall@{k}"] += n_hits / n_rel
                sums[f"precision@{k}"] += n_hits / k
                sums[f"ndcg@{k}"] += _ndcg_at_k(hits, n_rel, k)
                sums[f"map@{k}"] += _ap_at_k(hits, n_rel, k)
        n = len(user_ids)
        for k in ks:
            for m in ("recall", "precision", "ndcg", "map"):
                rows.append((tier, m, k, sums[f"{m}@{k}"] / n, n))
    return pd.DataFrame(rows, columns=["tier", "metric", "k", "value", "n_users"])


# ---------- rating-prediction side task (ALS only) ----------


def rating_metrics(
    pred_df: pd.DataFrame,
    bundle: Bundle,
    split: str = "test",
    pred_col: str = "rating_pred",
) -> Dict[str, float]:
    """RMSE/MAE on the subset of held-out interactions that have rating>0.

    `pred_df` must contain (user_id, item_id, <pred_col>). Only ALS dumps
    this; other models call `evaluate()` only.
    """
    if pred_col not in pred_df.columns:
        raise ValueError(f"pred_df is missing column {pred_col!r}; "
                         f"only the ALS rating-prediction side task should call this.")
    s = bundle.split
    held = s.loc[s["split"] == split, ["user_id", "item_id"]]
    ia = bundle.interactions_all
    if ia.empty:
        raise RuntimeError("Bundle was loaded with load_interactions=False; "
                           "rating_metrics needs the full interactions table.")
    truth = (
        held.merge(ia[["user_id", "item_id", "rating"]], on=["user_id", "item_id"], how="inner")
            .query("rating > 0")
    )
    joined = truth.merge(pred_df[["user_id", "item_id", pred_col]], on=["user_id", "item_id"], how="inner")
    if joined.empty:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0.0}
    err = joined[pred_col].to_numpy(dtype=np.float64) - joined["rating"].to_numpy(dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "n": float(len(joined)),
    }