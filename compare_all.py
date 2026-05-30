"""Score every model's saved predictions against the SAME user set
(the users multi_stage_llm scored) using the rating>=4 truth filter.
Run after the four training scripts finish."""

import numpy as np
import pandas as pd

# Rebuild interactions the same way the scripts do
books_core = pd.read_parquet('recsys_data_v1/parquet/books_core.parquet')
interactions = pd.read_parquet('recsys_data_v1/parquet/interactions_slim.parquet')
counts = interactions.groupby('user_id').size()
interactions = interactions[interactions['user_id'].isin(counts[counts > 4].index)]
user_id_map = {uid: i for i, uid in enumerate(interactions['user_id'].unique())}
interactions['dense_user_id'] = interactions['user_id'].map(user_id_map)
books_core['dense_id'] = range(len(books_core))
book_id_map = dict(zip(books_core['book_id'], books_core['dense_id']))
interactions['dense_book_id'] = interactions['book_id'].map(book_id_map)
interactions = interactions.dropna(subset=['dense_book_id'])
interactions['dense_book_id'] = interactions['dense_book_id'].astype(int)
interactions = interactions.sort_values(['dense_user_id', 'date_updated'])
interactions['rn'] = interactions.groupby('dense_user_id').cumcount(ascending=False)
interactions['split'] = 'train'
interactions.loc[interactions['rn'] == 1, 'split'] = 'val'
interactions.loc[interactions['rn'] == 0, 'split'] = 'test'

# rating >= 4 truth
test_pos = interactions[(interactions['split'] == 'test') & (interactions['rating'] >= 4)]
truth = test_pos.groupby('dense_user_id')['dense_book_id'].apply(set).to_dict()

# common user set = multi_stage_llm's scored users intersected with positive-test users
ms_users = set(pd.read_parquet('predictions/multi_stage_llm_topk.parquet')['user_id'].unique())
common = ms_users & set(truth.keys())
print(f"Comparison set: {len(common):,} users (multi_stage_llm's sample, filtered to rating>=4 test)")


def _dcg(h, k):
    h = h[:k]
    return 0.0 if h.sum() == 0 else float(np.sum(h / np.log2(np.arange(len(h)) + 2)))


def _ndcg(h, n, k):
    if n == 0: return 0.0
    dcg = _dcg(h, k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(min(n, k)) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def score(name, users):
    try:
        df = pd.read_parquet(f'predictions/{name}_topk.parquet')
    except Exception as e:
        return f"read error: {type(e).__name__}"
    preds = df.sort_values(['user_id', 'rank']).groupby('user_id')['item_id'].apply(list).to_dict()
    scored = users & set(preds.keys())
    if not scored:
        return None
    n = len(scored)
    sums = {f"{m}@{k}": 0.0 for m in ('recall', 'ndcg') for k in (5, 10, 20)}
    for u in scored:
        gt = truth[u]
        ranked = preds[u]
        hits = np.zeros(20)
        for i, iid in enumerate(ranked[:20]):
            if iid in gt: hits[i] = 1.0
        n_rel = len(gt)
        for k in (5, 10, 20):
            sums[f'recall@{k}'] += float(hits[:k].sum()) / n_rel
            sums[f'ndcg@{k}']   += _ndcg(hits, n_rel, k)
    return {k: v / n for k, v in sums.items()} | {'n_scored': n}


print(f"\n{'model':<28} {'n_scored':>10} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'NDCG@10':>10}")
print('-' * 78)
for name in [
    'baseline_popularity', 'baseline_bayesian_rating', 'baseline_pop_x_rating',
    'als', 'item2item', 'multi_stage', 'multi_stage_llm'
]:
    r = score(name, common)
    if isinstance(r, str):
        print(f"{name:<28} {r}")
        continue
    if r is None:
        print(f"{name:<28} (no overlap)")
        continue
    print(f"{name:<28} {r['n_scored']:>10,} {r['recall@5']:>8.4f} {r['recall@10']:>8.4f} {r['recall@20']:>8.4f} {r['ndcg@10']:>10.4f}")
