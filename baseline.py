import numpy as np
import pandas as pd
from src.io import save_predictions

# ---------- helpers (from src/eval.py) ----------

def _dcg_at_k(hits: np.ndarray, k: int) -> float:
    h = hits[:k]
    if h.sum() == 0:
        return 0.0
    return float(np.sum(h / np.log2(np.arange(len(h)) + 2)))


def _ndcg_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    dcg = _dcg_at_k(hits, k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(min(n_relevant, k)) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def _ap_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    h = hits[:k]
    if h.sum() == 0:
        return 0.0
    precision_at_i = np.cumsum(h) / np.arange(1, len(h) + 1)
    return float(np.sum(precision_at_i * h) / min(n_relevant, k))


def evaluate(recs, interactions, split='test', ks=(5, 10, 20), positive_only=True):
    """If positive_only=True, only rating>=4 interactions count as ground
    truth. Matches the eval used by multi_stage.py / multi_stage_llm.py so
    all models are comparable."""
    sub = interactions[interactions['split'] == split]
    if positive_only:
        sub = sub[sub['rating'] >= 4]
    truth = (
        sub.groupby('dense_user_id')['dense_book_id']
        .apply(set)
        .to_dict()
    )
    preds = (
        recs.sort_values(['user_id_dense', 'rank'])
        .groupby('user_id_dense')['book_id_dense']
        .apply(list)
        .to_dict()
    )
    ks = sorted(ks)
    kmax = max(ks)
    sums = {f"{m}@{k}": 0.0 for m in ("recall", "precision", "ndcg", "map") for k in ks}
    # Only count users we actually generated predictions for.
    scored_users = set(preds.keys()) & set(truth.keys())
    n_users = len(scored_users)
    if n_users == 0:
        return {k: 0.0 for k in sums} | {"n_eval_users": 0}
    for user_id in scored_users:
        gt = truth[user_id]
        ranked = preds[user_id]
        hits = np.zeros(kmax, dtype=np.float64)
        for i, item_id in enumerate(ranked[:kmax]):
            if item_id in gt:
                hits[i] = 1.0
        n_rel = len(gt)
        for k in ks:
            n_hits = float(hits[:k].sum())
            sums[f"recall@{k}"]    += n_hits / n_rel
            sums[f"precision@{k}"] += n_hits / k
            sums[f"ndcg@{k}"]      += _ndcg_at_k(hits, n_rel, k)
            sums[f"map@{k}"]       += _ap_at_k(hits, n_rel, k)
    return {k: v / n_users for k, v in sums.items()} | {"n_eval_users": n_users}


# ---------- baseline rankers ----------

def popularity_stats(train_df):
    """Per-book stats computed from the train split only (no leakage)."""
    stats = train_df.groupby('dense_book_id').agg(
        n_positive=('is_positive', 'sum'),
        n_rated=('rating', lambda r: (r > 0).sum()),
        sum_rating=('rating', lambda r: r[r > 0].sum()),
    ).reset_index()
    stats['avg_rating'] = np.where(
        stats['n_rated'] > 0, stats['sum_rating'] / stats['n_rated'], 0.0
    )
    return stats


def global_popularity(stats, k=20):
    return stats.sort_values('n_positive', ascending=False).head(k)['dense_book_id'].to_numpy()


def bayesian_rating(stats, k=20, prior_count=50):
    rated = stats[stats['n_rated'] > 0]
    C = rated['sum_rating'].sum() / rated['n_rated'].sum()
    m = prior_count
    s = stats.copy()
    s['score'] = (s['n_rated'] * s['avg_rating'] + m * C) / (s['n_rated'] + m)
    return s.sort_values('score', ascending=False).head(k)['dense_book_id'].to_numpy()


def weighted_pop_rating(stats, k=20):
    s = stats.copy()
    s['score'] = np.log1p(s['n_positive']) * s['avg_rating']
    return s.sort_values('score', ascending=False).head(k)['dense_book_id'].to_numpy()


def build_recs(pool_items, user_items, n_users, k=20):
    """Apply the global ranking per user, dropping items they've already read in train."""
    rows = []
    for user in range(n_users):
        seen = user_items.get(user, set())
        kept = [int(item) for item in pool_items if item not in seen][:k]
        for rank, item_id in enumerate(kept):
            rows.append((user, item_id, rank, float(k - rank)))
    return pd.DataFrame(rows, columns=['user_id_dense', 'book_id_dense', 'rank', 'score'])


# ---------- preprocessing ----------

print("Loading data...")
books_core = pd.read_parquet('recsys_data_v1/parquet/books_core.parquet')
interactions_core = pd.read_parquet('recsys_data_v1/parquet/interactions_core.parquet')

interactions = interactions_core[interactions_core['is_read'] == True].copy()

interaction_counts = interactions.groupby('user_id').size()
users_to_remove = interaction_counts[interaction_counts <= 4].index
interactions = interactions[~interactions['user_id'].isin(users_to_remove)]

interactions['is_positive'] = interactions['rating'] >= 4

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
interactions.drop(columns='rn', inplace=True)

print(f"Users: {interactions['dense_user_id'].nunique()}, Books: {interactions['dense_book_id'].nunique()}")

# ---------- per-user train history + popularity stats ----------

train = interactions[interactions['split'] == 'train']
n_users = interactions['dense_user_id'].max() + 1

user_items = train.groupby('dense_user_id')['dense_book_id'].apply(set).to_dict()
stats = popularity_stats(train)
print(f"Train stats: {len(stats)} books, {len(user_items)} users")

# ---------- rank, score, eval ----------

K = 20
POOL = 200
RANKERS = {
    'popularity':      global_popularity,
    'bayesian_rating': bayesian_rating,
    'pop_x_rating':    weighted_pop_rating,
}

for name, fn in RANKERS.items():
    print(f"\nRunning baseline: {name}")
    pool = fn(stats, k=POOL)
    recs = build_recs(pool, user_items, n_users, k=K)

    results = evaluate(recs, interactions, split='test', ks=(5, 10, 20, 100))
    print(f"  Test results ({name}):")
    for metric, value in results.items():
        print(f"    {metric}: {value:.4f}" if isinstance(value, float) else f"    {metric}: {value}")

    save_predictions(
        recs.rename(columns={'user_id_dense': 'user_id', 'book_id_dense': 'item_id'}),
        f'baseline_{name}',
    )
