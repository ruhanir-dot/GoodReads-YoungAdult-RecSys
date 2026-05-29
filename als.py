import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


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


def evaluate(recs, interactions, split='test', ks=(5, 10, 20)):
    truth = (
        interactions[interactions['split'] == split]
        .groupby('dense_user_id')['dense_book_id']
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
    n_users = len(truth)
    for user_id, gt in truth.items():
        ranked = preds.get(user_id, [])
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


def build_recs(model, sparse_matrix, n=20):
    user_ids = np.arange(sparse_matrix.shape[0])
    item_ids_batch, scores_batch = model.recommend(
        user_ids, sparse_matrix, N=n, filter_already_liked_items=True
    )
    rows = []
    for user, (items, scores) in enumerate(zip(item_ids_batch, scores_batch)):
        for rank, (item_id, score) in enumerate(zip(items, scores)):
            rows.append((user, int(item_id), rank, float(score)))
    return pd.DataFrame(rows, columns=['user_id_dense', 'book_id_dense', 'rank', 'score'])


# ---------- preprocessing ----------

print("Loading data...")
books_core = pd.read_parquet('parquet/parquet/books_core.parquet')
interactions_core = pd.read_parquet('parquet/parquet/interactions_core.parquet')

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

# ---------- sparse matrix ----------

train = interactions[interactions['split'] == 'train']
n_users = interactions['dense_user_id'].max() + 1
n_books = interactions['dense_book_id'].max() + 1

sparse_matrix = csr_matrix(
    (train['rating'].values, (train['dense_user_id'].values, train['dense_book_id'].values)),
    shape=(n_users, n_books)
)
print(f"Sparse matrix: {sparse_matrix.shape}")

# ---------- train and eval on test ----------

print("\nTraining model...")
model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=10, random_state=42)
model.fit(sparse_matrix)
recs = build_recs(model, sparse_matrix, n=20)

results = evaluate(recs, interactions, split='test', ks=(5, 10, 20, 100))
print("\nTest results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
