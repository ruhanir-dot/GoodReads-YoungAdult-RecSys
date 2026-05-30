import pandas as pd
import numpy as np

from src.io import save_predictions
from scipy.sparse import csr_matrix
from itertools import combinations


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


# ------ item2item ------
print('Loading Data...')
interactions = pd.read_parquet('parquet/parquet/interactions_core.parquet',
                               columns=['user_id', 'book_id', 'is_read', 'date_updated', 'rating'])
interactions = interactions[interactions['is_read'] == True].copy()

print('Cleaning Data...')
# k-core filter
interaction_counts = interactions.groupby('user_id').size()
users_to_remove = interaction_counts[interaction_counts <= 4].index
interactions = interactions[~interactions['user_id'].isin(users_to_remove)]

# dense IDs
user_id_map = {uid: i for i, uid in enumerate(interactions['user_id'].unique())}
interactions['dense_user_id'] = interactions['user_id'].map(user_id_map)


books_core = pd.read_parquet('parquet/parquet/books_core.parquet')
books_core['dense_id'] = range(len(books_core))

book_id_map = dict(zip(books_core['book_id'], books_core['dense_id']))
interactions['dense_book_id'] = interactions['book_id'].map(book_id_map)
interactions = interactions.dropna(subset=['dense_book_id'])
interactions['dense_book_id'] = interactions['dense_book_id'].astype(int)

# split

print('Train-Test Split...')
interactions = interactions.sort_values(['dense_user_id', 'date_updated'])
interactions['rn'] = interactions.groupby('dense_user_id').cumcount(ascending=False)
interactions['split'] = 'train'
interactions.loc[interactions['rn'] == 1, 'split'] = 'val'
interactions.loc[interactions['rn'] == 0, 'split'] = 'test'
interactions.drop(columns='rn', inplace=True)

train = interactions[interactions['split'] == 'train'][['dense_user_id', 'dense_book_id']]

user_items = train.groupby('dense_user_id')['dense_book_id'].apply(list).to_dict() # grouping
n_users = interactions['dense_user_id'].max() + 1
n_books = interactions['dense_book_id'].max() + 1




# cap to last 5 items per user
cap = 5
user_items_capped = {k: v[-cap:] for k, v in user_items.items()}

# build sparse item-item similarity matrix (Swing weighted)
print('Building matrix....')
co_i, co_j, co_v = [], [], []
for items in user_items_capped.values():
    weight = 1 / (len(items) + 5)
    for i, j in combinations(items, 2):
        co_i += [i, j]
        co_j += [j, i]
        co_v += [weight, weight]

S = csr_matrix(
    (np.array(co_v, dtype=np.float32), 
     (np.array(co_i, dtype=np.int32), np.array(co_j, dtype=np.int32))),
    shape=(n_books, n_books)
)


# build query matrix — 1 at each user's capped train items
q_rows, q_cols = [], []
for user_id, items in user_items_capped.items():
    for item in items:
        q_rows.append(user_id)
        q_cols.append(item)

Q = csr_matrix((np.ones(len(q_rows)), (q_rows, q_cols)), shape=(n_users, n_books))

batch_size = 10000
rows = []

print('Training...')
for batch_start in range(0, n_users, batch_size):
    batch_end = min(batch_start + batch_size, n_users)
    Q_batch = Q[batch_start:batch_end]
    scores_batch = (Q_batch @ S).tocsr()
    
    for local_id in range(scores_batch.shape[0]):
        user_id = batch_start + local_id
        row = scores_batch.getrow(local_id)
        if row.nnz == 0:
            continue
        seen = set(user_items.get(user_id, []))
        items_arr = row.indices
        scores_arr = row.data
        mask = np.array([i not in seen for i in items_arr])
        items_arr = items_arr[mask]
        scores_arr = scores_arr[mask]
        if len(items_arr) == 0:
            continue
        top_k = min(20, len(items_arr))
        top_idx = np.argpartition(scores_arr, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores_arr[top_idx])[::-1]]
        for rank, idx in enumerate(top_idx):
            rows.append((user_id, int(items_arr[idx]), rank, float(scores_arr[idx])))

recs = pd.DataFrame(rows, columns=['user_id_dense', 'book_id_dense', 'rank', 'score'])

save_predictions(recs.rename(columns={'user_id_dense': 'user_id', 'book_id_dense': 'item_id'}), 'item2item')

print('Testing metrics...')
results = evaluate(recs, interactions, split='test', ks=(5, 10, 20, 100))
print("\nTest results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

