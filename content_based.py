import pandas as pd
import numpy as np
import faiss
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

# --- begin content based ----
print('Loading Data...')
books_core = pd.read_parquet('parquet/parquet/books_core.parquet')
books_core['text_to_embed'] = books_core['description'].fillna(books_core['title']) # fill nulls with title

# cleaning 

print('Cleaning Data...')
interactions = pd.read_parquet('parquet/parquet/interactions_core.parquet', 
                               columns=['user_id', 'book_id', 'is_read', 'date_updated', 'rating'])
interactions = interactions[interactions['is_read'] == True].copy()

# k-core filter
interaction_counts = interactions.groupby('user_id').size()
users_to_remove = interaction_counts[interaction_counts <= 4].index
interactions = interactions[~interactions['user_id'].isin(users_to_remove)]

# dense IDs
user_id_map = {uid: i for i, uid in enumerate(interactions['user_id'].unique())}
interactions['dense_user_id'] = interactions['user_id'].map(user_id_map)



books_core['dense_id'] = range(len(books_core))

book_id_map = dict(zip(books_core['book_id'], books_core['dense_id']))
interactions['dense_book_id'] = interactions['book_id'].map(book_id_map)
interactions = interactions.dropna(subset=['dense_book_id'])
interactions['dense_book_id'] = interactions['dense_book_id'].astype(int)

# split
print('Splitting Data...')
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

# loading npy
print('Loading embeddings...')
desc_emb = np.load('desc_emb.npy')

# user content emb
user_content_emb = np.load('user_content_emb.npy')

print('Cosine Similarity....')
# normalize for cosine similarity
desc_emb_norm = desc_emb.copy().astype(np.float32)
faiss.normalize_L2(desc_emb_norm)

# build index
index = faiss.IndexFlatIP(desc_emb_norm.shape[1])
index.add(desc_emb_norm)

# normalize user embeddings 
user_emb_norm = user_content_emb.copy()
faiss.normalize_L2(user_emb_norm)

print('Ranking...')
# top 25 per user
distances, indices = index.search(user_emb_norm, 25)

rows = []
for user_id in range(n_users):
    seen = set(user_items.get(user_id, []))
    rank = 0
    for item_id, score in zip(indices[user_id], distances[user_id]):
        if item_id in seen or item_id < 0:
            continue
        rows.append((user_id, int(item_id), rank, float(score)))
        rank += 1
        if rank == 20:
            break

recs = pd.DataFrame(rows, columns=['user_id_dense', 'book_id_dense', 'rank', 'score'])


save_predictions(recs.rename(columns={'user_id_dense': 'user_id', 'book_id_dense': 'item_id'}), 'content')


print('Testing metrics...')
results = evaluate(recs, interactions, split='test', ks=(5, 10, 20, 100))
print("\nTest results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")