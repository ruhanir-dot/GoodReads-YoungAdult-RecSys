import sys
import time
from pathlib import Path

# Resolve project root before any project-local imports.
# On Colab the project lives at /content/drive/MyDrive/GoodReads-YoungAdult-RecSys
# (mount Drive in a notebook cell before `!python multi_stage.py`).
if 'google.colab' in sys.modules:
    PROJECT_ROOT = Path('/content/drive/MyDrive/GoodReads-YoungAdult-RecSys')
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Expected project at {PROJECT_ROOT}. "
            "Mount Drive first with `from google.colab import drive; drive.mount('/content/drive')`."
        )
    sys.path.insert(0, str(PROJECT_ROOT))   # so `from src.io import ...` resolves
else:
    PROJECT_ROOT = Path.cwd()

PARQUET_DIR = PROJECT_ROOT / 'recsys_data_v1' / 'parquet'

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
import lightgbm as lgb

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
    truth. This matches the ranker's training signal — recommending a book
    the user read but rated <=3 isn't a 'hit', so it shouldn't deflate recall."""
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
    # Only score users we actually generated predictions for. Users without
    # predictions (e.g. when MAX_TEST_USERS subsamples) would otherwise
    # contribute zeros and deflate the metric.
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


# ---------- config ----------
# scale knobs: GoodReads has ~380K kcore users vs Steam's 10K, so we subsample
# users for CEM tuning and ranker training/eval. Crank these up for a fuller run.
SEED = 42
TOP_N_PER_CHANNEL = 400
L_TOTAL = 400                # candidate pool size per user
FINAL_TOP = 20               # top-K predictions kept per user (matches als.py)
ALS_FACTORS, ALS_ITER, ALS_REG = 64, 15, 0.05
BPR_FACTORS, BPR_ITER, BPR_LR = 64, 100, 0.01
CEM_USERS = 2000             # users sampled for CEM tuning + ranker labels
MAX_TEST_USERS = 5000        # users scored at final eval (None for all)
np.random.seed(SEED)

# GPU: only `implicit`'s ALS + BPR benefit (T4 on Colab). Auto-detect; the
# rest of the pipeline (TF-IDF, ItemKNN, LightGBM, retrieval loops) is CPU.
try:
    import implicit
    USE_GPU = bool(getattr(implicit.gpu, 'HAS_CUDA', False))
except Exception:
    USE_GPU = False
print(f"implicit GPU available: {USE_GPU}")


# ---------- preprocessing (same skeleton as als.py) ----------

print("Loading data...")
print(f"  project root: {PROJECT_ROOT}")
print(f"  parquet dir : {PARQUET_DIR}")

books_core   = pd.read_parquet(PARQUET_DIR / 'books_core.parquet')
book_authors = pd.read_parquet(PARQUET_DIR / 'book_authors.parquet')
book_shelves = pd.read_parquet(PARQUET_DIR / 'book_shelves.parquet')

# Prefer the slim parquet (~150 MB, is_read already filtered) when present,
# otherwise fall back to the full 1.2 GB file and apply the filter here.
slim_path = PARQUET_DIR / 'interactions_slim.parquet'
if slim_path.exists():
    interactions = pd.read_parquet(slim_path)
else:
    interactions_core = pd.read_parquet(PARQUET_DIR / 'interactions_core.parquet')
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

n_users = int(interactions['dense_user_id'].max() + 1)
n_books = len(books_core)
print(f"Users: {n_users:,}  Books: {n_books:,}")


# ---------- item metadata preprocessing ----------

# Shelves: text corpus for TF-IDF (top-30 shelves per book by shelf_count)
print("Building item metadata...")
shelves_sorted = book_shelves.sort_values(['book_id', 'shelf_count'], ascending=[True, False])
shelves_per_book = (
    shelves_sorted.groupby('book_id')['shelf_name']
    .apply(lambda x: ' | '.join(x.head(30)))
    .to_dict()
)
books_core['shelf_text'] = books_core['book_id'].map(shelves_per_book).fillna('')

books_core = books_core.sort_values('dense_id').reset_index(drop=True)
text_for_tfidf = books_core['shelf_text'].values

# Primary author per book (role null/empty = primary author)
primary = book_authors[(book_authors['role'].isna()) | (book_authors['role'] == '')]
primary = primary.drop_duplicates(subset='book_id', keep='first')
book_author_map = dict(zip(primary['book_id'], primary['author_id']))
books_core['primary_author'] = books_core['book_id'].map(book_author_map).fillna('')

# Lookup arrays indexed by dense_id (so item_x[dense_id] is O(1))
item_publisher     = books_core['publisher'].fillna('').values
item_author        = books_core['primary_author'].values
item_year          = books_core['publication_year'].fillna(-1).values
item_avg_rating    = books_core['average_rating'].fillna(0).values
item_shelf_count   = books_core['popular_shelves_count'].fillna(0).values
item_authors_count = books_core['authors_count'].fillna(0).values

# Shelf sets per book (used for Jaccard overlap in interaction features)
book_shelves_set = book_shelves.groupby('book_id')['shelf_name'].apply(set).to_dict()
item_shelf_set = [book_shelves_set.get(bid, set()) for bid in books_core['book_id']]


# ---------- channels ----------

train = interactions[interactions['split'] == 'train']
val   = interactions[interactions['split'] == 'val']
test  = interactions[interactions['split'] == 'test']

print("\nChannel 1: TF-IDF over book shelves...")
tfidf_vec = TfidfVectorizer(
    token_pattern=r"[^|\s]+",
    lowercase=True,
    min_df=5,
    max_df=0.5,
    sublinear_tf=True,
    norm='l2',
)
item_tfidf = tfidf_vec.fit_transform(text_for_tfidf)
print(f"  TF-IDF matrix: {item_tfidf.shape}")


def tfidf_retrieve(history_dense_ids, top_n=TOP_N_PER_CHANNEL):
    if len(history_dense_ids) == 0:
        return []
    u = np.asarray(item_tfidf[history_dense_ids].mean(axis=0))   # user profile = mean of read items
    scores = (item_tfidf @ u.T).ravel()
    scores[history_dense_ids] = -np.inf                           # seen-mask
    top_idx = np.argpartition(-scores, min(top_n, len(scores) - 1))[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]


print("\nChannel 2: ALS Matrix Factorization...")
als_mat = csr_matrix(
    (train['rating'].values.astype(np.float32),
     (train['dense_user_id'].values, train['dense_book_id'].values)),
    shape=(n_users, n_books),
)
als_model = AlternatingLeastSquares(
    factors=ALS_FACTORS, regularization=ALS_REG, iterations=ALS_ITER,
    use_gpu=USE_GPU, random_state=SEED,
)
als_model.fit(als_mat, show_progress=False)
# .to_numpy() handles both GPU (CuPy-backed) and CPU paths; np.asarray works as fallback.
als_user_factors = als_model.user_factors.to_numpy() if hasattr(als_model.user_factors, 'to_numpy') else np.asarray(als_model.user_factors)
als_item_factors = als_model.item_factors.to_numpy() if hasattr(als_model.item_factors, 'to_numpy') else np.asarray(als_model.item_factors)


def als_retrieve(user_id, history_dense_ids, top_n=TOP_N_PER_CHANNEL):
    scores = als_item_factors @ als_user_factors[user_id]
    scores[history_dense_ids] = -np.inf
    top_idx = np.argpartition(-scores, min(top_n, len(scores) - 1))[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]


print("\nChannel 3: BPR (Bayesian Personalized Ranking)...")
bpr_mat = csr_matrix(
    (np.ones(len(train), dtype=np.float32),
     (train['dense_user_id'].values, train['dense_book_id'].values)),
    shape=(n_users, n_books),
)
bpr_model = BayesianPersonalizedRanking(
    factors=BPR_FACTORS, learning_rate=BPR_LR, iterations=BPR_ITER,
    use_gpu=USE_GPU, random_state=SEED,
)
bpr_model.fit(bpr_mat, show_progress=False)
bpr_user_factors = bpr_model.user_factors.to_numpy() if hasattr(bpr_model.user_factors, 'to_numpy') else np.asarray(bpr_model.user_factors)
bpr_item_factors = bpr_model.item_factors.to_numpy() if hasattr(bpr_model.item_factors, 'to_numpy') else np.asarray(bpr_model.item_factors)


def bpr_retrieve(user_id, history_dense_ids, top_n=TOP_N_PER_CHANNEL):
    scores = bpr_item_factors @ bpr_user_factors[user_id]
    scores[history_dense_ids] = -np.inf
    top_idx = np.argpartition(-scores, min(top_n, len(scores) - 1))[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]


print("\nChannel 4: ItemKNN (item-item cosine on user-item matrix)...")
itemknn_user_item = csr_matrix(
    (np.log1p(train['rating'].values.astype(np.float32)),
     (train['dense_user_id'].values, train['dense_book_id'].values)),
    shape=(n_users, n_books),
)
item_vecs = normalize(itemknn_user_item.T.tocsr(), norm='l2', axis=1)


def itemknn_retrieve(history_dense_ids, top_n=TOP_N_PER_CHANNEL):
    if len(history_dense_ids) == 0:
        return []
    pooled = item_vecs[history_dense_ids].sum(axis=0).T
    scores = np.asarray(item_vecs @ pooled).ravel()
    scores[history_dense_ids] = -np.inf
    top_idx = np.argpartition(-scores, min(top_n, len(scores) - 1))[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]


# ---------- per-user train histories ----------

train_user_items = (
    train.groupby('dense_user_id')['dense_book_id']
    .apply(lambda x: np.array(x, dtype=np.int64))
    .to_dict()
)


# ---------- CEM channel-weight tuning ----------

print("\nCEM channel-weight tuning (subsampled)...")
# Only count rating>=4 val interactions as positives — same definition the
# ranker is trained against and the test eval uses.
val_pos = val[val['rating'] >= 4]
val_truth_dict = val_pos.groupby('dense_user_id')['dense_book_id'].apply(set).to_dict()
print(f"  val users with a positive (rating>=4) held-out item: {len(val_truth_dict):,} / {val['dense_user_id'].nunique():,}")

cem_rng = np.random.default_rng(SEED)
all_val_users = np.array(list(val_truth_dict.keys()))
cem_user_sample = cem_rng.choice(
    all_val_users, size=min(CEM_USERS, len(all_val_users)), replace=False
)

# Precompute each user's top-L_TOTAL list from every channel once so CEM can
# evaluate weight vectors by just slicing these lists per channel-budget.
print(f"  Precomputing channel lists for {len(cem_user_sample):,} users at depth {L_TOTAL}...")
t0 = time.time()
channel_lists = {}
for i, uid in enumerate(cem_user_sample):
    uid = int(uid)
    history = train_user_items.get(uid, np.array([], dtype=np.int64))
    channel_lists[uid] = {
        'tfidf':   [iid for iid, _ in tfidf_retrieve(history, L_TOTAL)],
        'itemknn': [iid for iid, _ in itemknn_retrieve(history, L_TOTAL)],
        'als':     [iid for iid, _ in als_retrieve(uid, history, L_TOTAL)],
        'bpr':     [iid for iid, _ in bpr_retrieve(uid, history, L_TOTAL)],
    }
    if (i + 1) % 500 == 0:
        print(f"    {i+1:,}/{len(cem_user_sample):,}  ({time.time()-t0:.1f}s)")

CHANNEL_ORDER = ['tfidf', 'itemknn', 'als', 'bpr']


def budgets_from_weights(weights, L=L_TOTAL):
    budgets = [int(round(w * L)) for w in weights]
    diff = L - sum(budgets)
    if diff != 0:
        budgets[int(np.argmax(weights))] += diff   # rounding fix-up
    return budgets


def eval_weights(weights):
    """Mean recall of the unioned top-budget candidate pool vs val truth."""
    budgets = budgets_from_weights(weights)
    recalls = []
    for uid in cem_user_sample:
        uid = int(uid)
        truth = val_truth_dict.get(uid, set())
        if not truth:
            continue
        lists = channel_lists.get(uid, {})
        pool = set()
        for channel, budget in zip(CHANNEL_ORDER, budgets):
            pool.update(lists.get(channel, [])[:budget])
        recalls.append(len(truth & pool) / len(truth))
    return float(np.mean(recalls))


baseline_recall = eval_weights(np.array([0.25] * 4))
print(f"  baseline (equal weight) Recall@{L_TOTAL}: {baseline_recall:.4f}")

# CEM loop: Dirichlet over the 4-channel simplex; sample, evaluate, keep elite, update.
alpha = np.ones(4)
Q, q_elite, eta, n_iters, patience = 50, 0.2, 0.5, 15, 4
best_score, best_alpha, patience_ct = -np.inf, alpha.copy(), 0
rng = np.random.default_rng(SEED)

for it in range(n_iters):
    samples = rng.dirichlet(alpha, size=Q)
    scores = np.array([eval_weights(w) for w in samples])
    n_elite = max(1, int(q_elite * Q))
    elite = samples[np.argsort(-scores)[:n_elite]]

    elite_mean = elite.mean(axis=0)
    elite_var = elite.var(axis=0).mean()
    scale = max((elite_mean[0] * (1 - elite_mean[0]) / max(elite_var, 1e-6)) - 1, 1.0)
    alpha = (1 - eta) * alpha + eta * elite_mean * scale

    iter_best = scores.max()
    print(f"  iter {it:2d}: mean={scores.mean():.4f}  best={iter_best:.4f}  "
          f"alpha=[tfidf={alpha[0]:.2f}, itemknn={alpha[1]:.2f}, als={alpha[2]:.2f}, bpr={alpha[3]:.2f}]")
    if iter_best > best_score + 1e-5:
        best_score, best_alpha, patience_ct = iter_best, alpha.copy(), 0
    else:
        patience_ct += 1
        if patience_ct >= patience:
            print(f"  early stop after {it+1} iters")
            break

weights_cem = best_alpha / best_alpha.sum()
budgets_cem = budgets_from_weights(weights_cem)
final_recall = eval_weights(weights_cem)
CEM_BUDGETS = dict(zip(CHANNEL_ORDER, budgets_cem))

print(f"\nCEM final weights (L={L_TOTAL}):")
for c, w, b in zip(CHANNEL_ORDER, weights_cem, budgets_cem):
    print(f"  {c:8s}  w={w:.3f}  budget={b}")
print(f"  Recall@{L_TOTAL}: {baseline_recall:.4f} (equal) -> {final_recall:.4f} (CEM) "
      f"({(final_recall - baseline_recall) / max(baseline_recall, 1e-9) * 100:+.1f}%)")


# ---------- candidate pool builder (used for ranker train + final scoring) ----------

def build_candidate_pool(user_ids, budgets):
    rows = []
    for i, uid in enumerate(user_ids):
        uid = int(uid)
        history = train_user_items.get(uid, np.array([], dtype=np.int64))
        per_item = {}
        if budgets.get('tfidf', 0) > 0:
            for iid, sc in tfidf_retrieve(history, budgets['tfidf']):
                per_item.setdefault(iid, {})['s_tfidf'] = sc
        if budgets.get('itemknn', 0) > 0:
            for iid, sc in itemknn_retrieve(history, budgets['itemknn']):
                per_item.setdefault(iid, {})['s_itemknn'] = sc
        if budgets.get('als', 0) > 0:
            for iid, sc in als_retrieve(uid, history, budgets['als']):
                per_item.setdefault(iid, {})['s_als'] = sc
        if budgets.get('bpr', 0) > 0:
            for iid, sc in bpr_retrieve(uid, history, budgets['bpr']):
                per_item.setdefault(iid, {})['s_bpr'] = sc
        for iid, sd in per_item.items():
            rows.append((
                uid, iid,
                sd.get('s_tfidf', 0.0),
                sd.get('s_itemknn', 0.0),
                sd.get('s_als', 0.0),
                sd.get('s_bpr', 0.0),
                len(sd),
            ))
        if (i + 1) % 2000 == 0:
            print(f"    pool: {i+1:,}/{len(user_ids):,}")
    return pd.DataFrame(rows, columns=[
        'dense_user_id', 'dense_book_id',
        's_tfidf', 's_itemknn', 's_als', 's_bpr', 'n_channels',
    ])


print(f"\nBuilding ranker-training pool for {len(cem_user_sample):,} users...")
t0 = time.time()
ranker_pool = build_candidate_pool(cem_user_sample, CEM_BUDGETS)
print(f"  done in {time.time()-t0:.1f}s. {len(ranker_pool):,} rows "
      f"({len(ranker_pool)/len(cem_user_sample):.1f}/user)")


# ---------- Retrieval recall @ retrieval step ----------

print("\nRetrieval Recall @ retrieval step (before ranker):")
pool_by_user = ranker_pool.groupby('dense_user_id')['dense_book_id'].apply(set).to_dict()
recalls = []
for uid in cem_user_sample:
    uid = int(uid)
    truth = val_truth_dict.get(uid, set())
    if not truth:
        continue
    pool = pool_by_user.get(uid, set())
    recalls.append(len(truth & pool) / len(truth))
avg_pool_size = ranker_pool.groupby('dense_user_id').size().mean()
print(f"  mean recall          : {np.mean(recalls):.4f}")
print(f"  median recall        : {np.median(recalls):.4f}")
print(f"  avg pool size per user: {avg_pool_size:.1f}")
print(f"  users with recall=0  : {sum(r == 0 for r in recalls):,} / {len(recalls):,}")
print(f"  users with recall=1  : {sum(r == 1 for r in recalls):,} / {len(recalls):,}")


# ---------- feature engineering ----------

print("\nFeature engineering...")

# User features
train_with_date = train[['dense_user_id', 'dense_book_id', 'rating', 'date_updated']].copy()
train_with_date['day'] = pd.to_datetime(train_with_date['date_updated']).dt.normalize()
user_feats = train_with_date.groupby('dense_user_id').agg(
    user_history_size=('dense_book_id', 'count'),
    user_total_rating=('rating', 'sum'),
    user_mean_rating=('rating', 'mean'),
    user_days_active=('day', lambda s: max((s.max() - s.min()).days + 1, 1)),
).reset_index()
user_feats['user_total_rating'] = np.log1p(user_feats['user_total_rating'])
user_feats['user_mean_rating']  = np.log1p(user_feats['user_mean_rating'])


def n_unique_shelves(items_read):
    s = set()
    for iid in items_read:
        s.update(item_shelf_set[iid])
    return len(s)


user_shelves = (
    train.groupby('dense_user_id')['dense_book_id']
    .apply(n_unique_shelves)
    .rename('user_unique_shelves')
    .reset_index()
)
user_feats = user_feats.merge(user_shelves, on='dense_user_id')

# Item features
item_pop = train.groupby('dense_book_id').agg(
    item_popularity=('dense_user_id', 'count'),
    item_mean_rating=('rating', 'mean'),
).reset_index()
item_pop['item_popularity']  = np.log1p(item_pop['item_popularity'])
item_pop['item_mean_rating'] = np.log1p(item_pop['item_mean_rating'].fillna(0))

item_feats = pd.DataFrame({
    'dense_book_id': np.arange(n_books),
    'item_release_year': item_year,
    'item_release_year_missing': (item_year == -1).astype(int),
    'item_avg_rating': item_avg_rating,
    'item_shelf_count': item_shelf_count,
    'item_authors_count': item_authors_count,
})
item_feats = item_feats.merge(item_pop, on='dense_book_id', how='left')
item_feats['item_popularity']  = item_feats['item_popularity'].fillna(0.0)
item_feats['item_mean_rating'] = item_feats['item_mean_rating'].fillna(0.0)
item_feats['item_is_cold']     = (item_feats['item_popularity'] == 0).astype(int)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def attach_interaction_features(pool_df, user_ids):
    """Per-user history summaries -> per-(user, candidate) features."""
    user_shelf_set = {}
    user_pub_set   = {}
    user_auth_set  = {}
    user_year_mean = {}
    for uid in user_ids:
        uid = int(uid)
        history = train_user_items.get(uid, np.array([], dtype=np.int64))
        shelves, pubs, auths, years = set(), set(), set(), []
        for iid in history:
            shelves.update(item_shelf_set[iid])
            p = item_publisher[iid]
            if p:
                pubs.add(p)
            a = item_author[iid]
            if a:
                auths.add(a)
            y = item_year[iid]
            if y != -1 and not np.isnan(y):
                years.append(y)
        user_shelf_set[uid] = shelves
        user_pub_set[uid]   = pubs
        user_auth_set[uid]  = auths
        user_year_mean[uid] = float(np.mean(years)) if years else 2014.0

    out = pool_df.copy()
    u_arr = out['dense_user_id'].values
    i_arr = out['dense_book_id'].values

    out['score_shelf_overlap'] = [
        jaccard(item_shelf_set[i], user_shelf_set.get(u, set()))
        for u, i in zip(u_arr, i_arr)
    ]
    out['score_pub_match'] = [
        int(item_publisher[i] in user_pub_set.get(u, set()))
        for u, i in zip(u_arr, i_arr)
    ]
    out['score_auth_match'] = [
        int(item_author[i] in user_auth_set.get(u, set()))
        for u, i in zip(u_arr, i_arr)
    ]
    out['score_year_distance'] = [
        abs((item_year[i] if (item_year[i] != -1 and not np.isnan(item_year[i]))
             else user_year_mean.get(u, 2014.0)) - user_year_mean.get(u, 2014.0))
        for u, i in zip(u_arr, i_arr)
    ]
    return out


print("  attaching interaction features...")
ranker_pool = attach_interaction_features(ranker_pool, cem_user_sample)
ranker_pool = ranker_pool.merge(user_feats, on='dense_user_id', how='left')
ranker_pool = ranker_pool.merge(item_feats, on='dense_book_id', how='left')

# Labels: candidate is positive iff it's in this user's val-held-out set.
val_pairs = set(zip(val_pos['dense_user_id'], val_pos['dense_book_id']))
ranker_pool['label'] = [
    int((u, i) in val_pairs)
    for u, i in zip(ranker_pool['dense_user_id'], ranker_pool['dense_book_id'])
]
print(f"  ranker_pool: {len(ranker_pool):,} rows  positives: {int(ranker_pool['label'].sum()):,}")


FEATURE_COLS = [
    's_tfidf', 's_itemknn', 's_als', 's_bpr', 'n_channels',
    'score_shelf_overlap', 'score_pub_match', 'score_auth_match', 'score_year_distance',
    'user_history_size', 'user_total_rating', 'user_mean_rating',
    'user_days_active', 'user_unique_shelves',
    'item_release_year', 'item_release_year_missing', 'item_avg_rating',
    'item_shelf_count', 'item_authors_count',
    'item_popularity', 'item_mean_rating', 'item_is_cold',
]


# ---------- Stage 2: LightGBM LambdaRank ----------

print("\nTraining LightGBM LambdaRank...")
unique_users = ranker_pool['dense_user_id'].unique()
split_rng = np.random.default_rng(SEED)
shuffled = split_rng.permutation(len(unique_users))
n_eval = max(1, int(0.2 * len(unique_users)))
eval_set  = set(unique_users[shuffled[:n_eval]])
train_set = set(unique_users[shuffled[n_eval:]])

t_df = (ranker_pool[ranker_pool['dense_user_id'].isin(train_set)]
        .sort_values('dense_user_id').reset_index(drop=True))
e_df = (ranker_pool[ranker_pool['dense_user_id'].isin(eval_set)]
        .sort_values('dense_user_id').reset_index(drop=True))

X_train, y_train = t_df[FEATURE_COLS].values, t_df['label'].values
group_train      = t_df.groupby('dense_user_id').size().values
X_eval,  y_eval  = e_df[FEATURE_COLS].values, e_df['label'].values
group_eval       = e_df.groupby('dense_user_id').size().values

print(f"  train: {len(t_df):,} rows / {len(group_train):,} users  ({int(y_train.sum()):,} positives)")
print(f"  eval : {len(e_df):,} rows / {len(group_eval):,} users  ({int(y_eval.sum()):,} positives)")

ranker = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=10,
    reg_lambda=0.0,
    random_state=SEED,
    seed=SEED,
    bagging_seed=SEED,
    feature_fraction_seed=SEED,
    data_random_seed=SEED,
    extra_seed=SEED,
    objective_seed=SEED,
    deterministic=True,
    force_col_wise=True,
    n_jobs=1,
    verbose=-1,
)

ranker.fit(
    X_train, y_train, group=group_train,
    eval_set=[(X_eval, y_eval)], eval_group=[group_eval], eval_at=[10],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
)
print(f"  best iteration: {ranker.best_iteration_}")

importance = (pd.DataFrame({'feature': FEATURE_COLS,
                            'importance': ranker.feature_importances_})
              .sort_values('importance', ascending=False).reset_index(drop=True))
print("\nFeature importance (top 10):")
print(importance.head(10).to_string(index=False))


# ---------- Final scoring on test split (chunked to bound memory) ----------
# Without chunking, the full test_pool can hit 30M+ rows and OOM the
# attach_interaction_features copy. Chunking keeps peak memory per batch
# under ~2 GB regardless of total test-user count.

test_user_ids = test['dense_user_id'].unique()
if MAX_TEST_USERS is not None and len(test_user_ids) > MAX_TEST_USERS:
    test_user_ids = cem_rng.choice(test_user_ids, size=MAX_TEST_USERS, replace=False)
print(f"\nScoring {len(test_user_ids):,} test users (chunked)...")

CHUNK = 10000
top_recs = []
t0 = time.time()
for start in range(0, len(test_user_ids), CHUNK):
    chunk_ids = test_user_ids[start:start + CHUNK]
    chunk_pool = build_candidate_pool(chunk_ids, CEM_BUDGETS)
    chunk_pool = attach_interaction_features(chunk_pool, chunk_ids)
    chunk_pool = chunk_pool.merge(user_feats, on='dense_user_id', how='left')
    chunk_pool = chunk_pool.merge(item_feats, on='dense_book_id', how='left')
    for col in FEATURE_COLS:
        if chunk_pool[col].isna().any():
            chunk_pool[col] = chunk_pool[col].fillna(0)
    chunk_pool['score_rank'] = ranker.predict(chunk_pool[FEATURE_COLS].values)
    chunk_pool['rank'] = (
        chunk_pool.groupby('dense_user_id')['score_rank']
        .rank(method='first', ascending=False).astype(int) - 1
    )
    chunk_top = (
        chunk_pool[chunk_pool['rank'] < FINAL_TOP]
        [['dense_user_id', 'dense_book_id', 'rank', 'score_rank']]
    )
    top_recs.append(chunk_top)
    del chunk_pool
    print(f"  scored {start + len(chunk_ids):,}/{len(test_user_ids):,}  ({time.time()-t0:.1f}s)")

recs = (
    pd.concat(top_recs, ignore_index=True)
    .sort_values(['dense_user_id', 'rank'])
    .rename(columns={
        'dense_user_id': 'user_id_dense',
        'dense_book_id': 'book_id_dense',
        'score_rank':    'score',
    })
    [['user_id_dense', 'book_id_dense', 'rank', 'score']]
)


# ---------- evaluation ----------

print("\nTest results:")
results = evaluate(recs, interactions, split='test', ks=(5, 10, 20, 100))
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\nStage breakdown:")
print(f"  Retrieval pool recall (val, CEM-tuned): {final_recall:.4f}")
print(f"  After ranker recall@10 (test):          {results['recall@10']:.4f}")
print(f"  After ranker ndcg@10  (test):           {results['ndcg@10']:.4f}")

save_predictions(
    recs.rename(columns={'user_id_dense': 'user_id', 'book_id_dense': 'item_id'}),
    'multi_stage',
)
