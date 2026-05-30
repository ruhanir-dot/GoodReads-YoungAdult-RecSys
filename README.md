# GoodReads Young Adult RecSys

A book recommender system built on the Goodreads Young Adult dataset. The
pipeline turns raw NDJSON dumps into a bundle of features, embeddings, and
LLM-derived profiles, and runs six recommenders against a shared evaluation
harness so they can be compared on the same held-out users with the same
metric definition.

## Models

| Script | Model | Notes |
|---|---|---|
| [baseline.py](baseline.py) | three popularity baselines | global popularity, Bayesian-shrunk avg rating, weighted pop × rating |
| [als.py](als.py) | ALS matrix factorization | `implicit.als`, 64 factors, ratings as confidence |
| [content_based.py](content_based.py) | content-based (TF-IDF over shelves) | cosine similarity between user history and item profiles |
| [item2item.py](item2item.py) | item-item Swing similarity | sparse co-occurrence matrix, capped history |
| [multi_stage.py](multi_stage.py) | two-stage retrieval + ranking | TF-IDF + ItemKNN + ALS + BPR retrieval (CEM-tuned weights) → LightGBM LambdaRank |
| [multi_stage_llm.py](multi_stage_llm.py) | multi-stage + LLM features | adds two LLM cosine features via [llm_features.py](llm_features.py) |

All six write predictions to `predictions/{name}_topk.parquet` and share the
same evaluation function: **Recall/Precision/NDCG/MAP @ {5, 10, 20}**, computed
over **rating ≥ 4** ground-truth test interactions only.

## Evaluation conventions

Every model uses the same `evaluate()` semantics so the numbers are
directly comparable:

- **Truth filter**: `rating >= 4` interactions in the test split only. We do
  not credit the model for "predicting" a book the user read and disliked.
- **Scored-users only**: metrics average over the intersection of
  `truth_users` and `predicted_users`, so subsampled runs aren't deflated by
  zero-prediction users.
- **Leave-last-out split**: each user's last interaction by `date_updated`
  is held out as `test`, second-to-last as `val`, the rest as `train`. K-core
  filter keeps users with more than 4 interactions.

Run [compare_all.py](compare_all.py) after training to print all models'
metrics on the same scored-user set.

## Repository layout

`recsys_data_v1/` and `data/` directories are user-provided (gitignored).

```
.
├── data/                                 # raw Goodreads NDJSON dumps (*.json gitignored)
├── recsys_data_v1/
│   ├── parquet/                          # typed columnar view of the raw JSON, what our final models use
│   │   ├── books_core.parquet            # 93,398 book versions
│   │   ├── interactions_core.parquet     # full interactions (~35M rows, 1.2 GB)
│   │   ├── book_authors.parquet
│   │   ├── book_shelves.parquet
│   │   └── book_series.parquet
│   └── preprocessed_v1/                  # the modeling-ready bundle (see BUNDLE_REFERENCE.md)
│       ├── id_maps/                      # uid_map, book_iid_map
│       ├── interactions/                 # interactions_all, split, popularity
│       ├── features/                     # book_features, user_features, item_tags, user_tags
│       ├── embeddings/                   # desc_emb, user_content_emb, item/user_profile_emb, faiss_desc
│       ├── profiles/                     # raw LLM JSONL output (item/user profiles)
│       ├── vocab/                        # tag_vocab, categorical_vocabs, cw/dealbreaker/style vocabs
│       └── manifest.json                 # SHA + row counts + LLM coverage stats
├── src/                                  # shared library used by every notebook
│   ├── __init__.py                       # re-exports the public API
│   ├── data.py                           # Bundle dataclass + load_bundle()
│   ├── splits.py                         # train/val/test selectors + eval_users
│   ├── candidates.py                     # filter_seen, popularity_fallback, merge_topk
│   ├── eval.py                           # Recall/Precision/NDCG/MAP @ K, by-tier eval
│   └── io.py                             # save_predictions / load_predictions
├── notebooks/
│   ├── 00_src_usage.ipynb                # tour of the src/ API on example notebook branch
├── data_exploration.ipynb
├── multi_stage_recsys.ipynb              # reference notebook the multi_stage scripts were adapted from
├── baseline.py                           # three popularity baselines
├── als.py                                # ALS matrix factorization
├── content_based.py                      # TF-IDF content-based
├── item2item.py                          # item-item Swing similarity
├── multi_stage.py                        # multi-channel retrieval + LightGBM ranker
├── multi_stage_llm.py                    # multi_stage + LLM profile features
├── llm_features.py                       # LLMFeatures class (used by multi_stage_llm.py)
├── compare_all.py                        # post-hoc unified comparison across models
├── predictions/                          # per-model {name}_topk.parquet dumps
├── BUNDLE_REFERENCE.md                   # full reference for every file in the bundle
└── requirements.txt
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Need only:
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyarrow` (core)
- `implicit` (ALS + BPR)
- `lightgbm` (ranker)


### Run a model

Each script is self-contained — just `python <script>.py`. Predictions land
in `predictions/{name}_topk.parquet` and metrics print to stdout.

```bash
python baseline.py        # ~1 min   (writes 3 baseline files)
python als.py             # ~5 min
python content_based.py   # ~10 min
python item2item.py       # ~15 min
python multi_stage.py     # ~15 min at smoke settings, ~80 min at full
python multi_stage_llm.py # same as multi_stage + ~2 min for LLM features
```

### Tunables (multi_stage.py / multi_stage_llm.py)

Both scripts have two scale knobs near the top:

```python
CEM_USERS      = 2000     # users sampled for CEM tuning + ranker labels
MAX_TEST_USERS = 5000     # users scored at final eval (None for all)
```

Smoke (2000 / 5000) finishes in ~15 min. Full scale (5000 / 100000) takes
~80 min. The chunked test-scoring loop keeps memory bounded regardless of
`MAX_TEST_USERS`.

### Compare all models

After running the scripts you care about:

```bash
python compare_all.py
```

This re-scores every saved prediction file against the same user set
(`multi_stage_llm`'s sample, restricted to `rating >= 4` test items) so you
get an apples-to-apples table.

## Data

None of the data is in the actual repo but repo is set up to show what
structure should look like for everything to work.
[BUNDLE_REFERENCE.md](BUNDLE_REFERENCE.md) documents every file in the
bundle: schema, origin (raw vs derived vs LLM), how it was made, and how to
use it downstream.

## Conventions

- **In-memory ID space.** Each script builds its own `dense_user_id` /
  `dense_book_id` from the raw string IDs after k-core filtering. The bundle
  also has its own `uid` / `iid` integer space; `llm_features.py` bridges
  between them via `recsys_data_v1/preprocessed_v1/id_maps/`.
- **Predictions schema.** Every model writes
  `predictions/{name}_topk.parquet` with columns
  `(user_id, item_id, rank, score)`. `rank` is 0-indexed.
- **Eval set.** Recall/NDCG averaged only over users whose held-out test
  item has `rating >= 4`. Set `positive_only=False` in any `evaluate()` call
  to use the unfiltered truth instead.
