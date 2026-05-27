# GoodReads Young Adult RecSys

A book recommender system built on the Goodreads Young Adult dataset. The
pipeline turns raw NDJSON dumps into a bundle of
features, embeddings, and LLM-derived profiles, and provides a shared
evaluation harness so multiple recommenders can be compared on the same
held-out users.

## Repository layout

```
.
├── data/                                 # raw Goodreads NDJSON dumps (*.json gitignored)
├── recsys_data_v1/
│   ├── processed/parquet/                # typed columnar view of the raw JSON (intermediate)
│   └── preprocessed_v1/                  # the modeling-ready bundle (see BUNDLE_REFERENCE.md)
│       ├── id_maps/                      # uid_map, book_iid_map
│       ├── interactions/                 # interactions_all, split, popularity
│       ├── features/                     # book_features, user_features, item_tags, user_tags
│       ├── embeddings/                   # desc_emb, user_content_emb, item/user_profile_emb, faiss_desc
│       ├── profiles/                     # raw LLM JSONL output (item/user profiles)
│       ├── vocab/                        # tag_vocab, categorical_vocabs, cw/dealbreaker/style vocabs
│       └── manifest.json                 # SHA + row counts + LLM coverage stats
├── src/                                  # shared library used by every modeling notebook
│   ├── __init__.py                       # re-exports the public API
│   ├── data.py                           # Bundle dataclass + load_bundle() (with optional LLM artifacts, path overrides, extras)
│   ├── splits.py                         # train / val / test selectors + eval_users + train_user_items
│   ├── candidates.py                     # filter_seen, popularity_fallback, merge_topk, topk_dataframe
│   ├── eval.py                           # Recall / Precision / NDCG / MAP @ K, by-tier eval, RMSE/MAE
│   └── io.py                             # save_predictions / load_predictions (fixed schema, dtype-validated)
├── notebooks/
│   └── 00_src_usage.ipynb                # tour of the src/ API
├── predictions/                          # per-model {name}_topk.parquet dumps
├── BUNDLE_REFERENCE.md                   # full reference for every file in the bundle
├── data_exploration.ipynb
├── data_exploration_filled.ipynb
└── requirements.txt
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use the bundle:

```python
import src

bundle = src.load_bundle()                 # loads features, embeddings, FAISS index, vocabs
train  = src.train_positives(bundle)       # (user_id, item_id) train pairs
users  = src.eval_users(bundle, "test")    # users with held-out items

# After your model produces topk predictions...
src.save_predictions(pred_df, name="als")
metrics = src.evaluate(pred_df, bundle, split="test", ks=(5, 10, 20))
```

See [notebooks/00_src_usage.ipynb](notebooks/00_src_usage.ipynb) for a walkthrough this will be on a seperate branch not on main.

## Data
None of the data is in the actual repo but repo is setup to show what structure should look like for everything to work. 
[BUNDLE_REFERENCE.md](BUNDLE_REFERENCE.md) documents every file in the
bundle: schema, origin (raw vs derived vs LLM), how it was made, and how
to use it downstream.

## Conventions

- **Join keys.** In-memory DataFrames use `user_id` / `item_id` (int32).
  The on-disk parquets still use `uid` / `iid`; the loader renames them.
- **Predictions schema.** Every model writes
  `predictions/{name}_topk.parquet` with columns
  `(user_id, item_id, rank, score)`. `rank` is 0-indexed.
- **Eval set.** Ranking metrics are averaged only over users with at
  least one held-out item in the chosen split (`src.eval_users`).