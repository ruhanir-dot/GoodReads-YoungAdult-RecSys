# v3 Profiling Report (User + Book)

> 2026-05-30 ｜ code `profiling/src/profile_v3/` ｜ output `profiling/data/v3/` ｜ model Qwen3.5-4B (transformers, 2×H200)

## 0. Results at a glance
| | Count | Errors | Notes |
|---|---|---|---|
| **User profiles** | **98,746** | **0 (0.000%)** | `{uid, language, like, dislike}` |
| **Book profiles** | **34,916** | **0** | `{iid, language, tags}` |
| **Vocab (top-3000)** | 3,000 / 227,280 distinct | — | covers **59.6%** of the 1,076,800 tag occurrences |

Zero parse errors end-to-end (after adding `repetition_penalty=1.2` to kill repetition loops +
`max_new=256` + JSON-salvage tolerant parsing).

## 1. Sampling
- **User**: **train-split** reviews only (no leakage); most recent by time, **≤20 positive (rating≥4) + ≤10 negative (rating≤2)**; each ≤800 chars → concatenated, then **tokenizer-truncated to 2048 tokens**. Threshold: ≥3 train reviews.
- **Book**: **title + description (≤3000) + language_code** only, **no reviews** (leakage-free by construction).

## 2. Field generation
- **User.language**: a code-level **langdetect frequency** normalized dict (not the LLM). 9,627 multilingual users. Main languages: en 91,511 · es 2,876 · id 1,012 · it/de/pt …
- **Book.language**: LLM output; the **11,275** books with a missing `language_code` have it inferred by the LLM from title/desc text (reviews strictly forbidden).
- **like / dislike / tags**: open vocabulary (unconstrained). like averages 6.7 items, tags average 6.3.

## 3. Vocab (top-3000)
Covers 59.6%; the long tail (remaining 224K terms) is dropped as specified. Top-20 examples:
`Emotional Depth, Coming-of-Age, Dystopian, Fast Paced Action, Romance, Series Continuity, Young Adult, Moral Ambiguity, Character Growth, Slow Pacing, Strong Female Protagonists, Predictable Plots, Morally Grey Characters, Slow Burn Romance, World Building, Mystery …`

## 4. Examples
```json
USER  {"uid": 5, "language": {"en": 1.0},
       "like": ["Dystopian","High Stakes","Fast Paced","Addictive Plot","Great Story"],
       "dislike": ["Romance Tropes","Predictable","Generic"]}
BOOK  {"iid": 1, "language": "en",
       "tags": ["Coming-of-Age","Teen Drama","Divorce","Family Conflict","Tragedy","Romance","Growth"]}
```

## 5. Leakage-safety
- User profiles use **only `split='train'` reviews** (enforced by SQL join); val/test reviews never enter.
- Book profiles use only static metadata (title/desc/language), **never reviews** → independent of any user's train/test boundary.
- Language inference: books inferred from their own text, users computed from the language frequency of their own reviews — neither crosses entities.

## 6. Known minor items (non-blocking)
- **empty-dislike 32.0%**: by design — many users never wrote a ≤2-star pan, so they naturally have no dislike.
- **empty-like 2.3% (2,317)**: a few users' reviews are too short/neutral to extract any like tags.
- **book.language: ~541 books use a 3-letter code ("eng" instead of "en")**: occasional LLM non-normalization; a one-line map (eng→en, fre→fr…) cleans it.

## 7. Output files (`profiling/data/v3/`)
`user_profiles.jsonl` (98,746) · `book_profiles.jsonl` (34,916) · `vocab.json` (top-3000).
Comparison reference: `profiling/data/v2/{user_profiles,item_profiles}.jsonl` (older version, kept for comparison).

> Note: the sampled-input parquet (`user_inputs.parquet` / `book_inputs.parquet`) and the per-shard run
> logs from the original run are intermediate/large and are **not** included in this PR — they're
> regenerable from the ingest parquet via `profile_v3/build_inputs.py`.
