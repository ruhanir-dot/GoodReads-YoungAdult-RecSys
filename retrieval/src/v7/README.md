# retrieval/src/v7 — Multi-channel recall (6 channels + RRF)

v7 = implements the multi-channel recall that v6 pointed to as its next step. Shares the v6
leave-k-out base (`data/v6/base`) + the leakage-free set-based eval.

## 6 channels
two_tower (v6 ckpt) · mf (SVD-MF) · itemknn (co-occurrence i2i) · content (bge FAISS) ·
series_author (book_series/authors inverted index) · popularity. **Merged with RRF (c=60).**

## Conclusions
- Merged R@100 0.402 → **0.437** (+8.7%, concentrated at small K).
- **Specialist channels are 3–6× stronger on the tail** (series_author 0.046 / content 0.031 / itemknn 0.023 vs two_tower 0.008).
- **⚠ Naive global RRF is head-dominated**: merged tail 0.017 < specialist 0.046 → the next step needs **tiered routing / quotas / a learned fusion**.

## Run
```bash
cd retrieval/src/v7 && source ./_env.sh
$PY build_channels.py                                   # MF/itemknn/content/rules artifacts -> data/v7
CUDA_VISIBLE_DEVICES=$(best_gpu) $PY v7_recall.py        # 6 channels + RRF + candidate-pool eval -> data/v7/v7_results.json
```

## Modules
| File | Role |
|---|---|
| `build_channels.py` | MF (SVD) + itemknn co-occurrence + content FAISS + series/author inverted index |
| `v7_recall.py` | per-channel top-1000 → RRF merge → per-channel/merged set-Recall + contribution + slices |
| `model.py` | reuses the v6 TwoTowerV6 (loads the two_tower ckpt) |

## Notes
- Eval runs on 50k randomly sampled eval users (set-Recall is stable; aggregating over the full sparse 222k is too heavy).
- The key open problem has shifted from "single-tower modeling" to "**multi-channel fusion**".
