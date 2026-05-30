# retrieval/src/v6 — leave-k-out + graded negatives + semantic-ID source comparison (self-contained)

v6 = the user-requested redesign. **Not directly comparable to v1–v5** (split / positive-negative
rules / eval protocol all changed). Code is self-contained; data is written to `data/v6/` (base reuses
v5's bge content, which is split-independent).

> env `a40` (nlplab2, 8×A40, MPS bypass). `_env.sh` is host-aware.

## Three changes
1. **split = leave-k-out (proportional, temporal)**: per user, hold out the most recent ~10% test / ~10% val / ~80% train (only evaluate `n≥10`). Eval switches to **set-based Recall@K**.
2. **graded negatives**: random-soft (in-batch) + explicit-soft (rating==3, λ0.5) + explicit-hard (rating≤2, λ2.0), leakage-free.
3. **semantic-ID source comparison**: M0 atomic / M1 content codes / M2 collaborative codes (co-occurrence SVD) / M3 large content codebook (K1024/L4).

## Conclusions
- **Semantic codes never help the warm catalog** at any source/capacity (M0 atomic is best, 0.4007).
- **Graded negatives are the only clean small gain** (N0 < N1 < N2).
- **The M2 collaborative-code cold-item gain is leakage** (doesn't count); the two-tower can't rescue cold items.
- **Next step: multi-channel recall.**

## Run
```bash
cd retrieval/src/v6 && source ./_env.sh
$PY prepare_data_v6.py && $PY build_negatives_v6.py && $PY build_collab_emb.py
for s in content collab big; do CUDA_VISIBLE_DEVICES=$(best_gpu) RECSYS_RQSRC=$s $PY rqvae.py; done
$PY prepare_coldstart.py
bash run_v6_sweep.sh && $PY aggregate_v6.py
```

## Modules
| File | Role |
|---|---|
| `prepare_data_v6.py` | leave-k-out temporal split → data/v6/base |
| `build_negatives_v6.py` | hard (≤2) + soft (==3) pools, leakage-free |
| `build_collab_emb.py` | co-occurrence SVD collaborative item vectors (M2 code source) |
| `rqvae.py` | RQ-VAE, `RECSYS_RQSRC=content/collab/big` |
| `model.py` | `TwoTowerV6`: atomic/hybrid id + graded-negative loss |
| `v6.py` | driver (`RECSYS_MODEL`=M0–M3, `RECSYS_NEG`=N0–N2), set-based eval |
| `v6_cold.py` | cold-item holdout (the M2 collaborative-code cold result is leakage, see notes) |
| `aggregate_v6.py` / `run_v6_sweep.sh` | aggregate / background 8-GPU sweep |

## Notes
- GPU steps need `CUDA_MPS_PIPE_DIRECTORY=""` (baked into `_env.sh`); use idle GPUs.
- **M2 cold-item leakage**: the collaborative vectors contain interactions that were supposed to be held out for cold items; the true cold-start conclusion comes from M1 (content codes).
