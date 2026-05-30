# retrieval/src/v5 — Two-Tower Recall v5 (semantic ID + dislike hard negatives + language; self-contained)

v5 = pulls **three orthogonal new levers** on the validated two-tower backbone (each one targeting a
weakness v4 quantified), using the **fresh v3 LLM profiles**. This folder is self-contained;
**code lives in `src/v5/`, all data is written to `data/v5/`** (base arrays are reused read-only from `data/v1/`).

> Environment: **a40** (nlplab2, 8×A40, conda `a40` = torch 2.6/cu124, MPS bypass). `_env.sh` is host-aware.

## The three levers
| Lever | Mechanism | v4 weakness it targets | Code |
|---|---|---|---|
| **A** semantic ID | RQ-VAE quantizes item content into discrete codes; item-ID = Σ code emb (atomic/semantic/hybrid) | cold items (≈0), long tail | `rqvae.py` + `model.py` |
| **B** dislike hard negatives | items a user rated `≤2` in train become explicit negatives in the loss | discrimination / precision | `build_dislike_pool.py` + loss |
| **C** language | user↔book language: C1 feature / C2 retrieval-time penalty | non-English tail | `build_language.py` + scoring penalty |
| **D** = A2+B1+C2 | all three combined | overall | `v5.py` variant=D |

## What needs redoing (h200 → a40)
- **Reuse (machine-independent, deterministic CPU/duckdb)**: the base arrays in `data/v1/` (split / pairs / hist / numeric & categorical features / popularity / val / test / meta) — not rebuilt.
- **Redo on a40 (GPU/bge, and it's brand-new v3 data) → written to `data/v5/`**: all bge encodings (desc / tags / like / dislike / content), RQ-VAE semantic codes, language features, dislike pool.

## One-shot / step-by-step
```bash
cd retrieval/src/v5
source ./_env.sh
# 0) data prep (a40, writes data/v5)
CUDA_VISIBLE_DEVICES=$(best_gpu) HF_HUB_OFFLINE=1 $PY encode_v3.py   # bge: desc/tags/content/like/dislike
$PY build_language.py                                                # book_lang / user_lang_w / lang_vocab
$PY build_dislike_pool.py                                            # dislike_pad / dislike_pool (train rating<=2)
CUDA_VISIBLE_DEVICES=$(best_gpu) $PY rqvae.py                        # RQ-VAE -> sem_codes.npy (+ usage report)
$PY prepare_coldstart.py                                             # 15% item holdout -> data/v5/cold
# 1) single config
CUDA_VISIBLE_DEVICES=$(best_gpu) RECSYS_VARIANT=D RECSYS_SEED=42 $PY v5.py
# 2) full ablation (background, 8-GPU parallel; A/B/C/D × 3 seeds + lever-A cold items)
bash run_v5_sweep.sh
# 3) aggregate -> report table + v5_summary.json
$PY aggregate_v5.py
```

## Modules
| File | Role |
|---|---|
| `config.py` | BASE_DIR=data/v1 (read) / V5_DIR=data/v5 (write); RQ-VAE / lever / training hyperparams |
| `encode_v3.py` | a40 bge encoding of 5 content sources (desc/tags/content/like/dislike) → data/v5 |
| `build_language.py` | normalized book language codes (973 entries, 3→2 letters) + shared lang vocab + user language distribution |
| `build_dislike_pool.py` | train `rating≤2` → per-user dislike hard-negative pool (CSR + padded) |
| `rqvae.py` | residual-quantized VAE (EMA + data init + dead-code revival) → `sem_codes.npy` [n,L] |
| `model.py` | `TwoTowerV5`: id_mode (atomic/semantic/hybrid) + explicit-negative loss + user-lang branch |
| `v5.py` | main driver (RECSYS_VARIANT=A0/A1/A2/B1/C1/C2/D); warm full-catalog + user/item tier + per-language |
| `v5_cold.py` / `prepare_coldstart.py` | lever-A cold-item holdout (does semantic ID rescue cold items) |
| `aggregate_v5.py` | aggregate all eval_*.json → comparison table + `v5_summary.json` |
| `run_v5_sweep.sh` | background 8-GPU parallel full ablation |

## Notes
- GPU steps need `CUDA_MPS_PIPE_DIRECTORY=""` (baked into `_env.sh`); use currently idle GPUs (`idle_gpus`/`best_gpu`).
- Semantic codes: after RQ-VAE dead-code revival, codebook utilization ~250/256 per layer, code-tuple collision ~0.8% (see `data/v5/rqvae_report.json`).
- The eval protocol matches v1/v4 (leave-last-out, train-item mask, exact full-catalog scoring), so results are comparable to the v4 base R@100 ≈ 0.486.
