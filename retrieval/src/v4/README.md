# retrieval/src/v4 — Two-Tower Recall v4 (self-contained)

v4 = item-content + architecture/hyperparameter exploration of the two-tower recall model.
This folder is **self-contained**: it holds every module v4 needs to run (version-specific code
plus one copy each of the shared modules), so you can `cd` in and run it standalone.

> Framework: two-tower (user = pooled collaborative history; item = hybrid id + content + num + cat).
> v3's user-content branch was shown to add nothing → v4 drops it and spends the compute on the
> item side + architecture instead.

## One-shot / step-by-step
```bash
cd retrieval/src/v4
source ./_env.sh                       # exports PY, the MPS bypass, idle_gpus()/best_gpu()
# 0) base data (already present under data/v1, no need to rerun; to rebuild:)
$PY prepare_data.py                    # parquet + profiles → data/v1/* (LOO split + features + history + popularity)
CUDA_VISIBLE_DEVICES=$(best_gpu) $PY encode_items_bge.py       # item_text_emb.npy (bge title+desc)
CUDA_VISIBLE_DEVICES=$(best_gpu) $PY encode_item_profiles.py   # item_profile_emb.npy (bge v2 profile)
# 1) single config (content = desc | profile | both; arch/seed configurable)
CUDA_VISIBLE_DEVICES=$(best_gpu) RECSYS_CONTENT=both RECSYS_SEED=42 $PY v4.py
# 2) reproduce the report: content × 3 seeds (true variance) / architecture sweep
bash run_v4_seeds.sh                    # desc/profile/both × seed{42,43,44} → data/v4/sweep/
bash run_v4_sweep.sh                    # content × architecture, 8 configs
# 3) cold-item holdout
$PY prepare_coldstart.py               # hold out 15% of items → data/v4/cold
CUDA_VISIBLE_DEVICES=$(best_gpu) RECSYS_CONTENT=both $PY v4_cold.py
# 4) FAISS index + demo
CUDA_VISIBLE_DEVICES=$(best_gpu) $PY build_index.py
```

## Modules
| File | Role |
|---|---|
| `config.py` | paths / hyperparams / feature dims (shared; `OUT_DIR=data/v1` is the shared base) |
| `env.py` | root resolution + `STEP_DATA` (robust to nesting depth: `src/` or `src/v4/` both resolve to `retrieval/data`) |
| `_env.sh` | runtime env (h200 PY, MPS bypass, idle/best GPU) |
| `model.py` | `TwoTower` (shared item-ID & tag emb; in-batch softmax + logQ) |
| `dataset.py` | loads the data/v1 arrays (`load_meta`, etc.) |
| `evaluate.py` | train mask / positive counts / activity-tier helpers |
| `prepare_data.py` / `encode_items_bge.py` / `encode_item_profiles.py` | build shared base + content bge |
| `v4.py` | **v4 main driver** (content = desc/profile/both, configurable arch/seed; warm catalog + user/item tiered eval) |
| `v4_cold.py` / `prepare_coldstart.py` | cold-item holdout train/eval |
| `build_index.py` | FAISS flat-IP index + recommendation demo |
| `run_v4_seeds.sh` / `run_v4_sweep.sh` | background multi-config sweep (2-GPU parallel, logs to `../../data/v4/sweep/`) |

## Conclusions
Content source (desc/profile/both) and architecture tuning are **all within noise**; the two-tower
warm-catalog plateau is **R@100 ≈ 0.486**. Canonical config = hybrid (id + both-content), MLP
`[256,128]`, d_out 64 (≈ desc-only).

## Notes
- GPU steps need `CUDA_MPS_PIPE_DIRECTORY=""` (MPS-failure bypass on this node; baked into `_env.sh`).
- Use the h200 conda env and the **currently idle GPU** (`idle_gpus`/`best_gpu`); don't saturate the shared machine.
