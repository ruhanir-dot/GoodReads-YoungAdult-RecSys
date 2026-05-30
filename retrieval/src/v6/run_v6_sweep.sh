#!/usr/bin/env bash
# v6 sweep (a40): semantic-ID source comparison (M0-M3, neg N2) + graded-negative ablation
# (M0 x N0/N1) + cold-item holdout (M0/M1/M2). leave-k-out base. All concurrent across idle GPUs.
#   bash run_v6_sweep.sh
set -u; cd "$(dirname "$0")"; source ./_env.sh
LOG=../../data/v6/sweep; mkdir -p "$LOG"
IFS=, read -ra GPUS <<< "$(idle_gpus)"; [ "${#GPUS[@]}" -eq 0 ] && GPUS=(0 1)
echo "v6 sweep on idle GPUs: ${GPUS[*]} | PY=$PY"

jobs=()
for m in M0 M1 M2 M3; do for s in 42 43 44; do jobs+=("warm|$m|N2|$s"); done; done   # semantic comparison
for nn in N0 N1; do for s in 42 43 44; do jobs+=("warm|M0|$nn|$s"); done; done        # negative ablation
for m in M0 M1 M2; do jobs+=("cold|$m|N2|42"); done                                    # cold-item holdout

pids=()
for i in "${!jobs[@]}"; do
  IFS='|' read -r mode m nn s <<< "${jobs[$i]}"
  g=${GPUS[$(( i % ${#GPUS[@]} ))]}
  if [ "$mode" = warm ]; then tag="${m}_${nn}_s${s}"; script=v6.py
  else tag="cold_${m}_s${s}"; script=v6_cold.py; fi
  CUDA_VISIBLE_DEVICES=$g RECSYS_MODEL=$m RECSYS_NEG=$nn RECSYS_SEED=$s RECSYS_TAG=$tag HF_HUB_OFFLINE=1 \
    $PY -u "$script" > "$LOG/${tag}.log" 2>&1 &
  pids+=($!); echo "launched $mode $tag on GPU $g (pid ${pids[-1]})"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "V6 SWEEP DONE (failures=$fail)"
