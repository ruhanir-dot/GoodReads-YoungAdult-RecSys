#!/usr/bin/env bash
# Reproduce the v4 seed sweep (content {desc,profile,both} x seed {42,43,44}) on the
# A40 box (env `a40`, torch 2.6/cu124), to compare against the original H200 run
# (env `h200`, torch 2.10/cu130). NON-DESTRUCTIVE: writes to `*_a40`-tagged files so the
# original H200 eval_{c}_s{s}.json / ckpt_{c}_s{s}.pt baselines are left untouched.
#
# All 9 jobs run concurrently, round-robin across the currently-idle GPUs (this box has 8).
set -u; cd "$(dirname "$0")"; source ./_env.sh
LOG=../../data/v4/sweep; mkdir -p "$LOG"

IFS=, read -ra GPUS <<< "$(idle_gpus)"
[ "${#GPUS[@]}" -eq 0 ] && GPUS=(0 1)
echo "repro on idle GPUs: ${GPUS[*]}  | PY=$PY"

jobs=(); for c in desc profile both; do for s in 42 43 44; do jobs+=("$c|$s"); done; done

pids=()
for i in "${!jobs[@]}"; do
  c=$(echo "${jobs[$i]}" | cut -d'|' -f1); s=$(echo "${jobs[$i]}" | cut -d'|' -f2)
  g=${GPUS[$(( i % ${#GPUS[@]} ))]}
  tag="${c}_s${s}_a40"
  CUDA_VISIBLE_DEVICES=$g RECSYS_CONTENT=$c RECSYS_SEED=$s RECSYS_MLP=256,128 RECSYS_DOUT=64 RECSYS_DC=128 \
    RECSYS_TAG=$tag HF_HUB_OFFLINE=1 $PY -u v4.py > "$LOG/repro_${tag}.log" 2>&1 &
  pids+=($!)
  echo "launched $tag on GPU $g (pid ${pids[-1]})"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "V4 A40 REPRO DONE (failures=$fail)"
