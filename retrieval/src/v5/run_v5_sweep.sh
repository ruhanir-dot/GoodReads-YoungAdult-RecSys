#!/usr/bin/env bash
# v5 ablation sweep (a40), on the REBUILT rating>=4 base (clean lever-B; NOT comparable to v1-v4 0.486).
# Core A/B/C/D warm ablation (3 seeds) + tags-as-content ablation (A0/D x desc/tags/both) +
# lever-A cold-item holdout. All jobs concurrent, round-robin across idle GPUs.
#
#   bash run_v5_sweep.sh
set -u; cd "$(dirname "$0")"; source ./_env.sh
LOG=../../data/v5/sweep; mkdir -p "$LOG"
IFS=, read -ra GPUS <<< "$(idle_gpus)"; [ "${#GPUS[@]}" -eq 0 ] && GPUS=(0 1)
echo "v5 sweep on idle GPUs: ${GPUS[*]} | PY=$PY"

jobs=()
# core warm ablation (content=desc)
for v in A0 A1 A2 B1 C1 C2 D; do for s in 42 43 44; do jobs+=("warm|$v|$s|desc"); done; done
# tags-as-content (the audit's key gap): A0 isolates content lever; D = combined + tags content
for c in tags both; do for s in 42 43 44; do jobs+=("warm|A0|$s|$c"); done; done
for s in 42 43 44; do jobs+=("warm|D|$s|both"); done
# lever-A cold-item holdout
for v in A0 A1 A2; do jobs+=("cold|$v|42|desc"); done

pids=()
for i in "${!jobs[@]}"; do
  IFS='|' read -r mode v s c <<< "${jobs[$i]}"
  g=${GPUS[$(( i % ${#GPUS[@]} ))]}
  if [ "$mode" = warm ]; then
    [ "$c" = desc ] && tag="${v}_s${s}" || tag="${v}_${c}_s${s}"
    script=v5.py
  else tag="cold_${v}_s${s}"; script=v5_cold.py; fi
  CUDA_VISIBLE_DEVICES=$g RECSYS_VARIANT=$v RECSYS_SEED=$s RECSYS_CONTENT=$c RECSYS_TAG=$tag HF_HUB_OFFLINE=1 \
    $PY -u "$script" > "$LOG/${tag}.log" 2>&1 &
  pids+=($!); echo "launched $mode $tag (content=$c) on GPU $g (pid ${pids[-1]})"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "V5 SWEEP DONE (failures=$fail)"
