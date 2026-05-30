#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source ./_env.sh
LOG=../../data/v4/sweep; mkdir -p "$LOG"
IDLE=$(idle_gpus); G0=$(echo "$IDLE"|cut -d, -f1); G1=$(echo "$IDLE"|cut -d, -f2); G0=${G0:-0}; G1=${G1:-1}; [ "$G1" = "$G0" ] && G1=$G0
echo "seed sweep on GPUs $G0,$G1"
jobs=()
for c in desc profile both; do for s in 42 43 44; do jobs+=("$c|$s"); done; done
run1(){ local g=$1 j=$2; local c=$(echo $j|cut -d'|' -f1) s=$(echo $j|cut -d'|' -f2)
  CUDA_VISIBLE_DEVICES=$g RECSYS_CONTENT=$c RECSYS_SEED=$s RECSYS_MLP=256,128 RECSYS_DOUT=64 RECSYS_DC=128 \
    RECSYS_TAG=${c}_s${s} HF_HUB_OFFLINE=1 $PY -u v4.py > "$LOG/seed_${c}_s${s}.log" 2>&1; }
n=${#jobs[@]}; i=0
while [ $i -lt $n ]; do
  run1 "$G0" "${jobs[$i]}" & P0=$!
  if [ $((i+1)) -lt $n ]; then run1 "$G1" "${jobs[$((i+1))]}" & P1=$!; else P1=""; fi
  wait $P0; [ -n "$P1" ] && wait $P1
  echo "wave: ${jobs[$i]} ${jobs[$((i+1))]:-}"; i=$((i+2))
done
echo "V4 SEEDS DONE"
