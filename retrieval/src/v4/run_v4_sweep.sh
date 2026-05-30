#!/usr/bin/env bash
# v4 sweep: content source x architecture. Runs 2 configs at a time across two idle GPUs.
# Launch via the harness background mechanism (run_in_background).
set -u; cd "$(dirname "$0")"
source ./_env.sh
LOG=../../data/v4/sweep; mkdir -p "$LOG"
IDLE=$(idle_gpus); G0=$(echo "$IDLE" | cut -d, -f1); G1=$(echo "$IDLE" | cut -d, -f2)
G0=${G0:-0}; G1=${G1:-1}; [ "$G1" = "$G0" ] && G1=$G0
echo "sweep on GPUs $G0,$G1"

# TAG | CONTENT | MLP | DOUT | DC
configs=(
"descA|desc|256,128|64|128"
"profA|profile|256,128|64|128"
"bothA|both|256,128|64|128"
"profWide|profile|512,256|64|128"
"profBigOut|profile|256,128|128|128"
"profDeep|profile|384,192,96|64|256"
"bothRich|both|512,256|128|256"
"descRich|desc|512,256|128|256"
)

run1() {  # $1=gpu  $2=config
  local g=$1 cfg=$2
  local tag=$(echo "$cfg" | cut -d'|' -f1)
  local content=$(echo "$cfg" | cut -d'|' -f2)
  local mlp=$(echo "$cfg" | cut -d'|' -f3)
  local dout=$(echo "$cfg" | cut -d'|' -f4)
  local dc=$(echo "$cfg" | cut -d'|' -f5)
  CUDA_VISIBLE_DEVICES=$g RECSYS_CONTENT=$content RECSYS_MLP=$mlp RECSYS_DOUT=$dout RECSYS_DC=$dc \
    RECSYS_TAG=$tag HF_HUB_OFFLINE=1 $PY -u v4.py > "$LOG/$tag.log" 2>&1
}

n=${#configs[@]}; i=0
while [ $i -lt $n ]; do
  run1 "$G0" "${configs[$i]}" &
  P0=$!
  if [ $((i+1)) -lt $n ]; then run1 "$G1" "${configs[$((i+1))]}" & P1=$!; else P1=""; fi
  wait $P0; [ -n "$P1" ] && wait $P1
  echo "wave done: ${configs[$i]} ${configs[$((i+1))]:-}"
  i=$((i+2))
done
echo "V4 SWEEP DONE"
