#!/usr/bin/env bash
# Full v3 profiling: USER then BOOK, 2 shards each on the first two idle GPUs. Resumable.
# Launch via the harness background mechanism (run_in_background).
set -u; cd "$(dirname "$0")"
PY=/home/jacky/miniforge3/envs/h200/bin/python
export CUDA_MPS_PIPE_DIRECTORY="" HF_HUB_OFFLINE=1
G0=${RECSYS_G0:-0}; G1=${RECSYS_G1:-1}
L=../../data/v3
echo "=== USER profiling (GPUs $G0,$G1) ==="
CUDA_VISIBLE_DEVICES=$G0 $PY -u run_user.py --shard 0 --nshards 2 > $L/user_s0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=$G1 $PY -u run_user.py --shard 1 --nshards 2 > $L/user_s1.log 2>&1 &
P1=$!
wait $P0 $P1
$PY -u run_user.py --merge > $L/user_merge.log 2>&1
echo "=== BOOK profiling (GPUs $G0,$G1) ==="
CUDA_VISIBLE_DEVICES=$G0 $PY -u run_book.py --shard 0 --nshards 2 > $L/book_s0.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=$G1 $PY -u run_book.py --shard 1 --nshards 2 > $L/book_s1.log 2>&1 &
P3=$!
wait $P2 $P3
$PY -u run_book.py --merge > $L/book_merge.log 2>&1
echo "=== VOCAB (top-3000) ==="
$PY -u build_vocab.py > $L/vocab.log 2>&1
echo "ALL PROFILING DONE"
