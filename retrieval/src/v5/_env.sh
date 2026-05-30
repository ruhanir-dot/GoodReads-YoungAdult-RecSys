# Shared launch env for this project. `source` it from run scripts.
# Host-aware conda env (override anytime with RECSYS_PY=/path/to/python):
#   nlplab2  = 8x A40 box   -> conda env `a40`   (torch 2.6 / cu124)
#   <h200 box> = 4x H200 box -> conda env `h200` (torch 2.10 / cu130)
# The A40 env (a40) was added 2026-05-29 to reproduce v4 off the H200 box.
_recsys_default_py() {
  case "$(hostname)" in
    nlplab2*) echo /home/jacky/miniforge3/envs/a40/bin/python ;;
    *)        echo /home/jacky/miniforge3/envs/h200/bin/python ;;
  esac
}
export PY="${RECSYS_PY:-$(_recsys_default_py)}"
export CUDA_MPS_PIPE_DIRECTORY=""          # MPS bypass (node MPS broken on H200 box; harmless if unused)

# Print comma-separated indices of idle GPUs (util<10% and <40GB used).
idle_gpus() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '($2+0)<10 && ($3+0)<40000 {print $1}' | paste -sd, -
}
# Print the single most-idle GPU index (lowest util, then lowest mem).
best_gpu() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n -k3 -n | head -1 | awk -F', *' '{print $1}'
}
