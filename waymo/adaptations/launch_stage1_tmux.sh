#!/usr/bin/env bash
# Invoke this script on the GPU training host; preparation runs on CPU first.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export RUN_NAME="${RUN_NAME:-waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_scratch100k_20261006}"
export SESSION_NAME="${SESSION_NAME:-daf_map_stage1_cuda${CUDA_DEVICE}}"
export MAP_CACHE_DIR="${MAP_CACHE_DIR:-$REPO_ROOT/data/waymo_map_smarttrajtok_5m_v2}"
export SCENARIO_ROOT="${SCENARIO_ROOT:-/p/liverobotics/waymo_open_dataset_motion/scenario}"
export INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/waymo_scenario_index_v1}"
export PREP_WORKERS="${PREP_WORKERS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO_ROOT"
if [[ "${1:-}" != "--worker" ]]; then
  command -v tmux >/dev/null
  [[ -x "$PYTHON" ]] || { echo "Python missing: $PYTHON" >&2; exit 1; }
  [[ "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 1; }
  [[ ! -e "waymo/checkpoints/$RUN_NAME" ]] || { echo 'Checkpoint already exists; choose a new RUN_NAME' >&2; exit 1; }
  tmux has-session -t "$SESSION_NAME" 2>/dev/null && { echo 'tmux session already exists' >&2; exit 1; }
  mkdir -p waymo/logs/wm
  [[ ! -e "waymo/logs/wm/$RUN_NAME.log" ]] || { echo 'Log exists; choose a new RUN_NAME' >&2; exit 1; }
  printf -v worker_cmd '%q ' bash "$REPO_ROOT/waymo/adaptations/launch_stage1_tmux.sh" --worker
  printf -v logfile '%q' "$REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"
  # Explicit exports survive even when the tmux server predates this invocation.
  prefix=''
  for name in PYTHON CUDA_DEVICE RUN_NAME SESSION_NAME MAP_CACHE_DIR SCENARIO_ROOT INDEX_DIR PREP_WORKERS; do
    printf -v entry '%q=%q ' "$name" "${!name}"; prefix+="$entry"
  done
  tmux new-session -d -s "$SESSION_NAME" "env $prefix bash -c $(printf '%q' "set -o pipefail; $worker_cmd 2>&1 | tee $logfile")"
  echo "tmux attach -t $SESSION_NAME"
  echo "log: $REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"
  exit 0
fi
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable on this host/device"'
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k_with_lengths"
"$PYTHON" waymo/adaptations/index_scenarios.py --scenario_root "$SCENARIO_ROOT" --output_dir "$INDEX_DIR" --workers "$PREP_WORKERS"
"$PYTHON" waymo/adaptations/prepare_map_cache.py --data_root "$DATA_ROOT" --scenario_root "$SCENARIO_ROOT" --index_dir "$INDEX_DIR" --output_dir "$MAP_CACHE_DIR" --workers "$PREP_WORKERS"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" waymo/adaptations/run_stage1.py --map_cache_dir "$MAP_CACHE_DIR" --run_name "$RUN_NAME"
