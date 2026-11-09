#!/usr/bin/env bash
set -euo pipefail
main() {
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  cd "$REPO_ROOT"
  if [[ -z "${PYTHON:-}" ]]; then
    owner_root="$(cd "$REPO_ROOT/../.." && pwd)"
    candidates=("${CONDA_PREFIX:-/nonexistent}/bin/python" "$owner_root/.conda/envs/dreamer4/bin/python"
                "${HOME}/.conda/envs/dreamer4/bin/python" "$(command -v python || true)")
    for candidate in "${candidates[@]}"; do
      if [[ -x "$candidate" ]] && "$candidate" -c 'import torch,numpy,google.protobuf' >/dev/null 2>&1; then PYTHON="$candidate";break;fi
    done
    [[ -n "${PYTHON:-}" ]] || { echo 'Activate dreamer4 or set PYTHON' >&2;exit 1; }
  fi
  export PYTHON
  export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k_with_lengths}"
  export MAP_CACHE_DIR="${MAP_CACHE_DIR:-$REPO_ROOT/data/waymo_map_smarttrajtok_5m_v2}"
  export INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/waymo_scenario_index_v1}"
  export TOPOLOGY_DIR="${TOPOLOGY_DIR:-$REPO_ROOT/data/waymo_lane_connectivity_5m_v1}"
  export CUDA_DEVICE="${CUDA_DEVICE:-1}"
  export RUN_NAME="${RUN_NAME:-waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007}"
  export SESSION_NAME="${SESSION_NAME:-daf_laneconn_stage1_20261007_cuda1}"
  export PREP_WORKERS="${PREP_WORKERS:-4}"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  if [[ "${1:-}" != '--worker' ]]; then
    [[ "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid run name' >&2;exit 1; }
    [[ ! -e "waymo/checkpoints/$RUN_NAME" && ! -e "waymo/logs/wm/$RUN_NAME.log" ]] || { echo 'Fresh output exists' >&2;exit 1; }
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then echo 'Session exists' >&2;exit 1;fi
    mkdir -p waymo/logs/wm
    prefix=''
    for name in PYTHON DATA_ROOT MAP_CACHE_DIR INDEX_DIR TOPOLOGY_DIR CUDA_DEVICE RUN_NAME SESSION_NAME PREP_WORKERS; do
      printf -v entry '%q=%q ' "$name" "${!name}";prefix+="$entry"
    done
    printf -v worker '%q ' bash "$REPO_ROOT/waymo/lane_connectivity/launch_tmux.sh" --worker
    printf -v log '%q' "$REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"
    tmux new-session -d -s "$SESSION_NAME" "env $prefix bash -c $(printf '%q' "set -o pipefail; $worker 2>&1 | tee $log")"
    echo "tmux attach -t $SESSION_NAME"
    exit 0
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -c 'import torch;assert torch.cuda.is_available(),"CUDA unavailable"'
  echo 'stage=prepare_lane_topology'
  "$PYTHON" waymo/lane_connectivity/prepare_topology.py --data_root "$DATA_ROOT" --map_cache_dir "$MAP_CACHE_DIR" --index_dir "$INDEX_DIR" --output_dir "$TOPOLOGY_DIR" --workers "$PREP_WORKERS"
  # Recheck after CPU preparation; never take a device now used by another job.
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id="$CUDA_DEVICE")"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
  if [[ "$apps" == *"$uuid"* ]]; then echo "CUDA $CUDA_DEVICE became occupied; refusing to interfere" >&2;exit 1;fi
  echo 'stage=preflight_then_train'
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" waymo/lane_connectivity/run_stage1.py --topology_dir "$TOPOLOGY_DIR" --data_root "$DATA_ROOT" --map_cache_dir "$MAP_CACHE_DIR" --run_name "$RUN_NAME"
}
main "$@"; exit $?
