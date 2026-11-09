#!/usr/bin/env bash
set -euo pipefail
main() {
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  cd "$ROOT"
  PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
  CUDA_DEVICE="${CUDA_DEVICE:?Set CUDA_DEVICE=0 or 1}"
  case "$CUDA_DEVICE" in
    0) TAG=mapadapt_100k; RUN=waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_scratch100k_20261007; FILE=final_step_000100000.pt ;;
    1) TAG=laneconn_90k; RUN=waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007; FILE=step_000090000.pt ;;
    *) echo 'Expected CUDA_DEVICE 0 or 1' >&2;exit 1 ;;
  esac
  SESSION="vis_${TAG}_intersection10_random32_cuda${CUDA_DEVICE}_20261008"
  OUT="$ROOT/waymo/eval_results/world_model/mapadapt_laneconn_intersection10_random32_plan40_b40_h80_20261008/$TAG"
  LOG="$ROOT/waymo/logs/evaluation/${TAG}_intersection10_random32_20261008.log"
  CKPT="$ROOT/waymo/checkpoints/$RUN/$FILE"
  [[ -f "$CKPT" && -x "$PYTHON" ]]
  if [[ "${1:-}" != --worker ]]; then
    [[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo 'Fresh output exists' >&2;exit 1; }
    if tmux has-session -t "=$SESSION" 2>/dev/null;then echo 'Session exists' >&2;exit 1;fi
    printf -v command '%q ' env PYTHON="$PYTHON" CUDA_DEVICE="$CUDA_DEVICE" bash "$ROOT/waymo/evaluation/launch_vis_mapadapt_laneconn_images_tmux.sh" --worker
    tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
    echo "Started $SESSION; output=$OUT; log=$LOG"
    exit 0
  fi
  mkdir -p "$OUT" "$(dirname "$LOG")"
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
  exec > >(tee -a "$LOG") 2>&1
  echo "Started: CUDA=$CUDA_DEVICE checkpoint=$CKPT scenes=10 K=32 context=11 H40 B40 total=80 images_only=true"
  "$PYTHON" waymo/evaluation/visualize_flow_noise_coverage.py \
    --checkpoint "$CKPT" --mode random32 --images_only --output_dir "$OUT" \
    --map_cache_dir "$ROOT/data/waymo_map_smarttrajtok_5m_v2" \
    --lane_topology_dir "$ROOT/data/waymo_lane_connectivity_5m_v1" --max_scenes 10 --batch_size 4
  echo 'Evaluation complete'
}
main "$@"; exit $?
