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
  SESSION="eval_${TAG}_512_k32_holonomic_cuda${CUDA_DEVICE}_20261008"
  OUT="$ROOT/waymo/eval_results/world_model/mapadapt_laneconn_512_k32_closedloop_holonomic_20261008/$TAG"
  LOG="$ROOT/waymo/logs/evaluation/${TAG}_512_k32_closedloop_holonomic_20261008.log"
  CKPT="$ROOT/waymo/checkpoints/$RUN/$FILE"
  [[ -f "$CKPT" && -x "$PYTHON" ]]
  if [[ "${1:-}" != --worker ]]; then
    [[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo 'Fresh output exists' >&2;exit 1; }
    if tmux has-session -t "=$SESSION" 2>/dev/null;then echo 'Session exists' >&2;exit 1;fi
    printf -v command '%q ' env PYTHON="$PYTHON" CUDA_DEVICE="$CUDA_DEVICE" bash "$ROOT/waymo/evaluation/launch_eval_mapadapt_laneconn_512_tmux.sh" --worker
    tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
    echo "Started $SESSION; output=$OUT; log=$LOG"
    exit 0
  fi
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id="$CUDA_DEVICE")"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
  [[ "$apps" != *"$uuid"* ]] || { echo 'GPU occupied' >&2;exit 1; }
  mkdir -p "$OUT" "$(dirname "$LOG")"
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
  exec > >(tee -a "$LOG") 2>&1
  echo "Started: CUDA=$CUDA_DEVICE checkpoint=$CKPT scenes=512 K=32 context=11 H40 B5 total=80"
  "$PYTHON" waymo/evaluation/eval_waymo_direct_action_flow_multisample.py \
    --checkpoint "$CKPT" --val_data_dir "$ROOT/data/waymo_vector_dataset_ooi_centered_50k_with_lengths/val" \
    --map_cache_dir "$ROOT/data/waymo_map_smarttrajtok_5m_v2" --lane_topology_dir "$ROOT/data/waymo_lane_connectivity_5m_v1" \
    --output_json "$OUT/result.json" --rollout_ade_csv "$OUT/rollout_ade.csv" \
    --scene_metrics_csv "$OUT/scene_metrics.csv" --details_npz "$OUT/details.npz" \
    --device cuda --weights ema --focus_mode generate_all --eval_batch_size 4 --eval_max_batches 128 \
    --num_workers 4 --num_rollouts 32 --rollout_steps 80 --commitment_steps 5 --solver_steps 8 \
    --seed 12346 --log_every 1 --physical_metrics --metric_validity holonomic \
    --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977
  echo 'Evaluation complete'
}
main "$@"; exit $?
