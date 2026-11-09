#!/usr/bin/env bash
set -euo pipefail
main() {
 ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
 cd "$ROOT"
 PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
 VARIANT="${VARIANT:?Set mapadapt or laneconn}"
 case "$VARIANT" in
 mapadapt) GPU=3; SOURCE=waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_scratch100k_20261007; FILE=final_step_000100000.pt; TAG=mapadapt100k ;;
 laneconn) GPU=2; SOURCE=waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007; FILE=step_000090000.pt; TAG=laneconn90k ;;
 *) exit 1;; esac
 RUN="waymo_daf_${TAG}_mon8_gt_unroll80_b5_detach0_10k_20261009"
 SESSION="mon80_${TAG}_cuda${GPU}_20261009"
 OUT="$ROOT/waymo/checkpoints/$RUN"
 LOG="$ROOT/waymo/logs/wm/$RUN.log"
 if [[ "${1:-}" != --worker ]];then
  [[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo 'Fresh output exists';exit 1; }
  if tmux has-session -t "=$SESSION" 2>/dev/null;then echo 'Session exists';exit 1;fi
  printf -v command '%q ' env PYTHON="$PYTHON" VARIANT="$VARIANT" bash "$ROOT/waymo/training/world_model/launch_adapted_mon80_tmux.sh" --worker
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
  echo "Started $SESSION; log=$LOG"
  exit 0
 fi
 mkdir -p "$(dirname "$LOG")"
 export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
 exec > >(tee -a "$LOG") 2>&1
 echo "MON K8 H40 B5 unroll80 detach0 CUDA=$GPU source=$SOURCE/$FILE"
 "$PYTHON" waymo/training/world_model/train_waymo_direct_action_flow_mon80.py \
 --checkpoint "$ROOT/waymo/checkpoints/$SOURCE/$FILE" \
 --train_data "$ROOT/data/waymo_vector_dataset_ooi_centered_50k_with_lengths/train" \
 --val_data "$ROOT/data/waymo_vector_dataset_ooi_centered_50k_with_lengths/val" --output_dir "$OUT" \
 --map_cache_dir "$ROOT/data/waymo_map_smarttrajtok_5m_v2" --lane_topology_dir "$ROOT/data/waymo_lane_connectivity_5m_v1" \
 --detach_every 0 --seed 20260911 --batch_size 2 --num_workers 4 --num_candidates 8 --solver_steps 8 \
 --max_steps 10000 --lr 1e-6 --ema_decay .999 --eval_every 500 --eval_batches 8 --eval_batch_size 4 \
 --eval_rollouts 4 --physical_eval_batches 0 --save_every 100 --snapshot_every 1000 --log_every 10
}
main "$@";exit $?
