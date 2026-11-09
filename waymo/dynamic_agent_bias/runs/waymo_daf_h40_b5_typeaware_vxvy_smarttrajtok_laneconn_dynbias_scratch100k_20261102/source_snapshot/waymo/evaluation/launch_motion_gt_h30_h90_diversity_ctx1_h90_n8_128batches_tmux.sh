#!/usr/bin/env bash
# Evaluate one selected N=8 + Motion GT checkpoint with context=1, future=90,
# eight stochastic rollouts, and the fixed 512-scene (128 x batch-size 4) set.

set -euo pipefail

STAGE="${STAGE:?Set STAGE to h30 or h90}"
CUDA_DEVICE="${CUDA_DEVICE:?Set CUDA_DEVICE}"
REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
BASE="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage"
OUT="$REPO_ROOT/waymo/eval_results/world_model/motion_gt_h30_h90_diversity_ctx1_h90_n8_128batches_20260824"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"

case "$STAGE" in
  h30)
    LABEL="motion_gt_h30_step21000"
    CKPT="$BASE/waymo_wm_stage1best_mon8_fullmotion_physproxy_ctx1_h30_d1_chunk32s30_30k_from3k/step_00021000.pt"
    ;;
  h90)
    LABEL="motion_gt_h90_step27000"
    CKPT="$BASE/waymo_wm_stage1best_mon8_fullmotion_physproxy_h30best30k_ctx1_h90_d1_chunk32s30_30k/step_00027000.pt"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

SESSION_NAME="${SESSION_NAME:-wm_diversity_motion_gt_${STAGE}_cuda${CUDA_DEVICE}}"
PIPELINE_LOG="$LOG_DIR/${SESSION_NAME}.log"
OUTPUT="$OUT/${LABEL}_ctx1_h90_n8_128batches.json"
mkdir -p "$OUT" "$LOG_DIR"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER" "$CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 STAGE="$STAGE" CUDA_DEVICE="$CUDA_DEVICE" \
    SESSION_NAME="$SESSION_NAME" REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $PIPELINE_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) $LABEL diversity evaluation start ====="
  echo "cuda=$CUDA_VISIBLE_DEVICES protocol=ctx1_future90_N8_batches128_batchsize4_scenes512"
  echo "checkpoint=$CKPT"
  if [[ -f "$OUTPUT" ]]; then
    echo "Already evaluated: $OUTPUT"
  else
    "$PYTHON" "$EVAL_SCRIPT" \
      --data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
      --val_data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
      --tokenizer_ckpt "$TOKENIZER" --eval_ckpt "$CKPT" --output_json "$OUTPUT" \
      --device cuda --eval_seq_len 91 --eval_ctx 1 --horizons 90 \
      --eval_batch_size 4 --eval_max_batches 128 --eval_subset_size 512 \
      --eval_subset_seed 20260824 \
      --num_workers 4 --eval_num_rollouts 8 --eval_multisample_seed 20260824 \
      --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
      --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
      --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
      --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
      --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4 \
      --use_ego_actions --ego_action_source focus --ego_action_normalization raw \
      --no-ego_action_clamp --agent_far_weight 0.25 --agent_near_radius_m 50 \
      --agent_distance_source focus
  fi
  echo "===== $(date) $LABEL diversity evaluation complete ====="
} 2>&1 | tee -a "$PIPELINE_LOG"
