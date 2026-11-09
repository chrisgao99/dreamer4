#!/usr/bin/env bash
# Restart H90 D1 World Model fine-tuning from the last-good H30 step10k,
# using the tokenizer exactly as trained: overlapping 32-frame chunks, stride 30.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
INIT_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_wm_time1_mapx1_best_upto600k_exact_ctx1_h30_b1_50k/step_00010000.pt"

RUN_NAME="waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k"
CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$RUN_NAME"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"
PIPELINE_LOG="$LOG_DIR/${RUN_NAME}_pipeline.log"

SESSION_NAME="${SESSION_NAME:-wm_time1_mapx1_h30step10k_h90_d1_chunk32s30_cuda0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

for path in "$PYTHON" "$TRAIN_SCRIPT" "$TOKENIZER_CKPT" "$INIT_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for path in "$DATA_ROOT/train" "$DATA_ROOT/val"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 1; }
done
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$REPO_ROOT/waymo/wandb"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    SESSION_NAME="$SESSION_NAME" CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Training log: $LOG_FILE"
  echo "Pipeline log: $PIPELINE_LOG"
  echo "Checkpoints: $CKPT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

{
  echo "===== $(date) H90 D1 chunk32s30 World Model training start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "run=$RUN_NAME init_ckpt=$INIT_CKPT"
  echo "objective=rollout ctx=1 horizon=90 eval_d=1.0 samples_per_frame=1 batch=1 steps=50000 lr=5e-6"
  echo "tokenizer_chunk_window=32 tokenizer_chunk_stride=30 ranges_for_T91=[0,32),[30,62),[59,91)"
} | tee -a "$PIPELINE_LOG"

start=(--init_ckpt "$INIT_CKPT")
if [[ -f "$CKPT_DIR/latest.pt" ]]; then
  start=(--resume "$CKPT_DIR/latest.pt")
fi

"$PYTHON" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_ROOT/train" --val_data_dir "$DATA_ROOT/val" \
  --tokenizer_ckpt "$TOKENIZER_CKPT" --ckpt_dir "$CKPT_DIR" "${start[@]}" \
  --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
  --seq_len 91 --eval_seq_len 91 --eval_ctx 1 --eval_horizon 90 \
  --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
  --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
  --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
  --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
  --grad_clip 1 --amp_dtype bf16 \
  --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4 \
  --agent_kinematic_xy_weight 5 --agent_speed_yaw_kinematic_weight 2 \
  --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
  --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
  --train_decoded_loss_weight 0 --train_objective rollout \
  --batch_size 1 --eval_batch_size 4 --max_steps 50000 \
  --log_every 20 --eval_every 5000 --eval_max_batches 32 --save_every 5000 --no-save_latest_each_epoch \
  --lr 5e-6 --weight_decay 0 \
  --wandb --wandb_project waymo-world-model --wandb_run_name "$RUN_NAME" \
  2>&1 | tee -a "$LOG_FILE"

final_ckpt="$CKPT_DIR/final_step_00050000.pt"
[[ -f "$final_ckpt" ]] || { echo "Training ended without $final_ckpt" >&2; exit 1; }
echo "===== $(date) H90 D1 chunk32s30 World Model training complete =====" | tee -a "$PIPELINE_LOG"
