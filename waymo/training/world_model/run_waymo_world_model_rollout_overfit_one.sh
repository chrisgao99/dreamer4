#!/usr/bin/env bash
# Free-running rollout overfit on the exact fixed-start training evaluation task.

set -euo pipefail

HORIZON="${1:-}"
case "$HORIZON" in
  10) DEFAULT_CUDA=0 ;;
  30) DEFAULT_CUDA=1 ;;
  90) DEFAULT_CUDA=2 ;;
  *)
    echo "Usage: $0 {10|30|90}" >&2
    exit 2
    ;;
esac

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b32_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_decmap_noamp/best.pt"
INIT_CKPT="$REPO_ROOT/waymo/checkpoints/world_model_overfit/wm_overfit_scenes8_randwin_self05_ctx1_h10_20260720/final_step_00050000.pt"
TRAIN_DATA="$REPO_ROOT/waymo/data/world_model_overfit_subsets/wm_overfit_scenes8_randwin_self05_ctx1_h10_20260720/train"

CUDA_DEVICE="${CUDA_DEVICE:-$DEFAULT_CUDA}"
MAX_STEPS="${MAX_STEPS:-5000}"
RUN_NAME="${RUN_NAME:-wm_rollout_overfit_train8_fixed_ctx1_h${HORIZON}_d025_initself05_20260722}"
SESSION_NAME="${SESSION_NAME:-$RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/world_model_rollout_overfit/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/world_model_rollout_overfit}"
LOG="${LOG:-$LOG_DIR/$RUN_NAME.log}"

for required_file in "$PYTHON" "$TRAIN_SCRIPT" "$TOKENIZER_CKPT" "$INIT_CKPT"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done
if [[ ! -d "$TRAIN_DATA" ]]; then
  echo "Missing training data: $TRAIN_DATA" >&2
  exit 1
fi
if [[ -e "$CKPT_DIR/latest.pt" || -e "$CKPT_DIR/final_step_$(printf '%08d' "$MAX_STEPS").pt" ]]; then
  echo "Output checkpoint already exists; refusing to overwrite: $CKPT_DIR" >&2
  exit 1
fi
mkdir -p "$CKPT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 \
    REPO_ROOT="$REPO_ROOT" \
    PYTHON="$PYTHON" \
    CUDA_DEVICE="$CUDA_DEVICE" \
    MAX_STEPS="$MAX_STEPS" \
    RUN_NAME="$RUN_NAME" \
    SESSION_NAME="$SESSION_NAME" \
    CKPT_DIR="$CKPT_DIR" \
    LOG_DIR="$LOG_DIR" \
    LOG="$LOG" \
    bash "$SCRIPT_PATH" "$HORIZON"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $LOG"
  echo "Checkpoints: $CKPT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1

{
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME session=$SESSION_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "train_data=$TRAIN_DATA samples=$(find "$TRAIN_DATA" -maxdepth 1 -type l | wc -l)"
  echo "init_ckpt=$INIT_CKPT"
  echo "objective=rollout fixed_start=1 ctx=1 horizon=$HORIZON seq_len=$((HORIZON + 1))"
  echo "sampler=shortcut eval_d=0.25 max_rollout_window=11"
  echo "batch_size=1 max_steps=$MAX_STEPS lr=1e-5 decoded_loss_weight=0"
  echo "checkpoint_dir=$CKPT_DIR"
  echo "========================"

  "$PYTHON" "$TRAIN_SCRIPT" \
    --data_dir "$TRAIN_DATA" \
    --val_data_dir "$TRAIN_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" \
    --init_ckpt "$INIT_CKPT" \
    --ckpt_dir "$CKPT_DIR" \
    --device cuda \
    --seed 0 \
    --seq_len "$((HORIZON + 1))" \
    --eval_seq_len "$((HORIZON + 1))" \
    --eval_ctx 1 \
    --eval_horizon "$HORIZON" \
    --max_rollout_window 11 \
    --eval_schedule shortcut \
    --eval_d 0.25 \
    --batch_size 1 \
    --eval_batch_size 4 \
    --num_workers 2 \
    --max_steps "$MAX_STEPS" \
    --log_every 20 \
    --eval_every 200 \
    --eval_max_batches 0 \
    --save_every 1000 \
    --no-save_latest_each_epoch \
    --d_model_dyn 512 \
    --dyn_depth 8 \
    --n_heads 8 \
    --time_every 4 \
    --packing_factor 2 \
    --n_register 8 \
    --k_max 64 \
    --train_objective rollout \
    --lr 1e-5 \
    --weight_decay 0 \
    --grad_clip 1.0 \
    --amp_dtype bf16 \
    --agent_xy_loss smooth_l1 \
    --agent_xy_parameterization absolute \
    --focus_agent_weight 4 \
    --agent_kinematic_xy_weight 5 \
    --agent_speed_yaw_kinematic_weight 2 \
    --use_ego_actions \
    --ego_action_source focus \
    --ego_action_normalization raw \
    --no-ego_action_clamp \
    --agent_far_weight 0.25 \
    --agent_near_radius_m 50.0 \
    --agent_distance_source focus \
    --train_decoded_loss_weight 0

  echo "Finished at $(date)"
} 2>&1 | tee -a "$LOG"
