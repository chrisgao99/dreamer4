#!/usr/bin/env bash
# Evaluate the robust DirectActionFlow 90k checkpoint with 11 context frames,
# a full 80-step receding rollout, and 128 batches x batch size 4.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_direct_action_flow_receding.py"
VAL_DATA="${VAL_DATA:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val}"
EVAL_CKPT="${EVAL_CKPT:-$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h15_phys5_yaw075_huber_bounded_tmax090_lr5e5_from62500_v2/step_000090000.pt}"

RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_step90k_ctx11_h80_val128}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/evaluation}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/result.json}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

SESSION_NAME="${SESSION_NAME:-wm_daf_90k_ctx11_h80_val128}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SOLVER_STEPS="${SOLVER_STEPS:-8}"
EVAL_SEED="${EVAL_SEED:-12346}"

for required_file in "$PYTHON" "$EVAL_SCRIPT" "$EVAL_CKPT"; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }

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
    VAL_DATA="$VAL_DATA" \
    EVAL_CKPT="$EVAL_CKPT" \
    RUN_NAME="$RUN_NAME" \
    OUT_DIR="$OUT_DIR" \
    LOG_DIR="$LOG_DIR" \
    OUTPUT_JSON="$OUTPUT_JSON" \
    LOG_FILE="$LOG_FILE" \
    SESSION_NAME="$SESSION_NAME" \
    CUDA_DEVICE="$CUDA_DEVICE" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" \
    NUM_WORKERS="$NUM_WORKERS" \
    SOLVER_STEPS="$SOLVER_STEPS" \
    EVAL_SEED="$EVAL_SEED" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "GPU: $CUDA_DEVICE"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Result: $OUTPUT_JSON"
  exit 0
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Result already exists; refusing to overwrite: $OUTPUT_JSON" >&2
  exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) DirectActionFlow 90k full rollout evaluation start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "checkpoint=$EVAL_CKPT weights=ema"
  echo "protocol=context11 native_horizon15 commitment5 rollout80 solver_steps=$SOLVER_STEPS"
  echo "evaluation=batches${EVAL_MAX_BATCHES}_batch_size${EVAL_BATCH_SIZE}_max_scenes$((EVAL_MAX_BATCHES * EVAL_BATCH_SIZE))"
  echo "metric=mean_per_scene_nonfocus_xy_ADE_over_all_80_steps seed=$EVAL_SEED"

  "$PYTHON" "$EVAL_SCRIPT" \
    --checkpoint "$EVAL_CKPT" \
    --val_data_dir "$VAL_DATA" \
    --output_json "$OUTPUT_JSON" \
    --device cuda \
    --weights ema \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_max_batches "$EVAL_MAX_BATCHES" \
    --num_workers "$NUM_WORKERS" \
    --rollout_steps 80 \
    --solver_steps "$SOLVER_STEPS" \
    --seed "$EVAL_SEED" \
    --log_every 8

  echo "===== $(date) DirectActionFlow 90k full rollout evaluation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
