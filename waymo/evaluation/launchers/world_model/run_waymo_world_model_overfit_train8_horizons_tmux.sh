#!/usr/bin/env bash
# Evaluate one of the two 8-scene overfit checkpoints on its training subset.

set -euo pipefail

VARIANT="${1:-}"
if [[ "$VARIANT" != "self05" && "$VARIANT" != "self0" ]]; then
  echo "Usage: $0 {self05|self0}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b32_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_decmap_noamp/best.pt"

case "$VARIANT" in
  self05)
    RUN_NAME="wm_overfit_scenes8_randwin_self05_ctx1_h10_20260720"
    CUDA_DEVICE="${CUDA_DEVICE:-0}"
    ;;
  self0)
    RUN_NAME="wm_overfit_scenes8_randwin_self0_ctx1_h10_20260720"
    CUDA_DEVICE="${CUDA_DEVICE:-1}"
    ;;
esac

EVAL_CKPT="$REPO_ROOT/waymo/checkpoints/world_model_overfit/$RUN_NAME/final_step_00050000.pt"
TRAIN_DATA="$REPO_ROOT/waymo/data/world_model_overfit_subsets/$RUN_NAME/train"
SESSION_NAME="${SESSION_NAME:-${RUN_NAME}_train8_horizons_cuda${CUDA_DEVICE}}"
HORIZONS="${HORIZONS:-10 30 50 80 90}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"

OUT_DIR="$REPO_ROOT/waymo/evaluation/reports/$RUN_NAME"
OUTPUT_JSON="$OUT_DIR/final_step_00050000_train8_ctx01_horizons_10_30_50_80_90.json"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG="$LOG_DIR/${RUN_NAME}_final_step_00050000_train8_ctx01_horizons_cuda${CUDA_DEVICE}.log"

for required_file in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$EVAL_CKPT"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done
if [[ ! -d "$TRAIN_DATA" ]]; then
  echo "Missing training subset: $TRAIN_DATA" >&2
  exit 1
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

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
    SESSION_NAME="$SESSION_NAME" \
    HORIZONS="$HORIZONS" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    bash "$SCRIPT_PATH" "$VARIANT"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $LOG"
  echo "Results: $OUTPUT_JSON"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1

{
  echo "===== $(date) ====="
  echo "session=$SESSION_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "eval_ckpt=$EVAL_CKPT"
  echo "train_data=$TRAIN_DATA samples=$(find "$TRAIN_DATA" -maxdepth 1 -type l | wc -l)"
  echo "eval_ctx=1 horizons=$HORIZONS eval_seq_len=91"
  echo "eval_batch_size=$EVAL_BATCH_SIZE eval_max_batches=0"
  echo "output_json=$OUTPUT_JSON"
  echo "========================"

  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$TRAIN_DATA" \
    --val_data_dir "$TRAIN_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" \
    --eval_ckpt "$EVAL_CKPT" \
    --device cuda \
    --seed 0 \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_max_batches 0 \
    --num_workers "$NUM_WORKERS" \
    --eval_seq_len 91 \
    --eval_ctx 1 \
    --horizons "$HORIZONS" \
    --max_rollout_window 11 \
    --d_model_dyn 512 \
    --dyn_depth 8 \
    --n_heads 8 \
    --packing_factor 2 \
    --n_register 8 \
    --time_every 4 \
    --k_max 64 \
    --eval_schedule shortcut \
    --eval_d 0.25 \
    --use_ego_actions \
    --ego_action_source focus \
    --ego_action_normalization raw \
    --no-ego_action_clamp \
    --agent_far_weight 0.25 \
    --agent_near_radius_m 50.0 \
    --agent_distance_source focus \
    --output_json "$OUTPUT_JSON"

  echo "Finished at $(date)"
} 2>&1 | tee -a "$LOG"
