#!/usr/bin/env bash
# Evaluate the final map-conditioned world model from one context frame on CUDA 0.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"

RUN_NAME="waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_b2_norecon_mapx1_1m"
EVAL_CKPT="${EVAL_CKPT:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME/final_step_01000000.pt}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/train}"
VAL_DATA="${VAL_DATA:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val}"

SESSION_NAME="${SESSION_NAME:-wm_mapx1_final_ctx1_horizons_cuda0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HORIZONS="${HORIZONS:-10 30 50 80 90}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/evaluation/reports/$RUN_NAME}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/final_ctx01_horizons_10_30_50_80_90_batches${EVAL_MAX_BATCHES}.json}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/evaluation}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}_final_ctx01_horizons_cuda0.log}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_file "$PYTHON"
require_file "$EVAL_SCRIPT"
require_file "$EVAL_CKPT"
require_file "$TOKENIZER_CKPT"
require_dir "$TRAIN_DATA"
require_dir "$VAL_DATA"
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available." >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
  fi

  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  launch_cmd=(
    env
    "RUN_INSIDE_TMUX=1"
    "REPO_ROOT=$REPO_ROOT"
    "PYTHON=$PYTHON"
    "EVAL_CKPT=$EVAL_CKPT"
    "TOKENIZER_CKPT=$TOKENIZER_CKPT"
    "TRAIN_DATA=$TRAIN_DATA"
    "VAL_DATA=$VAL_DATA"
    "SESSION_NAME=$SESSION_NAME"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "HORIZONS=$HORIZONS"
    "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
    "EVAL_MAX_BATCHES=$EVAL_MAX_BATCHES"
    "NUM_WORKERS=$NUM_WORKERS"
    "OUT_DIR=$OUT_DIR"
    "OUTPUT_JSON=$OUTPUT_JSON"
    "LOG_DIR=$LOG_DIR"
    "LOG=$LOG"
    bash "$SCRIPT_PATH"
  )
  printf -v tmux_command '%q ' "${launch_cmd[@]}"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on

  echo "Started tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: $LOG"
  echo "Results: $OUTPUT_JSON"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1

{
  echo
  echo "===== $(date) ====="
  echo "session=$SESSION_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "eval_ckpt=$EVAL_CKPT"
  echo "tokenizer_ckpt=$TOKENIZER_CKPT"
  echo "val_data=$VAL_DATA"
  echo "eval_ctx=1 horizons=$HORIZONS"
  echo "eval_seq_len=91 max_rollout_window=11"
  echo "eval_batch_size=$EVAL_BATCH_SIZE eval_max_batches=$EVAL_MAX_BATCHES"
  echo "output_json=$OUTPUT_JSON"
  echo "========================"

  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$TRAIN_DATA" \
    --val_data_dir "$VAL_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" \
    --eval_ckpt "$EVAL_CKPT" \
    --device cuda \
    --seed 0 \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_max_batches "$EVAL_MAX_BATCHES" \
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
    --dynamics_attend_map \
    --map_cross_every 1 \
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
