#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"

RUN_NAME="wm_ooi_all_tok_lat64_decmap_shortcut_seq32_randstart_d512_l8_b8_bf16_gpu2"
EVAL_CKPT="${EVAL_CKPT:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME/step_00075000.pt}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b32_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_decmap_noamp/best.pt}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_training_all/train}"
VAL_DATA="${VAL_DATA:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_training_all/val}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"

SESSION_NAME="${SESSION_NAME:-wm75k_context_rollouts}"
GPU_CONTEXT_1="${GPU_CONTEXT_1:-0}"
GPU_CONTEXT_11="${GPU_CONTEXT_11:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/evaluation/reports/wm75k_context_rollouts}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/evaluation}"
LOG_CONTEXT_1="${LOG_CONTEXT_1:-$LOG_DIR/wm75k_context_01.log}"
LOG_CONTEXT_11="${LOG_CONTEXT_11:-$LOG_DIR/wm75k_context_11.log}"

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

  SCRIPT_PATH="$(realpath "$0")"
  tmux new-session -d -s "$SESSION_NAME" -n context_01 \
    "RUN_INSIDE_TMUX=1 EVAL_GROUP=context_01 CUDA_VISIBLE_DEVICES='$GPU_CONTEXT_1' bash '$SCRIPT_PATH'"
  tmux new-window -d -t "$SESSION_NAME" -n context_11 \
    "RUN_INSIDE_TMUX=1 EVAL_GROUP=context_11 CUDA_VISIBLE_DEVICES='$GPU_CONTEXT_11' bash '$SCRIPT_PATH'"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Window context_01: GPU $GPU_CONTEXT_1, log $LOG_CONTEXT_1"
  echo "Window context_11: GPU $GPU_CONTEXT_11, log $LOG_CONTEXT_11"
  echo "Results: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set for each tmux evaluation window}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

common_args=(
  --data_dir "$TRAIN_DATA"
  --val_data_dir "$VAL_DATA"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --eval_ckpt "$EVAL_CKPT"
  --device cuda
  --seed "$SEED"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --eval_max_batches "$EVAL_MAX_BATCHES"
  --num_workers "$NUM_WORKERS"
  --eval_seq_len 91
  --max_rollout_window 32
  --d_model_dyn 512
  --dyn_depth 8
  --n_heads 8
  --packing_factor 2
  --n_register 8
  --time_every 4
  --k_max 64
  --eval_schedule shortcut
  --eval_d 0.25
)

case "${EVAL_GROUP:-}" in
  context_01)
    CONTEXT=1
    HORIZONS="10 30 50 80 90"
    OUTPUT_JSON="$OUT_DIR/context_01_horizons_10_30_50_80_90.json"
    GROUP_LOG="$LOG_CONTEXT_1"
    ;;
  context_11)
    CONTEXT=11
    HORIZONS="10 30 50 80"
    OUTPUT_JSON="$OUT_DIR/context_11_horizons_10_30_50_80.json"
    GROUP_LOG="$LOG_CONTEXT_11"
    ;;
  *)
    echo "Unknown EVAL_GROUP=${EVAL_GROUP:-unset}; expected context_01 or context_11" >&2
    exit 1
    ;;
esac

{
  echo
  echo "===== $(date) ====="
  echo "session=$SESSION_NAME"
  echo "eval_group=$EVAL_GROUP context=$CONTEXT horizons=$HORIZONS"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "eval_ckpt=$EVAL_CKPT"
  echo "tokenizer_ckpt=$TOKENIZER_CKPT"
  echo "decoder_map_conditioning=loaded_from_tokenizer_checkpoint"
  echo "val_data=$VAL_DATA"
  echo "eval_batch_size=$EVAL_BATCH_SIZE eval_max_batches=$EVAL_MAX_BATCHES"
  echo "eval_seq_len=91 max_rollout_window=32 seed=$SEED"

  "$PYTHON" "$EVAL_SCRIPT" \
    "${common_args[@]}" \
    --eval_ctx "$CONTEXT" \
    --horizons "$HORIZONS" \
    --output_json "$OUTPUT_JSON"

  echo
  echo "Finished $EVAL_GROUP at $(date)"
} 2>&1 | tee -a "$GROUP_LOG"
