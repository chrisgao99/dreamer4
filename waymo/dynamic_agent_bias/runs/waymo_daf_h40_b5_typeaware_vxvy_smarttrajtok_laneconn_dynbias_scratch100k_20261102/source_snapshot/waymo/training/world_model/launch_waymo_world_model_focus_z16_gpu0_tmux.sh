#!/usr/bin/env bash
# Train the standard Waymo world model on the frozen focus z=1x16 latent.
# Keep the training configuration aligned with
# launch_waymo_world_model_focus_agenttok_gpu2_tmux.sh for a fair ablation.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"

TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/focus_tokenizer_b_z1x16_raw_map/best.pt}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_training_all}"

RUN_NAME="${RUN_NAME:-wm_ooi_all_focus_z16_shortcut_seq32_randstart_d512_l8_b8_bf16_gpu0}"
SESSION_NAME="${SESSION_NAME:-wm_focus_z16_gpu0}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/world_model}"
LOG="${LOG:-$LOG_DIR/$RUN_NAME.log}"
RESUME="${RESUME:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WANDB="${USE_WANDB:-1}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-32}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"

D_MODEL_DYN="${D_MODEL_DYN:-512}"
DYN_DEPTH="${DYN_DEPTH:-8}"
N_HEADS="${N_HEADS:-8}"
PACKING_FACTOR=1
N_REGISTER="${N_REGISTER:-8}"
TIME_EVERY="${TIME_EVERY:-4}"
K_MAX="${K_MAX:-64}"
SELF_FRACTION="${SELF_FRACTION:-0.857142857}"

LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-8}"
EVAL_CTX="${EVAL_CTX:-11}"
EVAL_HORIZON="${EVAL_HORIZON:-21}"

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]
}

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
require_file "$TOKENIZER_CKPT"
require_file "$REPO_ROOT/waymo/training/train_waymo_world_model.py"
require_dir "$DATA_ROOT/train"
require_dir "$DATA_ROOT/val"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION_NAME" "RUN_INSIDE_TMUX=1 bash '$0'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: $LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS WANDB_MODE PYTHONUNBUFFERED=1

train_args=(
  waymo/training/train_waymo_world_model.py
  --data_dir "$DATA_ROOT/train"
  --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --ckpt_dir "$CKPT_DIR"
  --seed 0
  --seq_len "$SEQ_LEN"
  --random_time_window_start
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --d_model_dyn "$D_MODEL_DYN"
  --dyn_depth "$DYN_DEPTH"
  --n_heads "$N_HEADS"
  --packing_factor "$PACKING_FACTOR"
  --n_register "$N_REGISTER"
  --time_every "$TIME_EVERY"
  --k_max "$K_MAX"
  --bootstrap_start 0
  --self_fraction "$SELF_FRACTION"
  --train_objective shortcut
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --max_steps "$MAX_STEPS"
  --grad_clip 1.0
  --amp_dtype bf16
  --log_every "$LOG_EVERY"
  --eval_every "$EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --eval_max_batches "$EVAL_MAX_BATCHES"
  --eval_seq_len "$SEQ_LEN"
  --eval_ctx "$EVAL_CTX"
  --eval_horizon "$EVAL_HORIZON"
  --max_rollout_window "$SEQ_LEN"
  --eval_schedule shortcut
  --eval_d 0.25
  --wandb_project waymo-world-model
  --wandb_run_name "$RUN_NAME"
)

if [[ -n "$RESUME" ]]; then
  require_file "$RESUME"
  train_args+=(--resume "$RESUME")
elif [[ -f "$CKPT_DIR/latest.pt" ]]; then
  train_args+=(--resume "$CKPT_DIR/latest.pt")
  RESUME="$CKPT_DIR/latest.pt"
fi
if is_truthy "$USE_WANDB"; then
  train_args+=(--wandb)
fi

{
  echo
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME"
  echo "session=$SESSION_NAME"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "tokenizer_ckpt=$TOKENIZER_CKPT"
  echo "tokenizer_mode=frozen_focus_latent_z16; n_latents=1; d_bottleneck=16"
  echo "train_data=$DATA_ROOT/train"
  echo "val_data=$DATA_ROOT/val"
  echo "ckpt_dir=$CKPT_DIR"
  echo "resume=${RESUME:-none}"
  echo "batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE num_workers=$NUM_WORKERS"
  echo "seq_len=$SEQ_LEN random_time_window_start=1 max_steps=$MAX_STEPS"
  echo "d_model_dyn=$D_MODEL_DYN dyn_depth=$DYN_DEPTH n_heads=$N_HEADS packing_factor=$PACKING_FACTOR n_register=$N_REGISTER"
  echo "objective=shortcut k_max=$K_MAX self_fraction=$SELF_FRACTION amp_dtype=bf16"
  echo "eval_ctx=$EVAL_CTX eval_horizon=$EVAL_HORIZON eval_max_batches=$EVAL_MAX_BATCHES"
  echo "wandb=$USE_WANDB wandb_mode=$WANDB_MODE"
  echo "========================"
} | tee -a "$LOG"

"$PYTHON" "${train_args[@]}" 2>&1 | tee -a "$LOG"
