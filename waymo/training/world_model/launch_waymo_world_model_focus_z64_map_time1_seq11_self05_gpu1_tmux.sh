#!/usr/bin/env bash
# Train the focus-only z=1x64 world model with the main aligned configuration:
# 50k data, 11-step random windows, time attention every block, explicit map
# conditioning, 50/50 empirical/self shortcut training, no actions, and no
# decoded training loss.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"

TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/focus_tokenizer_c_z1x64_raw_map_lr1e4/best.pt}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"

RUN_NAME="${RUN_NAME:-wm_ooi50k_focus_z64_map_time1_seq11_self05_d512_l8_b8_bf16_gpu1}"
SESSION_NAME="${SESSION_NAME:-wm_focus_z64_map_t1_s05_gpu1}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/world_model}"
LOG="${LOG:-$LOG_DIR/$RUN_NAME.log}"
RESUME="${RESUME:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WANDB="${USE_WANDB:-1}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-11}"
MAX_STEPS="${MAX_STEPS:-500000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"

D_MODEL_DYN="${D_MODEL_DYN:-512}"
DYN_DEPTH="${DYN_DEPTH:-8}"
N_HEADS="${N_HEADS:-8}"
PACKING_FACTOR=1
N_REGISTER="${N_REGISTER:-8}"
TIME_EVERY="${TIME_EVERY:-1}"
MAP_CROSS_EVERY="${MAP_CROSS_EVERY:-1}"
K_MAX="${K_MAX:-64}"
SELF_FRACTION="${SELF_FRACTION:-0.5}"

LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-25000}"
SAVE_EVERY="${SAVE_EVERY:-25000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-128}"
EVAL_CTX="${EVAL_CTX:-1}"
EVAL_HORIZON="${EVAL_HORIZON:-10}"

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
require_file "$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
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
  waymo/training/world_model/train_waymo_world_model.py
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
  --dynamics_variant standard
  --dyn_depth "$DYN_DEPTH"
  --n_heads "$N_HEADS"
  --dropout 0.0
  --mlp_ratio 4.0
  --scale_pos_embeds
  --packing_factor "$PACKING_FACTOR"
  --n_register "$N_REGISTER"
  --time_every "$TIME_EVERY"
  --dynamics_attend_map
  --map_cross_every "$MAP_CROSS_EVERY"
  --k_max "$K_MAX"
  --bootstrap_start 0
  --self_fraction "$SELF_FRACTION"
  --train_objective shortcut
  --tf_context 10
  --train_decoded_loss_weight 0.0
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --max_steps "$MAX_STEPS"
  --grad_accum 1
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
  echo "tokenizer_mode=frozen_focus_latent_z64; n_latents=1; d_bottleneck=64"
  echo "train_data=$DATA_ROOT/train"
  echo "val_data=$DATA_ROOT/val"
  echo "ckpt_dir=$CKPT_DIR"
  echo "resume=${RESUME:-none}"
  echo "batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE num_workers=$NUM_WORKERS"
  echo "seq_len=$SEQ_LEN random_time_window_start=1 max_steps=$MAX_STEPS"
  echo "d_model_dyn=$D_MODEL_DYN dyn_depth=$DYN_DEPTH n_heads=$N_HEADS packing_factor=$PACKING_FACTOR n_register=$N_REGISTER"
  echo "time_every=$TIME_EVERY dynamics_attend_map=1 map_cross_every=$MAP_CROSS_EVERY"
  echo "objective=shortcut k_max=$K_MAX self_fraction=$SELF_FRACTION decoded_loss_weight=0 action=none amp_dtype=bf16"
  echo "eval_ctx=$EVAL_CTX eval_horizon=$EVAL_HORIZON eval_max_batches=$EVAL_MAX_BATCHES eval_every=$EVAL_EVERY"
  echo "wandb=$USE_WANDB wandb_mode=$WANDB_MODE"
  echo "========================"
} | tee -a "$LOG"

"$PYTHON" "${train_args[@]}" 2>&1 | tee -a "$LOG"
