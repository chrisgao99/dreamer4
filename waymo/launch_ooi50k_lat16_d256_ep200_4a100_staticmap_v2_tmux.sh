#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
RUN_NAME="${RUN_NAME:-ooi50k_lat16_d256_ep200_4a100_staticmap_v2_lossfix}"
SESSION_NAME="${SESSION_NAME:-ooi50k_staticmap_v2_ep200_lossfix}"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}.log}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/waymo/wandb}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-online}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-200}"
D_MODEL="${D_MODEL:-256}"
DEPTH="${DEPTH:-4}"
DECODER_DEPTH="${DECODER_DEPTH:-4}"
N_LATENTS="${N_LATENTS:-16}"
D_BOTTLENECK="${D_BOTTLENECK:-32}"
TIME_WINDOW="${TIME_WINDOW:-32}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"

ENCODER_VARIANT="${ENCODER_VARIANT:-static_map_query}"
MAP_DEPTH="${MAP_DEPTH:-2}"
MAP_CROSS_EVERY="${MAP_CROSS_EVERY:-1}"
MAP_QUERY_TOKENS="${MAP_QUERY_TOKENS:-latent_agent}"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    echo "Log: $LOG"
    exit 0
  fi
  tmux new-session -d -s "$SESSION_NAME" \
    "RUN_INSIDE_TMUX=1 bash '$0'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  echo "Detach with: Ctrl-b then d"
  echo "Log: $LOG"
  exit 0
fi

cd "$REPO_ROOT"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export WANDB_MODE
export WANDB_DIR

train_args=(
  waymo/train_waymo_vector_tokenizer.py
  --data_dir "$DATA_DIR/train"
  --val_data_dir "$DATA_DIR/val"
  --ckpt_dir "$CKPT_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --time_window "$TIME_WINDOW"
  --epochs "$EPOCHS"
  --d_model "$D_MODEL"
  --depth "$DEPTH"
  --decoder_depth "$DECODER_DEPTH"
  --n_latents "$N_LATENTS"
  --d_bottleneck "$D_BOTTLENECK"
  --encoder_variant "$ENCODER_VARIANT"
  --map_depth "$MAP_DEPTH"
  --map_cross_every "$MAP_CROSS_EVERY"
  --map_query_tokens "$MAP_QUERY_TOKENS"
  --log_every "$LOG_EVERY"
  --eval_every "$EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --wandb
  --wandb_project waymo-vector-tokenizer
  --wandb_run_name "$RUN_NAME"
)

if [[ -f "$CKPT_DIR/latest.pt" ]]; then
  train_args+=(--resume "$CKPT_DIR/latest.pt")
fi

{
  echo
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME"
  echo "session=$SESSION_NAME"
  echo "cuda=$CUDA_VISIBLE_DEVICES"
  echo "wandb_mode=$WANDB_MODE"
  echo "data_dir=$DATA_DIR"
  echo "ckpt_dir=$CKPT_DIR"
  echo "log=$LOG"
  echo "batch_size=$BATCH_SIZE epochs=$EPOCHS d_model=$D_MODEL depth=$DEPTH decoder_depth=$DECODER_DEPTH n_latents=$N_LATENTS d_bottleneck=$D_BOTTLENECK"
  echo "encoder_variant=$ENCODER_VARIANT map_depth=$MAP_DEPTH map_cross_every=$MAP_CROSS_EVERY map_query_tokens=$MAP_QUERY_TOKENS"
  echo "loss_fix=normalized_agent_targets_no_double_transpose"
  echo "========================"
} | tee -a "$LOG"

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${train_args[@]}" \
  2>&1 | tee -a "$LOG"
