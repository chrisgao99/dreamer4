#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
RUN_NAME="${RUN_NAME:-ooi50k_lat16_d256_ep200_2a100_staticmap_v2_fulltraj_trajloss}"
SESSION_NAME="${SESSION_NAME:-ooi50k_staticmap_v2_fulltraj_trajloss}"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}.log}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/waymo/wandb}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
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
TIME_WINDOW="${TIME_WINDOW:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"

ENCODER_VARIANT="${ENCODER_VARIANT:-static_map_query}"
MAP_DEPTH="${MAP_DEPTH:-2}"
MAP_CROSS_EVERY="${MAP_CROSS_EVERY:-1}"
MAP_QUERY_TOKENS="${MAP_QUERY_TOKENS:-latent_agent}"
AGENT_DELTA_XY_WEIGHT="${AGENT_DELTA_XY_WEIGHT:-5}"
AGENT_FDE_XY_WEIGHT="${AGENT_FDE_XY_WEIGHT:-2}"
FOCUS_AGENT_WEIGHT="${FOCUS_AGENT_WEIGHT:-4}"

check_required_paths() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found or not executable: $PYTHON" >&2
    exit 1
  fi
  if [[ ! -f "$REPO_ROOT/waymo/training/train_waymo_vector_tokenizer.py" ]]; then
    echo "Training script not found under REPO_ROOT: $REPO_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" ]]; then
    echo "Expected dataset directories are missing:" >&2
    echo "  $DATA_DIR/train" >&2
    echo "  $DATA_DIR/val" >&2
    echo "Set DATA_DIR to the directory containing train/ and val/ before launching." >&2
    exit 1
  fi
  if ! compgen -G "$DATA_DIR/train/*.npz" >/dev/null; then
    echo "No training .npz files found in: $DATA_DIR/train" >&2
    exit 1
  fi
  if ! compgen -G "$DATA_DIR/val/*.npz" >/dev/null; then
    echo "No validation .npz files found in: $DATA_DIR/val" >&2
    exit 1
  fi
}

check_required_paths

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"
  if ! command -v tmux >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
      module load tmux >/dev/null 2>&1 || true
    elif [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck source=/dev/null
      source /etc/profile.d/modules.sh
      module load tmux >/dev/null 2>&1 || true
    fi
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found after trying: module load tmux" >&2
    echo "running in the current shell instead."
    RUN_INSIDE_TMUX=1 exec bash "$0"
  fi
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
  waymo/training/train_waymo_vector_tokenizer.py
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
  --agent_delta_xy_weight "$AGENT_DELTA_XY_WEIGHT"
  --agent_fde_xy_weight "$AGENT_FDE_XY_WEIGHT"
  --focus_agent_weight "$FOCUS_AGENT_WEIGHT"
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
  echo "batch_size=$BATCH_SIZE epochs=$EPOCHS d_model=$D_MODEL depth=$DEPTH decoder_depth=$DECODER_DEPTH n_latents=$N_LATENTS d_bottleneck=$D_BOTTLENECK time_window=$TIME_WINDOW"
  echo "encoder_variant=$ENCODER_VARIANT map_depth=$MAP_DEPTH map_cross_every=$MAP_CROSS_EVERY map_query_tokens=$MAP_QUERY_TOKENS"
  echo "agent_delta_xy_weight=$AGENT_DELTA_XY_WEIGHT agent_fde_xy_weight=$AGENT_FDE_XY_WEIGHT focus_agent_weight=$FOCUS_AGENT_WEIGHT"
  echo "loss_fix=normalized_agent_targets_no_double_transpose"
  echo "========================"
} | tee -a "$LOG"

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "${train_args[@]}" \
  2>&1 | tee -a "$LOG"
