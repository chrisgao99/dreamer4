#!/usr/bin/env bash
# Sequential CUDA 2/3 pipeline: pretrain frozen P(q|z), then train the 1M-step V1.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TORCHRUN="${TORCHRUN:-/p/yufeng/.conda/envs/dreamer4/bin/torchrun}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
READER_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_semantic_reader.py"
WORLD_MODEL_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_motion_latent_v1.py"

TOKENIZER_RUN="ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/$TOKENIZER_RUN/best.pt}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"

READER_RUN_NAME="${READER_RUN_NAME:-waymo_semantic_reader_agent32_zonly_d256_depth2_b8_20k_v1}"
WM_RUN_NAME="${WM_RUN_NAME:-waymo_motion_latent_v1_qkin_preader_ctx1ctx11_b8_mapx1_1m}"
SESSION_NAME="${SESSION_NAME:-wm_motion_latent_v1_cuda23}"
READER_CKPT_DIR="${READER_CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$READER_RUN_NAME}"
WM_CKPT_DIR="${WM_CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$WM_RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/wm}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_DIR/${WM_RUN_NAME}_pipeline.log}"
READER_LOG="${READER_LOG:-$LOG_DIR/${READER_RUN_NAME}.log}"
WM_LOG="${WM_LOG:-$LOG_DIR/${WM_RUN_NAME}.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
READER_BATCH_SIZE="${READER_BATCH_SIZE:-4}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
READER_MAX_STEPS="${READER_MAX_STEPS:-20000}"
WM_MAX_STEPS="${WM_MAX_STEPS:-1000000}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
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
require_file "$TORCHRUN"
require_file "$TOKENIZER_CKPT"
require_file "$READER_SCRIPT"
require_file "$WORLD_MODEL_SCRIPT"
require_dir "$DATA_ROOT/train"
require_dir "$DATA_ROOT/val"
mkdir -p "$READER_CKPT_DIR" "$WM_CKPT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  launch_env=(
    "RUN_INSIDE_TMUX=1"
    "REPO_ROOT=$REPO_ROOT"
    "PYTHON=$PYTHON"
    "TORCHRUN=$TORCHRUN"
    "TOKENIZER_CKPT=$TOKENIZER_CKPT"
    "DATA_ROOT=$DATA_ROOT"
    "READER_RUN_NAME=$READER_RUN_NAME"
    "WM_RUN_NAME=$WM_RUN_NAME"
    "SESSION_NAME=$SESSION_NAME"
    "READER_CKPT_DIR=$READER_CKPT_DIR"
    "WM_CKPT_DIR=$WM_CKPT_DIR"
    "LOG_DIR=$LOG_DIR"
    "PIPELINE_LOG=$PIPELINE_LOG"
    "READER_LOG=$READER_LOG"
    "WM_LOG=$WM_LOG"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "NPROC_PER_NODE=$NPROC_PER_NODE"
    "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    "READER_BATCH_SIZE=$READER_BATCH_SIZE"
    "WM_BATCH_SIZE=$WM_BATCH_SIZE"
    "NUM_WORKERS=$NUM_WORKERS"
    "READER_MAX_STEPS=$READER_MAX_STEPS"
    "WM_MAX_STEPS=$WM_MAX_STEPS"
    "USE_WANDB=$USE_WANDB"
    "WANDB_MODE=$WANDB_MODE"
  )
  launch_cmd=(env "${launch_env[@]}" bash "$SCRIPT_PATH")
  printf -v tmux_command '%q ' "${launch_cmd[@]}"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  echo "Started tmux session: $SESSION_NAME"
  echo "Pipeline log: $PIPELINE_LOG"
  echo "Reader log: $READER_LOG"
  echo "World-model log: $WM_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS WANDB_MODE PYTHONUNBUFFERED=1

reader_args=(
  --standalone
  --nproc_per_node "$NPROC_PER_NODE"
  "$READER_SCRIPT"
  --data_dir "$DATA_ROOT/train"
  --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --ckpt_dir "$READER_CKPT_DIR"
  --batch_size "$READER_BATCH_SIZE"
  --eval_batch_size "$READER_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --reader_context 32
  --reader_depth 2
  --max_steps "$READER_MAX_STEPS"
  --lr 1e-4
  --weight_decay 1e-4
  --amp_dtype bf16
  --log_every 100
  --eval_every 1000
  --eval_batches 16
  --save_every 5000
  --wandb_project waymo-world-model
  --wandb_run_name "$READER_RUN_NAME"
)
if [[ -f "$READER_CKPT_DIR/latest.pt" ]]; then
  reader_args+=(--resume "$READER_CKPT_DIR/latest.pt")
fi
if is_truthy "$USE_WANDB"; then
  reader_args+=(--wandb)
fi

{
  echo "===== $(date) pipeline start ====="
  echo "cuda=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE global_batch=$((READER_BATCH_SIZE * NPROC_PER_NODE))"
  echo "stage=semantic_reader max_steps=$READER_MAX_STEPS"
} | tee -a "$PIPELINE_LOG"

"$TORCHRUN" "${reader_args[@]}" 2>&1 | tee -a "$READER_LOG"

if [[ ! -f "$READER_CKPT_DIR/best.pt" ]]; then
  echo "Semantic reader finished without best.pt; refusing to start world model." | tee -a "$PIPELINE_LOG" >&2
  exit 1
fi

wm_args=(
  --standalone
  --nproc_per_node "$NPROC_PER_NODE"
  "$WORLD_MODEL_SCRIPT"
  --data_dir "$DATA_ROOT/train"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --semantic_reader_ckpt "$READER_CKPT_DIR/best.pt"
  --ckpt_dir "$WM_CKPT_DIR"
  --batch_size "$WM_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_steps "$WM_MAX_STEPS"
  --rollout_end 90
  --max_context 11
  --d_model 512
  --depth 8
  --n_heads 8
  --time_every 4
  --map_cross_every 1
  --packing_factor 2
  --n_register 8
  --k_max 64
  --kinematic_dt 0.1
  --motion_weight 1.0
  --motion_validity_weight 0.2
  --consistency_weight 0.1
  --bootstrap_start 20000
  --bootstrap_ramp_end 60000
  --bootstrap_weight 0.1
  --lr 1e-4
  --weight_decay 1e-2
  --grad_clip 1.0
  --amp_dtype bf16
  --log_every 100
  --save_every 50000
  --ego_action_source focus
  --ego_action_normalization raw
  --wandb_project waymo-world-model
  --wandb_run_name "$WM_RUN_NAME"
)
if [[ -f "$WM_CKPT_DIR/latest.pt" ]]; then
  wm_args+=(--resume "$WM_CKPT_DIR/latest.pt")
fi
if is_truthy "$USE_WANDB"; then
  wm_args+=(--wandb)
fi

{
  echo "===== $(date) semantic reader complete ====="
  echo "stage=world_model max_steps=$WM_MAX_STEPS global_batch=$((WM_BATCH_SIZE * NPROC_PER_NODE))"
  echo "reader_ckpt=$READER_CKPT_DIR/best.pt"
} | tee -a "$PIPELINE_LOG"

"$TORCHRUN" "${wm_args[@]}" 2>&1 | tee -a "$WM_LOG"

echo "===== $(date) pipeline complete =====" | tee -a "$PIPELINE_LOG"

