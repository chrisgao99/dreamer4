#!/usr/bin/env bash
# Run one focus-tokenizer experiment in the foreground on one visible GPU.

set -euo pipefail

REPRESENTATION="${1:?usage: $0 agent_token|latent_z16|latent_z64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/train_waymo_focus_tokenizer.py"

case "$REPRESENTATION" in
  agent_token)
    DEFAULT_RUN_NAME="focus_tokenizer_a_agent_token_raw_map"
    DEFAULT_D_LATENT=16
    ;;
  latent_z16)
    DEFAULT_RUN_NAME="focus_tokenizer_b_z1x16_raw_map"
    DEFAULT_D_LATENT=16
    ;;
  latent_z64)
    DEFAULT_RUN_NAME="focus_tokenizer_c_z1x64_raw_map_lr1e4"
    DEFAULT_D_LATENT=64
    ;;
  *)
    echo "Unknown representation: $REPRESENTATION" >&2
    exit 2
    ;;
esac
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python is missing or not executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Training script is missing: $TRAIN_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/val" ]]; then
  echo "Expected train/ and val/ below DATA_ROOT=$DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$CKPT_DIR"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/waymo/wandb}"
mkdir -p "$WANDB_DIR"

args=(
  "$TRAIN_SCRIPT"
  --representation "$REPRESENTATION"
  --data_dir "$DATA_ROOT/train"
  --val_data_dir "$DATA_ROOT/val"
  --ckpt_dir "$CKPT_DIR"
  --epochs "${EPOCHS:-200}"
  --max_steps "${MAX_STEPS:-150000}"
  --batch_size "${BATCH_SIZE:-64}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}"
  --num_workers "${NUM_WORKERS:-4}"
  --time_window "${TIME_WINDOW:-32}"
  --d_model "${D_MODEL:-256}"
  --d_latent "${D_LATENT:-$DEFAULT_D_LATENT}"
  --hidden_dim "${HIDDEN_DIM:-128}"
  --n_heads "${N_HEADS:-4}"
  --depth "${DEPTH:-4}"
  --decoder_depth "${DECODER_DEPTH:-2}"
  --map_depth "${MAP_DEPTH:-2}"
  --dropout "${DROPOUT:-0.05}"
  --mlp_ratio "${MLP_RATIO:-4.0}"
  --xy_weight "${XY_WEIGHT:-1.0}"
  --velocity_weight "${VELOCITY_WEIGHT:-0.5}"
  --yaw_weight "${YAW_WEIGHT:-0.5}"
  --valid_weight "${VALID_WEIGHT:-0.2}"
  --delta_xy_weight "${DELTA_XY_WEIGHT:-0.0}"
  --kinematic_xy_weight "${KINEMATIC_XY_WEIGHT:-5.0}"
  --speed_yaw_kinematic_weight "${SPEED_YAW_KINEMATIC_WEIGHT:-2.0}"
  --kinematic_dt "${KINEMATIC_DT:-0.1}"
  --lr "${LR:-3e-4}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}"
  --grad_clip "${GRAD_CLIP:-1.0}"
  --amp_dtype "${AMP_DTYPE:-bf16}"
  --log_every "${LOG_EVERY:-20}"
  --eval_every "${EVAL_EVERY:-500}"
  --eval_max_batches "${EVAL_MAX_BATCHES:-32}"
  --save_every "${SAVE_EVERY:-500}"
  --wandb_project "${WANDB_PROJECT:-waymo-focus-tokenizer}"
  --wandb_run_name "$RUN_NAME"
)

if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
if [[ "${RANDOM_TIME_WINDOW_START:-1}" == "0" ]]; then
  args+=(--no-random_time_window_start)
fi
if [[ "${WANDB:-1}" == "1" ]]; then
  args+=(--wandb)
fi

echo "representation=$REPRESENTATION"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "DATA_ROOT=$DATA_ROOT"
echo "CKPT_DIR=$CKPT_DIR"
echo "raw_feature_scaling=none"

exec "$PYTHON" "${args[@]}"
