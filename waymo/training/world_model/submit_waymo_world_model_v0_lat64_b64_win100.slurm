#!/bin/bash
#SBATCH --job-name=wm_v0_l64b64_w100
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200|a100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/latest.pt}"
RUN_NAME="${RUN_NAME:-waymo_wm_v0_lat64_b64_win100_ctx11_h80}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/wm}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/waymo/wandb}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-online}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-100000}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-8}"
SAVE_EVERY="${SAVE_EVERY:-5000}"

SEQ_LEN="${SEQ_LEN:-100}"
RANDOM_TIME_WINDOW_START="${RANDOM_TIME_WINDOW_START:-0}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-100}"
EVAL_CTX="${EVAL_CTX:-11}"
EVAL_HORIZON="${EVAL_HORIZON:-80}"
MAX_ROLLOUT_WINDOW="${MAX_ROLLOUT_WINDOW:-100}"
EVAL_SCHEDULE="${EVAL_SCHEDULE:-shortcut}"
EVAL_D="${EVAL_D:-0.25}"

D_MODEL_DYN="${D_MODEL_DYN:-512}"
DYN_DEPTH="${DYN_DEPTH:-8}"
N_HEADS="${N_HEADS:-8}"
TIME_EVERY="${TIME_EVERY:-4}"
PACKING_FACTOR="${PACKING_FACTOR:-2}"
N_REGISTER="${N_REGISTER:-8}"
K_MAX="${K_MAX:-64}"
SELF_FRACTION="${SELF_FRACTION:-0.857142857}"
BOOTSTRAP_START="${BOOTSTRAP_START:-0}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-shortcut}"
TF_CONTEXT="${TF_CONTEXT:-10}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

AGENT_XY_LOSS="${AGENT_XY_LOSS:-smooth_l1}"
AGENT_XY_PARAMETERIZATION="${AGENT_XY_PARAMETERIZATION:-absolute}"
FOCUS_AGENT_WEIGHT="${FOCUS_AGENT_WEIGHT:-4}"
AGENT_KINEMATIC_XY_WEIGHT="${AGENT_KINEMATIC_XY_WEIGHT:-5}"
AGENT_SPEED_YAW_KINEMATIC_WEIGHT="${AGENT_SPEED_YAW_KINEMATIC_WEIGHT:-2}"
USE_EGO_ACTIONS="${USE_EGO_ACTIONS:-0}"
EGO_ACTION_SOURCE="${EGO_ACTION_SOURCE:-focus}"
EGO_ACTION_NORMALIZATION="${EGO_ACTION_NORMALIZATION:-scaled}"
EGO_ACTION_CLAMP="${EGO_ACTION_CLAMP:-1}"
EGO_ACTION_XY_SCALE="${EGO_ACTION_XY_SCALE:-5.0}"
EGO_ACTION_YAW_SCALE="${EGO_ACTION_YAW_SCALE:-3.141592653589793}"
EGO_ACTION_SPEED_SCALE="${EGO_ACTION_SPEED_SCALE:-30.0}"
AGENT_FAR_WEIGHT="${AGENT_FAR_WEIGHT:-1.0}"
AGENT_NEAR_RADIUS_M="${AGENT_NEAR_RADIUS_M:-50.0}"
AGENT_DISTANCE_SOURCE="${AGENT_DISTANCE_SOURCE:-focus}"
TRAIN_DECODED_LOSS_WEIGHT="${TRAIN_DECODED_LOSS_WEIGHT:-0.0}"

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found or not executable: $PYTHON" >&2
  exit 1
fi
require_file "$TRAIN_SCRIPT"
require_file "$TOKENIZER_CKPT"
require_dir "$DATA_ROOT/train"
require_dir "$DATA_ROOT/val"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"

cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export WANDB_MODE
export WANDB_DIR

echo "===== Slurm job start: $(date) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN_NAME=$RUN_NAME"
echo "DATA_ROOT=$DATA_ROOT"
echo "TOKENIZER_CKPT=$TOKENIZER_CKPT"
echo "CKPT_DIR=$CKPT_DIR"
echo "SEQ_LEN=$SEQ_LEN EVAL_CTX=$EVAL_CTX EVAL_HORIZON=$EVAL_HORIZON MAX_ROLLOUT_WINDOW=$MAX_ROLLOUT_WINDOW"
echo "RANDOM_TIME_WINDOW_START=$RANDOM_TIME_WINDOW_START"
echo "TRAIN_OBJECTIVE=$TRAIN_OBJECTIVE TF_CONTEXT=$TF_CONTEXT EVAL_SCHEDULE=$EVAL_SCHEDULE EVAL_D=$EVAL_D"
echo "BATCH_SIZE=$BATCH_SIZE per process, NPROC_PER_NODE=$NPROC_PER_NODE"
echo "USE_EGO_ACTIONS=$USE_EGO_ACTIONS EGO_ACTION_SOURCE=$EGO_ACTION_SOURCE EGO_ACTION_NORMALIZATION=$EGO_ACTION_NORMALIZATION EGO_ACTION_CLAMP=$EGO_ACTION_CLAMP"
echo "AGENT_FAR_WEIGHT=$AGENT_FAR_WEIGHT AGENT_NEAR_RADIUS_M=$AGENT_NEAR_RADIUS_M TRAIN_DECODED_LOSS_WEIGHT=$TRAIN_DECODED_LOSS_WEIGHT"

train_args=(
  "$TRAIN_SCRIPT"
  --data_dir "$DATA_ROOT/train"
  --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --ckpt_dir "$CKPT_DIR"
  --seq_len "$SEQ_LEN"
  --eval_seq_len "$EVAL_SEQ_LEN"
  --eval_ctx "$EVAL_CTX"
  --eval_horizon "$EVAL_HORIZON"
  --max_rollout_window "$MAX_ROLLOUT_WINDOW"
  --eval_schedule "$EVAL_SCHEDULE"
  --eval_d "$EVAL_D"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_steps "$MAX_STEPS"
  --log_every "$LOG_EVERY"
  --eval_every "$EVAL_EVERY"
  --eval_max_batches "$EVAL_MAX_BATCHES"
  --save_every "$SAVE_EVERY"
  --d_model_dyn "$D_MODEL_DYN"
  --dyn_depth "$DYN_DEPTH"
  --n_heads "$N_HEADS"
  --time_every "$TIME_EVERY"
  --packing_factor "$PACKING_FACTOR"
  --n_register "$N_REGISTER"
  --k_max "$K_MAX"
  --self_fraction "$SELF_FRACTION"
  --bootstrap_start "$BOOTSTRAP_START"
  --train_objective "$TRAIN_OBJECTIVE"
  --tf_context "$TF_CONTEXT"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --amp_dtype "$AMP_DTYPE"
  --agent_xy_loss "$AGENT_XY_LOSS"
  --agent_xy_parameterization "$AGENT_XY_PARAMETERIZATION"
  --focus_agent_weight "$FOCUS_AGENT_WEIGHT"
  --agent_kinematic_xy_weight "$AGENT_KINEMATIC_XY_WEIGHT"
  --agent_speed_yaw_kinematic_weight "$AGENT_SPEED_YAW_KINEMATIC_WEIGHT"
  --ego_action_source "$EGO_ACTION_SOURCE"
  --ego_action_normalization "$EGO_ACTION_NORMALIZATION"
  --ego_action_xy_scale "$EGO_ACTION_XY_SCALE"
  --ego_action_yaw_scale "$EGO_ACTION_YAW_SCALE"
  --ego_action_speed_scale "$EGO_ACTION_SPEED_SCALE"
  --agent_far_weight "$AGENT_FAR_WEIGHT"
  --agent_near_radius_m "$AGENT_NEAR_RADIUS_M"
  --agent_distance_source "$AGENT_DISTANCE_SOURCE"
  --train_decoded_loss_weight "$TRAIN_DECODED_LOSS_WEIGHT"
  --wandb_project waymo-world-model
  --wandb_run_name "$RUN_NAME"
)

if is_truthy "$RANDOM_TIME_WINDOW_START"; then
  train_args+=(--random_time_window_start)
fi

if is_truthy "$USE_EGO_ACTIONS"; then
  train_args+=(--use_ego_actions)
fi

if is_truthy "$EGO_ACTION_CLAMP"; then
  train_args+=(--ego_action_clamp)
else
  train_args+=(--no-ego_action_clamp)
fi

if is_truthy "${WANDB:-1}"; then
  train_args+=(--wandb)
fi

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  "${train_args[@]}"

echo "===== Slurm job end: $(date) ====="
