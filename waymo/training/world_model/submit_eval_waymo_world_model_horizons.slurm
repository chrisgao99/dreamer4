#!/bin/bash
#SBATCH --job-name=wm_eval_horizons
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200|a100|a40|a6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-$HOME/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/latest.pt}"
RUN_NAME="${RUN_NAME:-waymo_wm_v0_lat64_b64_win100_ctx11_h80}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
EVAL_CKPT="${EVAL_CKPT:-$CKPT_DIR/step_00090000.pt}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/wm}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"

EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-100}"
EVAL_CTX="${EVAL_CTX:-11}"
HORIZONS="${HORIZONS:-10 30 50 80}"
MAX_ROLLOUT_WINDOW="${MAX_ROLLOUT_WINDOW:-100}"

D_MODEL_DYN="${D_MODEL_DYN:-512}"
DYN_DEPTH="${DYN_DEPTH:-8}"
N_HEADS="${N_HEADS:-8}"
TIME_EVERY="${TIME_EVERY:-4}"
PACKING_FACTOR="${PACKING_FACTOR:-2}"
N_REGISTER="${N_REGISTER:-8}"
K_MAX="${K_MAX:-64}"
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
require_file "$EVAL_SCRIPT"
require_file "$TOKENIZER_CKPT"
require_file "$EVAL_CKPT"
require_dir "$DATA_ROOT/train"
require_dir "$DATA_ROOT/val"

mkdir -p "$LOG_DIR" "$OUT_DIR"

cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS

ckpt_base="$(basename "$EVAL_CKPT" .pt)"
output_json="$OUT_DIR/${RUN_NAME}_${ckpt_base}_horizons.json"

echo "===== Slurm eval start: $(date) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN_NAME=$RUN_NAME"
echo "DATA_ROOT=$DATA_ROOT"
echo "TOKENIZER_CKPT=$TOKENIZER_CKPT"
echo "EVAL_CKPT=$EVAL_CKPT"
echo "HORIZONS=$HORIZONS"
echo "EVAL_MAX_BATCHES=$EVAL_MAX_BATCHES"
echo "OUTPUT_JSON=$output_json"
echo "USE_EGO_ACTIONS=$USE_EGO_ACTIONS EGO_ACTION_SOURCE=$EGO_ACTION_SOURCE EGO_ACTION_NORMALIZATION=$EGO_ACTION_NORMALIZATION EGO_ACTION_CLAMP=$EGO_ACTION_CLAMP"
echo "AGENT_FAR_WEIGHT=$AGENT_FAR_WEIGHT AGENT_NEAR_RADIUS_M=$AGENT_NEAR_RADIUS_M"

eval_args=(
  "$EVAL_SCRIPT"
  --data_dir "$DATA_ROOT/train"
  --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --eval_ckpt "$EVAL_CKPT"
  --output_json "$output_json"
  --eval_seq_len "$EVAL_SEQ_LEN"
  --eval_ctx "$EVAL_CTX"
  --horizons "$HORIZONS"
  --max_rollout_window "$MAX_ROLLOUT_WINDOW"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --eval_max_batches "$EVAL_MAX_BATCHES"
  --num_workers "$NUM_WORKERS"
  --d_model_dyn "$D_MODEL_DYN"
  --dyn_depth "$DYN_DEPTH"
  --n_heads "$N_HEADS"
  --time_every "$TIME_EVERY"
  --packing_factor "$PACKING_FACTOR"
  --n_register "$N_REGISTER"
  --k_max "$K_MAX"
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
)

if [[ "$USE_EGO_ACTIONS" == "1" || "$USE_EGO_ACTIONS" == "true" || "$USE_EGO_ACTIONS" == "TRUE" || "$USE_EGO_ACTIONS" == "yes" || "$USE_EGO_ACTIONS" == "YES" ]]; then
  eval_args+=(--use_ego_actions)
fi

if [[ "$EGO_ACTION_CLAMP" == "1" || "$EGO_ACTION_CLAMP" == "true" || "$EGO_ACTION_CLAMP" == "TRUE" || "$EGO_ACTION_CLAMP" == "yes" || "$EGO_ACTION_CLAMP" == "YES" ]]; then
  eval_args+=(--ego_action_clamp)
else
  eval_args+=(--no-ego_action_clamp)
fi

"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" "${eval_args[@]}"

echo "===== Slurm eval end: $(date) ====="
