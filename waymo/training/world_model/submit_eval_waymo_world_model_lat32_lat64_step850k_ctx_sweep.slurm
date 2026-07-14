#!/bin/bash
#SBATCH --job-name=wm_eval_latctx
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200|a100|a40|a6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%A_%a.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
BASE_SCRIPT="$REPO_ROOT/waymo/training/world_model/submit_eval_waymo_world_model_horizons.slurm"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -gt 3 ]]; then
  echo "SLURM_ARRAY_TASK_ID must be in [0, 3], got: $TASK_ID" >&2
  exit 1
fi

model_idx=$((TASK_ID / 2))
config_idx=$((TASK_ID % 2))

RUN_NAMES=(
  "waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_lat32b16_norecon_1m"
  "waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_lat64b32_norecon_1m"
)

TOKENIZER_CKPTS=(
  "$REPO_ROOT/waymo/checkpoints/ooi50k_lat32_b16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
  "$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b32_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
)

EVAL_CKPTS=(
  "$REPO_ROOT/waymo/checkpoints/waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_lat32b16_norecon_1m/step_00850000.pt"
  "$REPO_ROOT/waymo/checkpoints/waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_lat64b32_norecon_1m/step_00850000.pt"
)

CONFIG_LABELS=(
  "ctx1_seq91"
  "ctx11_seq100"
)

EVAL_CTXS=(1 11)
EVAL_SEQ_LENS=(91 100)
HORIZON_SETS=(
  "10 30 50 80 90"
  "10 30 50 80"
)

export RUN_NAME="${RUN_NAMES[$model_idx]}"
export TOKENIZER_CKPT="${TOKENIZER_CKPTS[$model_idx]}"
export EVAL_CKPT="${EVAL_CKPTS[$model_idx]}"
export HORIZONS="${HORIZON_SETS[$config_idx]}"
export EVAL_CTX="${EVAL_CTXS[$config_idx]}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LENS[$config_idx]}"

export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export MAX_ROLLOUT_WINDOW="${MAX_ROLLOUT_WINDOW:-11}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/lat32_lat64_step850k/${CONFIG_LABELS[$config_idx]}}"

export USE_EGO_ACTIONS="${USE_EGO_ACTIONS:-1}"
export EGO_ACTION_SOURCE="${EGO_ACTION_SOURCE:-focus}"
export EGO_ACTION_NORMALIZATION="${EGO_ACTION_NORMALIZATION:-raw}"
export EGO_ACTION_CLAMP="${EGO_ACTION_CLAMP:-0}"

export AGENT_FAR_WEIGHT="${AGENT_FAR_WEIGHT:-0.25}"
export AGENT_NEAR_RADIUS_M="${AGENT_NEAR_RADIUS_M:-50.0}"
export AGENT_DISTANCE_SOURCE="${AGENT_DISTANCE_SOURCE:-focus}"

echo "===== Latent ctx sweep task ====="
echo "TASK_ID=$TASK_ID model_idx=$model_idx config_idx=$config_idx"
echo "RUN_NAME=$RUN_NAME"
echo "TOKENIZER_CKPT=$TOKENIZER_CKPT"
echo "EVAL_CKPT=$EVAL_CKPT"
echo "EVAL_CTX=$EVAL_CTX EVAL_SEQ_LEN=$EVAL_SEQ_LEN HORIZONS=$HORIZONS"
echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE EVAL_MAX_BATCHES=$EVAL_MAX_BATCHES"
echo "MAX_ROLLOUT_WINDOW=$MAX_ROLLOUT_WINDOW"
echo "OUT_DIR=$OUT_DIR"

bash "$BASE_SCRIPT"
