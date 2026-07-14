#!/bin/bash
#SBATCH --job-name=wm_eval_h1_3
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200|a100|a40|a6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
BASE_SCRIPT="$REPO_ROOT/waymo/training/world_model/submit_eval_waymo_world_model_horizons.slurm"

export TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt}"
export HORIZONS="${HORIZONS:-1}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/h1}"

export USE_EGO_ACTIONS="${USE_EGO_ACTIONS:-1}"
export EGO_ACTION_SOURCE="${EGO_ACTION_SOURCE:-focus}"
export EGO_ACTION_NORMALIZATION="${EGO_ACTION_NORMALIZATION:-raw}"
export EGO_ACTION_CLAMP="${EGO_ACTION_CLAMP:-0}"

export AGENT_FAR_WEIGHT="${AGENT_FAR_WEIGHT:-0.25}"
export AGENT_NEAR_RADIUS_M="${AGENT_NEAR_RADIUS_M:-50.0}"
export AGENT_DISTANCE_SOURCE="${AGENT_DISTANCE_SOURCE:-focus}"

run_eval() {
  local label="$1"
  local run_name="$2"
  local eval_ckpt="$3"
  local eval_seq_len="$4"
  local max_rollout_window="$5"

  echo "===== H1 eval: ${label} ====="
  export RUN_NAME="$run_name"
  export EVAL_CKPT="$eval_ckpt"
  export EVAL_SEQ_LEN="$eval_seq_len"
  export EVAL_CTX="${EVAL_CTX:-1}"
  export MAX_ROLLOUT_WINDOW="$max_rollout_window"

  bash "$BASE_SCRIPT"
}

run_eval \
  "window10_fixed" \
  "waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_fixedstart_b2_norecon_1m" \
  "$REPO_ROOT/waymo/checkpoints/waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_fixedstart_b2_norecon_1m/step_00200000.pt" \
  "11" \
  "11"

run_eval \
  "window10_rand" \
  "waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_b2_norecon_1m" \
  "$REPO_ROOT/waymo/checkpoints/waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_b2_norecon_1m/step_00850000.pt" \
  "11" \
  "11"

run_eval \
  "window90" \
  "waymo_wm_v1_egoact_focus_raw_noclamp_full91_ctx1_h90_norecon_1m" \
  "$REPO_ROOT/waymo/checkpoints/waymo_wm_v1_egoact_focus_raw_noclamp_full91_ctx1_h90_norecon_1m/step_00200000.pt" \
  "91" \
  "91"
