#!/bin/bash
#SBATCH --job-name=wm_eval_l32_c11
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
BASE_SCRIPT="$REPO_ROOT/waymo/training/world_model/submit_eval_waymo_world_model_horizons.slurm"

export RUN_NAME="waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_lat32b16_norecon_1m"
export TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat32_b16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
export EVAL_CKPT="$REPO_ROOT/waymo/checkpoints/$RUN_NAME/step_00850000.pt"

export EVAL_CTX="11"
export EVAL_SEQ_LEN="100"
export HORIZONS="10 30 50 80"
export MAX_ROLLOUT_WINDOW="${MAX_ROLLOUT_WINDOW:-11}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/lat32_lat64_step850k/ctx11_seq100}"

export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"

export USE_EGO_ACTIONS="${USE_EGO_ACTIONS:-1}"
export EGO_ACTION_SOURCE="${EGO_ACTION_SOURCE:-focus}"
export EGO_ACTION_NORMALIZATION="${EGO_ACTION_NORMALIZATION:-raw}"
export EGO_ACTION_CLAMP="${EGO_ACTION_CLAMP:-0}"

export AGENT_FAR_WEIGHT="${AGENT_FAR_WEIGHT:-0.25}"
export AGENT_NEAR_RADIUS_M="${AGENT_NEAR_RADIUS_M:-50.0}"
export AGENT_DISTANCE_SOURCE="${AGENT_DISTANCE_SOURCE:-focus}"

bash "$BASE_SCRIPT"
