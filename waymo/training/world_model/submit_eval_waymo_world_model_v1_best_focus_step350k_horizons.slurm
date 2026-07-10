#!/bin/bash
#SBATCH --job-name=wm_eval_v1best
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

export RUN_NAME="${RUN_NAME:-waymo_wm_v1_egoact_focus_farw025_dec01_lat64_b64_win100_ctx11_h80_1m}"
export EVAL_CKPT="${EVAL_CKPT:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME/step_00350000.pt}"
export HORIZONS="${HORIZONS:-10 30 50 80}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"

export USE_EGO_ACTIONS="${USE_EGO_ACTIONS:-1}"
export EGO_ACTION_SOURCE="${EGO_ACTION_SOURCE:-focus}"
export AGENT_FAR_WEIGHT="${AGENT_FAR_WEIGHT:-0.25}"
export AGENT_NEAR_RADIUS_M="${AGENT_NEAR_RADIUS_M:-50.0}"
export AGENT_DISTANCE_SOURCE="${AGENT_DISTANCE_SOURCE:-focus}"

bash "$BASE_SCRIPT"
