#!/bin/bash
#SBATCH --job-name=wm_raw11_b2_t1best
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200|a100|a40|a6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.err

set -euo pipefail

BASE_SCRIPT="${BASE_SCRIPT:-/scratch/baz7dy/tri30/dreamer4/waymo/training/world_model/submit_waymo_world_model_v0_lat64_b64_win100.slurm}"

export RUN_NAME="${RUN_NAME:-waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_b2_norecon_time1_besttok_1m}"
export TOKENIZER_CKPT="${TOKENIZER_CKPT:-/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt}"
export MAX_STEPS="${MAX_STEPS:-1000000}"
export EVAL_EVERY="${EVAL_EVERY:-0}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
export SAVE_EVERY="${SAVE_EVERY:-50000}"

export SEQ_LEN="${SEQ_LEN:-11}"
export RANDOM_TIME_WINDOW_START="${RANDOM_TIME_WINDOW_START:-1}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-11}"
export EVAL_CTX="${EVAL_CTX:-1}"
export EVAL_HORIZON="${EVAL_HORIZON:-10}"
export MAX_ROLLOUT_WINDOW="${MAX_ROLLOUT_WINDOW:-11}"
export BATCH_SIZE="${BATCH_SIZE:-2}"

export TIME_EVERY="${TIME_EVERY:-1}"

export USE_EGO_ACTIONS="${USE_EGO_ACTIONS:-1}"
export EGO_ACTION_SOURCE="${EGO_ACTION_SOURCE:-focus}"
export EGO_ACTION_NORMALIZATION="${EGO_ACTION_NORMALIZATION:-raw}"
export EGO_ACTION_CLAMP="${EGO_ACTION_CLAMP:-0}"

export AGENT_FAR_WEIGHT="${AGENT_FAR_WEIGHT:-0.25}"
export AGENT_NEAR_RADIUS_M="${AGENT_NEAR_RADIUS_M:-50.0}"
export AGENT_DISTANCE_SOURCE="${AGENT_DISTANCE_SOURCE:-focus}"
export TRAIN_DECODED_LOSS_WEIGHT="${TRAIN_DECODED_LOSS_WEIGHT:-0.0}"

"$BASE_SCRIPT"
