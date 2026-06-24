#!/bin/bash
#SBATCH --job-name=wm_overfit
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|a6000|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/overfit/slurm-%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/overfit/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
LAUNCH_SCRIPT="$REPO_ROOT/waymo/training/launch_waymo_chunk32_raw_overfit_scenes.sh"

mkdir -p "$REPO_ROOT/waymo/logs/overfit"

echo "===== Slurm job start: $(date) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "VARIANT=${VARIANT:-raw_gmm_kin_fde}"
echo "OVERFIT_N=${OVERFIT_N:-1}"
echo "MAX_STEPS=${MAX_STEPS:-20000}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

cd "$REPO_ROOT"

export REPO_ROOT
export VARIANT="${VARIANT:-raw_gmm_kin_fde}"
export OVERFIT_N="${OVERFIT_N:-1}"
export LIST_FILE="${LIST_FILE:-waymo/evaluation/waymo_fulltraj_random10.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_WANDB="${USE_WANDB:-0}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export EPOCHS="${EPOCHS:-20000}"
export LOG_EVERY="${LOG_EVERY:-20}"
export EVAL_EVERY="${EVAL_EVERY:-500}"
export SAVE_EVERY="${SAVE_EVERY:-500}"

bash "$LAUNCH_SCRIPT"
