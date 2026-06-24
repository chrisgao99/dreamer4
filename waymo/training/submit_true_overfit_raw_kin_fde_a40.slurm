#!/bin/bash
#SBATCH --job-name=wm_true_raw_kin_fde
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|a6000|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/sfs/weka/scratch/baz7dy/tri30/dreamer4/waymo/logs/overfit/slurm-%x-%j.out
#SBATCH --error=/sfs/weka/scratch/baz7dy/tri30/dreamer4/waymo/logs/overfit/slurm-%x-%j.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/sfs/weka/scratch/baz7dy/tri30/dreamer4}"
export PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
export VARIANT="raw_kin_fde"
export RUN_NAME="${RUN_NAME:-true_overfit_raw_kin_fde_n1_a40_slurm}"
export OVERFIT_N="${OVERFIT_N:-1}"
export LIST_FILE="${LIST_FILE:-waymo/evaluation/waymo_fulltraj_random10.txt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

export MAX_STEPS="${MAX_STEPS:-20000}"
export EPOCHS="${EPOCHS:-20000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export SAVE_EVERY="${SAVE_EVERY:-500}"

export NO_AMP="${NO_AMP:-1}"
export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-0}"

mkdir -p "$REPO_ROOT/waymo/logs/overfit"
cd "$REPO_ROOT"

echo "===== Slurm job start: $(date) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "RUN_NAME=$RUN_NAME"
echo "VARIANT=$VARIANT"
echo "MAX_STEPS=$MAX_STEPS EPOCHS=$EPOCHS"

bash "$REPO_ROOT/waymo/training/launch_waymo_chunk32_raw_overfit_scenes.sh"
