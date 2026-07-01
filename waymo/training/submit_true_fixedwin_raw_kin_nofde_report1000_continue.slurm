#!/bin/bash
#SBATCH --job-name=wm_true_fw_n1000c
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a6000|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/sfs/weka/scratch/baz7dy/tri30/dreamer4/waymo/logs/overfit/true_fixedwin/slurm-%x-%j.out
#SBATCH --error=/sfs/weka/scratch/baz7dy/tri30/dreamer4/waymo/logs/overfit/true_fixedwin/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sfs/weka/scratch/baz7dy/tri30/dreamer4}"
RUN_NAME="${RUN_NAME:-true_fixedwin_raw_kin_nofde_report1000_start0_dropout0_wd0}"
LAUNCH_SCRIPT="$REPO_ROOT/waymo/training/run_true_fixedwin_raw_kin_nofde_manifest.sh"

mkdir -p "$REPO_ROOT/waymo/logs/overfit/true_fixedwin"
cd "$REPO_ROOT"

export LIST_FILE="$REPO_ROOT/waymo/evaluation/overfit_fixedwin_manifests/true_fixedwin_report1000.txt"
export RUN_NAME
export RESUME="${RESUME:-$REPO_ROOT/waymo/checkpoints/overfit/true_fixedwin/$RUN_NAME/latest.pt}"
export MAX_STEPS="${MAX_STEPS:-640000}"
export EPOCHS="${EPOCHS:-100000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LOG_EVERY="${LOG_EVERY:-20}"
export EVAL_EVERY="${EVAL_EVERY:-1000}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_WANDB="${USE_WANDB:-0}"

echo "===== Slurm continue job start: $(date) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "MAX_STEPS=$MAX_STEPS"

bash "$LAUNCH_SCRIPT"
