#!/bin/bash
#SBATCH --job-name=ooi50k_lat16_2gpu
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|a6000|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="/scratch/baz7dy/tri30/dreamer4"
LAUNCH_SCRIPT="$REPO_ROOT/waymo/launch_ooi50k_lat16_d256_ep200_4a100_tmux.sh"

mkdir -p "$REPO_ROOT/waymo/logs"

echo "===== Slurm job start: $(date) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_GPUS=${SLURM_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

cd "$REPO_ROOT"

export REPO_ROOT
export DATA_DIR="$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k"
export RUN_INSIDE_TMUX=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE=2
export BATCH_SIZE="${BATCH_SIZE:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"

bash "$LAUNCH_SCRIPT"
