#!/bin/bash
#SBATCH --job-name=ooi50k_raw_fde_noamp
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
LAUNCH_SCRIPT="$REPO_ROOT/waymo/training/launch_ooi50k_lat16_d256_ep200_2a100_staticmap_v2_chunk32_trajloss_randstart_tmux.sh"

mkdir -p "$REPO_ROOT/waymo/logs"

echo "===== Slurm job start: $(date) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_GPUS=${SLURM_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

cd "$REPO_ROOT"

export REPO_ROOT
export DATA_DIR="$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k"
export RUN_NAME="${RUN_NAME:-ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_fdeloss_focus_randstart_noamp}"
export SESSION_NAME="${SESSION_NAME:-ooi50k_staticmap_v2_chunk32_raw_fdeloss_focus_randstart_noamp}"
export RUN_INSIDE_TMUX=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NO_AMP="${NO_AMP:-1}"
export LR="${LR:-3e-4}"
export GRAD_CLIP="${GRAD_CLIP:-1.0}"

export AGENT_DELTA_XY_WEIGHT="${AGENT_DELTA_XY_WEIGHT:-0}"
export AGENT_FDE_XY_WEIGHT="${AGENT_FDE_XY_WEIGHT:-2}"
export AGENT_KINEMATIC_XY_WEIGHT="${AGENT_KINEMATIC_XY_WEIGHT:-0}"
export AGENT_SPEED_YAW_KINEMATIC_WEIGHT="${AGENT_SPEED_YAW_KINEMATIC_WEIGHT:-0}"
export FOCUS_AGENT_WEIGHT="${FOCUS_AGENT_WEIGHT:-4}"

bash "$LAUNCH_SCRIPT"
