#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
SLURM_SCRIPT="$REPO_ROOT/waymo/training/submit_waymo_chunk32_raw_overfit_one.slurm"

VARIANTS="${VARIANTS:-raw_gmm_kin_fde raw_kin_fde raw_kin_nofde}"
SCENE_COUNTS="${SCENE_COUNTS:-1 5 10}"
MAX_STEPS="${MAX_STEPS:-20000}"
LIST_FILE="${LIST_FILE:-waymo/evaluation/waymo_fulltraj_random10.txt}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_WANDB="${USE_WANDB:-0}"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Missing slurm script: $SLURM_SCRIPT" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/waymo/logs/overfit"

for variant in $VARIANTS; do
  for n in $SCENE_COUNTS; do
    job_name="wm_of_${variant}_n${n}"
    echo "Submitting $job_name"
    sbatch \
      --job-name "$job_name" \
      --export "ALL,REPO_ROOT=$REPO_ROOT,VARIANT=$variant,OVERFIT_N=$n,MAX_STEPS=$MAX_STEPS,LIST_FILE=$LIST_FILE,WANDB_MODE=$WANDB_MODE,USE_WANDB=$USE_WANDB" \
      "$SLURM_SCRIPT"
  done
done
