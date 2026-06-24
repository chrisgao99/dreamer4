#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WAYMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$WAYMO_ROOT/.." && pwd)"

cd "$REPO_ROOT"

export REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
export PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export LIST_FILE="${LIST_FILE:-waymo/evaluation/waymo_fulltraj_random10.txt}"
export OVERFIT_N="${OVERFIT_N:-1}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export EPOCHS="${EPOCHS:-20000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export SAVE_EVERY="${SAVE_EVERY:-500}"

export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NO_AMP="${NO_AMP:-1}"
export RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-0}"

VARIANTS="${VARIANTS:-raw_gmm_kin_fde raw_kin_fde raw_kin_nofde}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/waymo/logs/overfit}"
SUMMARY_LOG="${SUMMARY_LOG:-$LOG_ROOT/true_overfit_three_variants_a40_${STAMP}.log}"

mkdir -p "$LOG_ROOT"

{
  echo "===== true overfit three variants local run ====="
  echo "start_time=$(date)"
  echo "repo_root=$REPO_ROOT"
  echo "python=$PYTHON"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "variants=$VARIANTS"
  echo "list_file=$LIST_FILE"
  echo "overfit_n=$OVERFIT_N"
  echo "batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS"
  echo "max_steps=$MAX_STEPS epochs=$EPOCHS"
  echo "log_every=$LOG_EVERY eval_every=$EVAL_EVERY save_every=$SAVE_EVERY"
  echo "no_amp=$NO_AMP use_wandb=$USE_WANDB wandb_mode=$WANDB_MODE"
  echo "summary_log=$SUMMARY_LOG"
  echo "==============================================="
} | tee "$SUMMARY_LOG"

for variant in $VARIANTS; do
  export VARIANT="$variant"
  export RUN_NAME="${RUN_NAME_PREFIX:-true_overfit}_${variant}_n${OVERFIT_N}_a40_${STAMP}"

  {
    echo
    echo "============================================================"
    echo "Running variant=$VARIANT"
    echo "run_name=$RUN_NAME"
    echo "============================================================"
  } | tee -a "$SUMMARY_LOG"

  bash "$REPO_ROOT/waymo/training/launch_waymo_chunk32_raw_overfit_scenes.sh" 2>&1 | tee -a "$SUMMARY_LOG"
done

{
  echo
  echo "Done."
  echo "end_time=$(date)"
  echo "checkpoint_root=$REPO_ROOT/waymo/checkpoints/overfit"
  echo "log_root=$LOG_ROOT"
} | tee -a "$SUMMARY_LOG"
