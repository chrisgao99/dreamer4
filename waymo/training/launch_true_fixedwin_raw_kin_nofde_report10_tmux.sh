#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
LAUNCH_SCRIPT="$REPO_ROOT/waymo/training/run_true_fixedwin_raw_kin_nofde_manifest.sh"

export LIST_FILE="${LIST_FILE:-$REPO_ROOT/waymo/evaluation/overfit_fixedwin_manifests/true_fixedwin_report10.txt}"
export RUN_NAME="${RUN_NAME:-true_fixedwin_raw_kin_nofde_report10_start0_dropout0_wd0}"
export SESSION_NAME="${SESSION_NAME:-true_fixedwin_raw_kin_nofde_report10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BATCH_SIZE="${BATCH_SIZE:-10}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export EPOCHS="${EPOCHS:-20000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_WANDB="${USE_WANDB:-0}"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Run inside an existing tmux, or install/load tmux first." >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
  fi
  tmux new-session -d -s "$SESSION_NAME" "RUN_INSIDE_TMUX=1 bash '$0'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  echo "Detach with: Ctrl-b then d"
  echo "Log: $REPO_ROOT/waymo/logs/overfit/true_fixedwin/${RUN_NAME}.log"
  exit 0
fi

cd "$REPO_ROOT"
exec bash "$LAUNCH_SCRIPT"
