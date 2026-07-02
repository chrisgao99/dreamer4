#!/bin/bash
#SBATCH --job-name=rel_probe_z31
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|a100|a40|a6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/relation_probe/%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/relation_probe/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
WAYMO_ROOT="$REPO_ROOT/waymo"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
SCRIPT="$WAYMO_ROOT/relation_probe/train_probe.py"
CACHE_DIR="${CACHE_DIR:-$WAYMO_ROOT/cache/relation_probe_z31_current_v0_besttok}"
MODE="${MODE:-raw_only}"
RUN_ROOT="${RUN_ROOT:-$WAYMO_ROOT/checkpoints/relation_probe_z31_current_v0_besttok}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$MODE}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8192}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_STEPS="${MAX_STEPS:-20000}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-1000}"

mkdir -p "$WAYMO_ROOT/logs/relation_probe" "$RUN_DIR"
cd "$WAYMO_ROOT/relation_probe"

echo "start train probe: $(date)"
echo "MODE=$MODE"
echo "CACHE_DIR=$CACHE_DIR"
echo "RUN_DIR=$RUN_DIR"

"$PYTHON" "$SCRIPT" \
  --train_cache "$CACHE_DIR/train_cache.npz" \
  --val_cache "$CACHE_DIR/val_cache.npz" \
  --run_dir "$RUN_DIR" \
  --mode "$MODE" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_steps "$MAX_STEPS" \
  --log_every "$LOG_EVERY" \
  --eval_every "$EVAL_EVERY"

echo "end train probe: $(date)"
