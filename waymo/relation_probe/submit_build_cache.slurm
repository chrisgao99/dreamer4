#!/bin/bash
#SBATCH --job-name=rel_probe_cache_z31
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|a100|a40|a6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/relation_probe/%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/relation_probe/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
WAYMO_ROOT="$REPO_ROOT/waymo"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
SCRIPT="$WAYMO_ROOT/relation_probe/build_cache.py"
DATA_ROOT="${DATA_ROOT:-$WAYMO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$WAYMO_ROOT/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt}"
CACHE_DIR="${CACHE_DIR:-$WAYMO_ROOT/cache/relation_probe_z31_current_v0_besttok}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SCENES="${MAX_SCENES:-0}"

mkdir -p "$WAYMO_ROOT/logs/relation_probe" "$CACHE_DIR"
cd "$WAYMO_ROOT/relation_probe"

echo "start cache build: $(date)"
echo "TOKENIZER_CKPT=$TOKENIZER_CKPT"
echo "CACHE_DIR=$CACHE_DIR"

"$PYTHON" "$SCRIPT" \
  --data_dir "$DATA_ROOT/train" \
  --split train \
  --output_dir "$CACHE_DIR" \
  --tokenizer_ckpt "$TOKENIZER_CKPT" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_scenes "$MAX_SCENES"

"$PYTHON" "$SCRIPT" \
  --data_dir "$DATA_ROOT/val" \
  --split val \
  --output_dir "$CACHE_DIR" \
  --tokenizer_ckpt "$TOKENIZER_CKPT" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_scenes "$MAX_SCENES"

echo "end cache build: $(date)"
