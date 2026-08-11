#!/usr/bin/env bash
# Tokenizer-only GT -> encode -> decode ablation on the fixed val128 manifest.
# Runs three paired variants on physical CUDA 1. Every tokenizer call is <=32 frames.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_tokenizer_chunk_stitch_val.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
SUBSET_MANIFEST="$REPO_ROOT/waymo/evaluation/val_random128_seed0_manifest.json"

CUDA_DEVICE="${CUDA_DEVICE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SESSION_NAME="${SESSION_NAME:-tokenizer_chunk_stitch_val128_cuda1}"
RUN_ID="${RUN_ID:-tokenizer_only_chunk_stitch_val128_seed0}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/tokenizer/$RUN_ID}"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is not available" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" \
    SESSION_NAME="$SESSION_NAME" RUN_ID="$RUN_ID" OUT_DIR="$OUT_DIR" LOG_FILE="$LOG_FILE" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started detached tmux session: $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Results: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) tokenizer-only chunk-stitch val128 ablation start ====="
  echo "physical_cuda=$CUDA_DEVICE tokenizer=$TOKENIZER_CKPT"
  echo "protocol=fixed_val128 GT_encode_decode seq91 score_ctx1 max_tokenizer_chunk32"
  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" \
    --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
    --eval_batch_size 1 --eval_max_batches "$EVAL_MAX_BATCHES" \
    --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
    --eval_seq_len 91 --score_start 1 --score_horizons 10 30 50 80 90 \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --output_dir "$OUT_DIR"
  echo "===== $(date) tokenizer-only chunk-stitch val128 ablation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
